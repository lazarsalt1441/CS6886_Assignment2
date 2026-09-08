
import csv
import os

import torch
import torch.nn as nn

from baseline.model import get_model
from baseline.data import get_dataloaders
from baseline.evaluate import evaluate
from baseline.config import DEVICE, MODEL_PATH

from compression.quantization import symmetric_quantize_per_channel
from compression.activation_quantization import symmetric_quantize_activation


# ============================================================
# CONFIG
# ============================================================

SENSITIVITY_CSV = "layer_sensitivity.csv"

BASE_BITS = 7
ACTIVATION_BITS = 8

# Full-test-set constraint for the final mixed model.
TARGET_MIN_ACCURACY = 94.0

# Only consider individually safe downgrades from the profiler.
# These thresholds are deliberately permissive enough to include the
# large robust layers we observed, while excluding clearly fragile layers.
MAX_PROFILE_DROP_W5 = 0.10
MAX_PROFILE_DROP_W6 = 0.05

OUTPUT_ASSIGNMENT_CSV = "mixed_precision_assignment.csv"
OUTPUT_SEARCH_LOG_CSV = "mixed_precision_search_log.csv"


# ============================================================
# MODEL HELPERS
# ============================================================

def load_clean_model():
    """
    Reload the original trained FP32 checkpoint.
    Every candidate is evaluated from the same clean model.
    """
    model = get_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model, checkpoint


def get_quantizable_layers(model):
    """
    Conv2d and Linear layers whose weights are quantized.
    """
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
        and module.weight is not None
    }


def quantize_module_weight(module, bits):
    weight = module.weight.data

    quantized_weight, scales, q = (
        symmetric_quantize_per_channel(
            weight,
            bits=bits
        )
    )

    module.weight.data.copy_(
        quantized_weight
    )


def apply_assignment(model, assignment):
    """
    assignment:
        dict[layer_name] = weight_bits

    Every Conv2d / Linear layer is quantized according to assignment.
    """
    layers = get_quantizable_layers(model)

    for name, module in layers.items():

        bits = assignment[name]

        quantize_module_weight(
            module,
            bits=bits
        )


# ============================================================
# ACTIVATION QUANTIZATION
# ============================================================

def make_activation_hook(bits):

    def hook(module, inputs, output):

        if isinstance(output, torch.Tensor):
            return symmetric_quantize_activation(
                output,
                bits=bits
            )

        return output

    return hook


def add_activation_hooks(model):
    handles = []

    for module in get_quantizable_layers(model).values():

        handles.append(
            module.register_forward_hook(
                make_activation_hook(
                    ACTIVATION_BITS
                )
            )
        )

    return handles


# ============================================================
# EVALUATION
# ============================================================

def evaluate_assignment(
    assignment,
    test_loader,
    criterion
):
    """
    Evaluate a bit assignment on the FULL test set.
    """
    model, _ = load_clean_model()

    apply_assignment(
        model,
        assignment
    )

    hooks = add_activation_hooks(
        model
    )

    loss, accuracy = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE
    )

    for handle in hooks:
        handle.remove()

    del model

    return loss, accuracy


# ============================================================
# SENSITIVITY CSV
# ============================================================

def load_sensitivity():
    rows = []

    with open(
        SENSITIVITY_CSV,
        "r",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            row["num_weights"] = int(
                row["num_weights"]
            )

            row["trial_bits"] = int(
                row["trial_bits"]
            )

            row["accuracy_drop_pp"] = float(
                row["accuracy_drop_pp"]
            )

            row["saved_kb_vs_base"] = float(
                row["saved_kb_vs_base"]
            )

            rows.append(row)

    return rows


# ============================================================
# STORAGE
# ============================================================

def calculate_dense_mixed_precision_storage(
    assignment,
    layer_sizes,
    total_scales,
    total_params
):
    """
    Dense mixed-precision representation.

    Each layer l uses:
        N_l * b_l bits

    Plus:
        32 bits per per-channel scale
        FP32 storage for non-quantized parameters
    """

    quantized_weight_count = sum(
        layer_sizes.values()
    )

    weight_value_bits = 0

    for layer_name, num_weights in layer_sizes.items():

        bits = assignment[layer_name]

        weight_value_bits += (
            num_weights * bits
        )

    scale_bits = (
        total_scales * 32
    )

    fp32_exception_params = (
        total_params
        - quantized_weight_count
    )

    fp32_exception_bits = (
        fp32_exception_params * 32
    )

    compressed_bits = (
        weight_value_bits
        + scale_bits
        + fp32_exception_bits
    )

    original_bits = (
        total_params * 32
    )

    return {
        "compressed_mb":
            compressed_bits
            / 8
            / (1024 ** 2),

        "compression_ratio":
            original_bits
            / compressed_bits,

        "weight_value_mb":
            weight_value_bits
            / 8
            / (1024 ** 2),

        "scale_overhead_mb":
            scale_bits
            / 8
            / (1024 ** 2),

        "fp32_exception_mb":
            fp32_exception_bits
            / 8
            / (1024 ** 2),
    }


# ============================================================
# LOGGING
# ============================================================

def append_search_log(row):

    exists = os.path.exists(
        OUTPUT_SEARCH_LOG_CSV
    )

    with open(
        OUTPUT_SEARCH_LOG_CSV,
        "a",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=row.keys()
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def save_assignment(
    assignment,
    layer_sizes
):

    with open(
        OUTPUT_ASSIGNMENT_CSV,
        "w",
        newline=""
    ) as f:

        fieldnames = [
            "layer_name",
            "assigned_bits",
            "num_weights"
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for name, bits in assignment.items():

            writer.writerow(
                {
                    "layer_name":
                        name,

                    "assigned_bits":
                        bits,

                    "num_weights":
                        layer_sizes[name]
                }
            )


# ============================================================
# CANDIDATE BUILDING
# ============================================================

def build_candidates(
    sensitivity_rows,
    assignment
):
    """
    Build safe layer downgrade proposals.

    Main strategy:
      1. Prefer W5 for large layers that were individually robust.
      2. Then consider W6 for layers that could not safely go to W5.

    The candidates are sorted primarily by ABSOLUTE storage saved,
    not by the old KB/pp score. This matches the measured CSV better.
    """

    candidates = []

    # --------------------------------------------------------
    # W5 candidates
    # --------------------------------------------------------

    for row in sensitivity_rows:

        if row["trial_bits"] != 5:
            continue

        if (
            row["accuracy_drop_pp"]
            > MAX_PROFILE_DROP_W5
        ):
            continue

        candidates.append(
            {
                "layer_name":
                    row["layer_name"],

                "target_bits":
                    5,

                "profile_drop_pp":
                    row["accuracy_drop_pp"],

                "num_weights":
                    row["num_weights"],

                "saved_kb_from_w7":
                    row["saved_kb_vs_base"],

                "priority":
                    0
            }
        )

    # --------------------------------------------------------
    # W6 candidates
    # --------------------------------------------------------

    for row in sensitivity_rows:

        if row["trial_bits"] != 6:
            continue

        if (
            row["accuracy_drop_pp"]
            > MAX_PROFILE_DROP_W6
        ):
            continue

        candidates.append(
            {
                "layer_name":
                    row["layer_name"],

                "target_bits":
                    6,

                "profile_drop_pp":
                    row["accuracy_drop_pp"],

                "num_weights":
                    row["num_weights"],

                "saved_kb_from_w7":
                    row["saved_kb_vs_base"],

                "priority":
                    1
            }
        )

    # --------------------------------------------------------
    # Sort:
    #
    # W5 robust candidates first, because they offer 2 bits/weight.
    # Within each class, try biggest storage savings first.
    #
    # Negative profile drops are treated as "very robust", but
    # we do not interpret them as true accuracy improvements.
    # --------------------------------------------------------

    candidates.sort(
        key=lambda c: (
            c["priority"],
            -c["saved_kb_from_w7"],
            max(c["profile_drop_pp"], 0.0)
        )
    )

    return candidates


# ============================================================
# MAIN GREEDY SEARCH
# ============================================================

def main():

    print("=" * 80)
    print("Measured-sensitivity mixed-precision search")
    print("=" * 80)

    print(
        f"Device:               {DEVICE}"
    )

    print(
        f"Starting weights:     W{BASE_BITS}"
    )

    print(
        f"Activation precision: A{ACTIVATION_BITS}"
    )

    print(
        f"Accuracy floor:       "
        f"{TARGET_MIN_ACCURACY:.2f}%"
    )

    print(
        f"W5 profile threshold: "
        f"<= {MAX_PROFILE_DROP_W5:.2f} pp"
    )

    print(
        f"W6 profile threshold: "
        f"<= {MAX_PROFILE_DROP_W6:.2f} pp"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # Full test loader
    # --------------------------------------------------------

    _, test_loader = (
        get_dataloaders()
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    # --------------------------------------------------------
    # Model metadata
    # --------------------------------------------------------

    clean_model, checkpoint = (
        load_clean_model()
    )

    layers = (
        get_quantizable_layers(
            clean_model
        )
    )

    layer_sizes = {
        name:
            module.weight.numel()

        for name, module
        in layers.items()
    }

    total_scales = sum(
        module.weight.shape[0]
        for module in layers.values()
    )

    total_params = sum(
        p.numel()
        for p in clean_model.parameters()
    )

    del clean_model

    # --------------------------------------------------------
    # Begin with uniform W7
    # --------------------------------------------------------

    assignment = {
        name: BASE_BITS
        for name in layer_sizes
    }

    _, base_accuracy = (
        evaluate_assignment(
            assignment,
            test_loader,
            criterion
        )
    )

    print()
    print(
        f"Full-test uniform "
        f"W{BASE_BITS}A{ACTIVATION_BITS} "
        f"accuracy: "
        f"{base_accuracy:.2f}%"
    )

    if (
        base_accuracy
        < TARGET_MIN_ACCURACY
    ):

        raise RuntimeError(
            "Uniform starting model is already "
            "below TARGET_MIN_ACCURACY."
        )

    # --------------------------------------------------------
    # Candidate list
    # --------------------------------------------------------

    sensitivity_rows = (
        load_sensitivity()
    )

    candidates = (
        build_candidates(
            sensitivity_rows,
            assignment
        )
    )

    print()
    print(
        f"Candidate downgrades: "
        f"{len(candidates)}"
    )

    print()

    # Avoid trying W6 after we already accepted W5 for same layer.
    accepted_w5_layers = set()

    accepted_count = 0

    # --------------------------------------------------------
    # Greedy full-test search
    # --------------------------------------------------------

    for index, candidate in enumerate(
        candidates,
        start=1
    ):

        layer_name = (
            candidate["layer_name"]
        )

        target_bits = (
            candidate["target_bits"]
        )

        current_bits = (
            assignment[layer_name]
        )

        # Already at equal/lower precision.
        if target_bits >= current_bits:
            continue

        # If W5 already accepted, W6 proposal is irrelevant.
        if (
            layer_name
            in accepted_w5_layers
        ):
            continue

        print("-" * 80)

        print(
            f"[{index}/{len(candidates)}] "
            f"{layer_name}"
        )

        print(
            f"Proposal: "
            f"W{current_bits} -> W{target_bits}"
        )

        print(
            f"Parameters: "
            f"{candidate['num_weights']:,}"
        )

        print(
            f"Profile drop: "
            f"{candidate['profile_drop_pp']:.3f} pp"
        )

        print(
            f"Nominal saving vs W7: "
            f"{candidate['saved_kb_from_w7']:.2f} KB"
        )

        # -----------------------------------------------
        # Evaluate proposal jointly with all previously
        # accepted downgrades.
        # -----------------------------------------------

        proposed_assignment = (
            assignment.copy()
        )

        proposed_assignment[
            layer_name
        ] = target_bits

        _, proposed_accuracy = (
            evaluate_assignment(
                proposed_assignment,
                test_loader,
                criterion
            )
        )

        accepted = (
            proposed_accuracy
            >= TARGET_MIN_ACCURACY
        )

        if accepted:

            assignment = (
                proposed_assignment
            )

            accepted_count += 1

            if target_bits == 5:
                accepted_w5_layers.add(
                    layer_name
                )

            status = "ACCEPTED"

        else:
            status = "REJECTED"

        print(
            f"Full-test accuracy: "
            f"{proposed_accuracy:.2f}%"
        )

        print(status)

        # -----------------------------------------------
        # Current model storage after this decision
        # -----------------------------------------------

        current_storage = (
            calculate_dense_mixed_precision_storage(
                assignment,
                layer_sizes,
                total_scales,
                total_params
            )
        )

        append_search_log(
            {
                "candidate_index":
                    index,

                "layer_name":
                    layer_name,

                "previous_bits":
                    current_bits,

                "proposed_bits":
                    target_bits,

                "profile_drop_pp":
                    candidate[
                        "profile_drop_pp"
                    ],

                "num_weights":
                    candidate[
                        "num_weights"
                    ],

                "proposed_accuracy":
                    proposed_accuracy,

                "accepted":
                    int(accepted),

                "current_model_size_mb":
                    current_storage[
                        "compressed_mb"
                    ],

                "current_compression_ratio":
                    current_storage[
                        "compression_ratio"
                    ]
            }
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    final_loss, final_accuracy = (
        evaluate_assignment(
            assignment,
            test_loader,
            criterion
        )
    )

    final_storage = (
        calculate_dense_mixed_precision_storage(
            assignment,
            layer_sizes,
            total_scales,
            total_params
        )
    )

    save_assignment(
        assignment,
        layer_sizes
    )

    # Weighted bit statistics
    total_quantized_weights = sum(
        layer_sizes.values()
    )

    weighted_bits = sum(
        layer_sizes[name]
        * assignment[name]
        for name in assignment
    )

    average_weight_bits = (
        weighted_bits
        / total_quantized_weights
    )

    layer_histogram = {}

    weight_histogram = {}

    for name, bits in assignment.items():

        layer_histogram[bits] = (
            layer_histogram.get(
                bits,
                0
            )
            + 1
        )

        weight_histogram[bits] = (
            weight_histogram.get(
                bits,
                0
            )
            + layer_sizes[name]
        )

    print()
    print("=" * 80)
    print("FINAL MIXED-PRECISION RESULT")
    print("=" * 80)

    print(
        f"Baseline W7A8 accuracy: "
        f"{base_accuracy:.2f}%"
    )

    print(
        f"Final accuracy:         "
        f"{final_accuracy:.2f}%"
    )

    print(
        f"Accuracy drop vs W7A8: "
        f"{base_accuracy - final_accuracy:.2f} pp"
    )

    print(
        f"Accepted downgrades:    "
        f"{accepted_count}"
    )

    print(
        f"Average weight bits:    "
        f"{average_weight_bits:.3f}"
    )

    print(
        f"Model size:             "
        f"{final_storage['compressed_mb']:.4f} MB"
    )

    print(
        f"Model compression:      "
        f"{final_storage['compression_ratio']:.4f}x"
    )

    print(
        f"Weight-value storage:   "
        f"{final_storage['weight_value_mb']:.4f} MB"
    )

    print(
        f"Scale overhead:         "
        f"{final_storage['scale_overhead_mb']:.4f} MB"
    )

    print(
        f"FP32 exceptions:        "
        f"{final_storage['fp32_exception_mb']:.4f} MB"
    )

    print(
        f"Layer bit histogram:    "
        f"{layer_histogram}"
    )

    print(
        f"Weight bit histogram:   "
        f"{weight_histogram}"
    )

    print(
        f"Assignment written to:  "
        f"{OUTPUT_ASSIGNMENT_CSV}"
    )

    print(
        f"Search log written to:  "
        f"{OUTPUT_SEARCH_LOG_CSV}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()

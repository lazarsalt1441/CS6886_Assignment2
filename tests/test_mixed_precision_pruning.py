
import csv
import os

import torch
import torch.nn as nn

from baseline.model import get_model
from baseline.data import get_dataloaders
from baseline.evaluate import evaluate
from baseline.config import DEVICE, MODEL_PATH

from compression.pruning import global_magnitude_prune
from compression.quantization import symmetric_quantize_per_channel
from compression.activation_quantization import symmetric_quantize_activation


# ============================================================
# CONFIG
# ============================================================

ASSIGNMENT_CSV = "mixed_precision_assignment.csv"

ACTIVATION_BITS = 8

PRUNE_RATIOS = [
    0.00,
    0.10,
    0.15,
    0.20,
]

OUTPUT_CSV = "mixed_precision_pruning_results.csv"


# ============================================================
# HELPERS
# ============================================================

def bits_to_mb(bits):
    return bits / 8 / (1024 ** 2)


def load_clean_model():
    """
    Reload original FP32 checkpoint so every experiment starts
    from exactly the same baseline.
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
    Conv2d / Linear layers to which mixed-precision weight
    quantization is applied.
    """

    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
        and module.weight is not None
    }


# ============================================================
# LOAD MIXED-PRECISION ASSIGNMENT
# ============================================================

def load_assignment():
    """
    Reads:
        layer_name, assigned_bits, num_weights
    from mixed_precision_assignment.csv
    """

    assignment = {}

    with open(
        ASSIGNMENT_CSV,
        "r",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            assignment[
                row["layer_name"]
            ] = int(
                row["assigned_bits"]
            )

    return assignment


# ============================================================
# MIXED-PRECISION WEIGHT QUANTIZATION
# ============================================================

def apply_mixed_precision_quantization(
    model,
    assignment
):
    """
    Quantizes every Conv2d / Linear layer using the bit-width
    stored in mixed_precision_assignment.csv.

    Returns detailed storage statistics per layer.
    """

    layer_stats = {}

    layers = get_quantizable_layers(
        model
    )

    missing = (
        set(layers.keys())
        - set(assignment.keys())
    )

    if missing:
        raise RuntimeError(
            "Assignment CSV is missing layers:\n"
            + "\n".join(
                sorted(missing)
            )
        )

    for name, module in layers.items():

        bits = assignment[name]

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

        total_weights = (
            weight.numel()
        )

        nonzero_weights = (
            quantized_weight != 0
        ).sum().item()

        layer_stats[name] = {
            "bits":
                bits,

            "total_weights":
                total_weights,

            "nonzero_weights":
                nonzero_weights,

            "num_scales":
                scales.numel(),
        }

    return layer_stats


# ============================================================
# ACTIVATION QUANTIZATION
# ============================================================

def make_activation_hook(bits):

    def hook(
        module,
        inputs,
        output
    ):

        if isinstance(
            output,
            torch.Tensor
        ):

            return (
                symmetric_quantize_activation(
                    output,
                    bits=bits
                )
            )

        return output

    return hook


def add_activation_quantization(
    model
):

    handles = []

    for module in (
        get_quantizable_layers(
            model
        ).values()
    ):

        handles.append(
            module.register_forward_hook(
                make_activation_hook(
                    ACTIVATION_BITS
                )
            )
        )

    return handles


# ============================================================
# STORAGE CALCULATION
# ============================================================

def calculate_storage(
    model,
    layer_stats,
    prune_ratio
):
    """
    For prune_ratio == 0:
        Dense mixed-precision representation.

    For prune_ratio > 0:
        Bitmap sparse representation:
            1 mask bit per original eligible weight
            + b_l bits per nonzero quantized weight in layer l
            + FP32 scale metadata
            + FP32 exceptions

    This matches the sparse representation used in the earlier
    pruning experiments.
    """

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    original_bits = (
        total_params * 32
    )

    total_quantized_weights = sum(
        s["total_weights"]
        for s in layer_stats.values()
    )

    total_nonzero_weights = sum(
        s["nonzero_weights"]
        for s in layer_stats.values()
    )

    # --------------------------------------------
    # Mixed-bit weight values
    # --------------------------------------------

    if prune_ratio > 0:

        value_bits = sum(
            s["nonzero_weights"]
            * s["bits"]
            for s in layer_stats.values()
        )

        # One bitmap bit per original eligible weight.
        mask_bits = (
            total_quantized_weights
        )

    else:

        value_bits = sum(
            s["total_weights"]
            * s["bits"]
            for s in layer_stats.values()
        )

        mask_bits = 0

    # --------------------------------------------
    # Per-output-channel FP32 scales
    # --------------------------------------------

    total_scales = sum(
        s["num_scales"]
        for s in layer_stats.values()
    )

    scale_bits = (
        total_scales * 32
    )

    # --------------------------------------------
    # BatchNorm, biases, etc. remain FP32
    # --------------------------------------------

    fp32_exception_params = (
        total_params
        - total_quantized_weights
    )

    fp32_exception_bits = (
        fp32_exception_params * 32
    )

    compressed_bits = (
        value_bits
        + mask_bits
        + scale_bits
        + fp32_exception_bits
    )

    compression_ratio = (
        original_bits
        / compressed_bits
    )

    final_zero_fraction = (
        1
        - total_nonzero_weights
        / total_quantized_weights
    )

    weighted_bit_sum = sum(
        s["total_weights"]
        * s["bits"]
        for s in layer_stats.values()
    )

    average_assigned_bits = (
        weighted_bit_sum
        / total_quantized_weights
    )

    return {
        "original_model_mb":
            bits_to_mb(
                original_bits
            ),

        "compressed_model_mb":
            bits_to_mb(
                compressed_bits
            ),

        "compression_ratio":
            compression_ratio,

        "weight_value_mb":
            bits_to_mb(
                value_bits
            ),

        "mask_overhead_mb":
            bits_to_mb(
                mask_bits
            ),

        "scale_overhead_mb":
            bits_to_mb(
                scale_bits
            ),

        "fp32_exception_mb":
            bits_to_mb(
                fp32_exception_bits
            ),

        "total_quantized_weights":
            total_quantized_weights,

        "nonzero_quantized_weights":
            total_nonzero_weights,

        "final_zero_fraction":
            final_zero_fraction,

        "average_assigned_bits":
            average_assigned_bits,
    }


# ============================================================
# CSV
# ============================================================

def append_result(row):

    exists = os.path.exists(
        OUTPUT_CSV
    )

    with open(
        OUTPUT_CSV,
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


# ============================================================
# ONE EXPERIMENT
# ============================================================

def run_experiment(
    prune_ratio,
    assignment,
    test_loader,
    criterion
):

    print()
    print("=" * 80)

    print(
        f"Mixed precision + "
        f"{prune_ratio:.0%} pruning + "
        f"A{ACTIVATION_BITS}"
    )

    print("=" * 80)

    model, checkpoint = (
        load_clean_model()
    )

    baseline_accuracy = (
        checkpoint[
            "test_accuracy"
        ]
    )

    # --------------------------------------------------------
    # 1. Optional global magnitude pruning
    # --------------------------------------------------------

    if prune_ratio > 0:

        model, prune_stats = (
            global_magnitude_prune(
                model,
                prune_ratio=prune_ratio
            )
        )

        pruning_only_sparsity = (
            prune_stats[
                "actual_sparsity"
            ]
        )

    else:

        pruning_only_sparsity = 0.0

    # --------------------------------------------------------
    # 2. Mixed-precision weight quantization
    # --------------------------------------------------------

    layer_stats = (
        apply_mixed_precision_quantization(
            model,
            assignment
        )
    )

    # --------------------------------------------------------
    # 3. Storage after pruning + mixed precision
    # --------------------------------------------------------

    storage = (
        calculate_storage(
            model,
            layer_stats,
            prune_ratio
        )
    )

    # --------------------------------------------------------
    # 4. Activation quantization
    # --------------------------------------------------------

    hooks = (
        add_activation_quantization(
            model
        )
    )

    # --------------------------------------------------------
    # 5. Full test-set evaluation
    # --------------------------------------------------------

    test_loss, test_accuracy = (
        evaluate(
            model,
            test_loader,
            criterion,
            DEVICE
        )
    )

    for handle in hooks:
        handle.remove()

    accuracy_drop = (
        baseline_accuracy
        - test_accuracy
    )

    result = {
        "prune_ratio":
            prune_ratio,

        "activation_bits":
            ACTIVATION_BITS,

        "baseline_accuracy":
            baseline_accuracy,

        "test_accuracy":
            test_accuracy,

        "accuracy_drop_pp":
            accuracy_drop,

        "test_loss":
            test_loss,

        "average_weight_bits":
            storage[
                "average_assigned_bits"
            ],

        "model_size_mb":
            storage[
                "compressed_model_mb"
            ],

        "model_compression_ratio":
            storage[
                "compression_ratio"
            ],

        "pruning_only_sparsity":
            pruning_only_sparsity,

        "final_zero_fraction":
            storage[
                "final_zero_fraction"
            ],

        "weight_value_mb":
            storage[
                "weight_value_mb"
            ],

        "mask_overhead_mb":
            storage[
                "mask_overhead_mb"
            ],

        "scale_overhead_mb":
            storage[
                "scale_overhead_mb"
            ],

        "fp32_exception_mb":
            storage[
                "fp32_exception_mb"
            ],
    }

    print(
        f"Accuracy:                 "
        f"{test_accuracy:.2f}%"
    )

    print(
        f"Drop from FP32:           "
        f"{accuracy_drop:.2f} pp"
    )

    print(
        f"Average assigned bits:    "
        f"{storage['average_assigned_bits']:.3f}"
    )

    print(
        f"Requested prune ratio:    "
        f"{100 * prune_ratio:.1f}%"
    )

    print(
        f"Pruning-only sparsity:    "
        f"{100 * pruning_only_sparsity:.2f}%"
    )

    print(
        f"Final zero fraction:      "
        f"{100 * storage['final_zero_fraction']:.2f}%"
    )

    print(
        f"Model size:               "
        f"{storage['compressed_model_mb']:.4f} MB"
    )

    print(
        f"Model compression ratio:  "
        f"{storage['compression_ratio']:.4f}x"
    )

    print(
        f"Weight values:            "
        f"{storage['weight_value_mb']:.4f} MB"
    )

    print(
        f"Bitmap overhead:          "
        f"{storage['mask_overhead_mb']:.4f} MB"
    )

    print(
        f"Scale overhead:           "
        f"{storage['scale_overhead_mb']:.4f} MB"
    )

    print(
        f"FP32 exceptions:          "
        f"{storage['fp32_exception_mb']:.4f} MB"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 80)
    print("Mixed-precision + pruning sweep")
    print("=" * 80)

    print(
        f"Device:         {DEVICE}"
    )

    print(
        f"Assignment:     {ASSIGNMENT_CSV}"
    )

    print(
        f"Activation:     A{ACTIVATION_BITS}"
    )

    print(
        f"Prune ratios:   {PRUNE_RATIOS}"
    )

    print("=" * 80)

    assignment = (
        load_assignment()
    )

    _, test_loader = (
        get_dataloaders()
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    # Clear old output CSV so this run is unambiguous.
    if os.path.exists(
        OUTPUT_CSV
    ):
        os.remove(
            OUTPUT_CSV
        )

    results = []

    for prune_ratio in (
        PRUNE_RATIOS
    ):

        result = (
            run_experiment(
                prune_ratio,
                assignment,
                test_loader,
                criterion
            )
        )

        results.append(
            result
        )

        append_result(
            result
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    print(
        f"{'Prune':>8s} "
        f"{'Accuracy':>10s} "
        f"{'Drop':>10s} "
        f"{'Size MB':>10s} "
        f"{'Model CR':>10s} "
        f"{'Zero frac':>10s}"
    )

    print("-" * 100)

    for r in results:

        print(
            f"{100*r['prune_ratio']:7.1f}% "
            f"{r['test_accuracy']:9.2f}% "
            f"{r['accuracy_drop_pp']:9.2f} "
            f"{r['model_size_mb']:10.4f} "
            f"{r['model_compression_ratio']:10.4f} "
            f"{100*r['final_zero_fraction']:9.2f}%"
        )

    print()
    print(
        f"Results written to: "
        f"{OUTPUT_CSV}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()

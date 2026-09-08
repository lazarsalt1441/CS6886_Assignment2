import csv
import os
import time

import torch
import torch.nn as nn
import wandb

from baseline.model import get_model
from baseline.data import get_dataloaders
from baseline.evaluate import evaluate
from baseline.config import DEVICE, MODEL_PATH

from compression.pruning import global_magnitude_prune
from compression.quantization import symmetric_quantize_per_channel
from compression.activation_quantization import symmetric_quantize_activation


# ============================================================
# SWEEP CONFIG
# ============================================================

WEIGHT_BITS_LIST = [5, 6, 7, 8]
ACTIVATION_BITS_LIST = [7, 8]
PRUNE_RATIOS = [0.0, 0.10, 0.20, 0.30]

WANDB_PROJECT = "CS6886-MobileNetV2-Compression"

RESULTS_CSV = "sweep_results.csv"


# ============================================================
# HELPERS
# ============================================================

def bits_to_mb(bits):
    return bits / 8 / (1024 ** 2)


def load_clean_model():
    """
    Always reload the original trained FP32 checkpoint.
    Every experiment therefore starts from exactly the same model.
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


# ============================================================
# WEIGHT QUANTIZATION
# ============================================================

def quantize_model_weights(model, bits):

    total_weights = 0
    nonzero_weights = 0
    total_scales = 0

    for module in model.modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            if module.weight is None:
                continue

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

            total_weights += weight.numel()

            nonzero_weights += (
                quantized_weight != 0
            ).sum().item()

            total_scales += scales.numel()

    return {
        "total_weights": total_weights,
        "nonzero_weights": nonzero_weights,
        "total_scales": total_scales,
    }


# ============================================================
# ACTIVATION QUANTIZATION
# ============================================================

def make_activation_quant_hook(bits):

    def hook(module, inputs, output):

        if isinstance(output, torch.Tensor):

            return symmetric_quantize_activation(
                output,
                bits=bits
            )

        return output

    return hook


def add_activation_quant_hooks(model, bits):

    handles = []

    for module in model.modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            handle = module.register_forward_hook(
                make_activation_quant_hook(bits)
            )

            handles.append(handle)

    return handles


# ============================================================
# ACTIVATION COMPRESSION MEASUREMENT
# ============================================================

class ActivationStats:

    def __init__(self):
        self.original_bits = 0
        self.compressed_bits = 0
        self.elements = 0
        self.tensors = 0


def make_activation_stats_hook(stats, bits):

    def hook(module, inputs, output):

        if not isinstance(output, torch.Tensor):
            return output

        n = output.numel()

        # FP32
        stats.original_bits += n * 32

        # low-bit values
        stats.compressed_bits += n * bits

        # one FP32 scale for this activation tensor
        stats.compressed_bits += 32

        stats.elements += n
        stats.tensors += 1

        return symmetric_quantize_activation(
            output,
            bits=bits
        )

    return hook


def measure_activation_compression(
    model,
    test_loader,
    activation_bits
):

    stats = ActivationStats()

    handles = []

    for module in model.modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            handle = module.register_forward_hook(
                make_activation_stats_hook(
                    stats,
                    activation_bits
                )
            )

            handles.append(handle)

    # Measure ONE inference batch
    images, _ = next(iter(test_loader))

    images = images.to(DEVICE)

    with torch.no_grad():
        _ = model(images)

    for handle in handles:
        handle.remove()

    ratio = (
        stats.original_bits
        / stats.compressed_bits
    )

    return {
        "activation_original_mb":
            bits_to_mb(stats.original_bits),

        "activation_compressed_mb":
            bits_to_mb(stats.compressed_bits),

        "activation_compression_ratio":
            ratio,

        "activation_elements":
            stats.elements,

        "activation_tensors":
            stats.tensors,
    }


# ============================================================
# MODEL STORAGE
# ============================================================

def calculate_model_storage(
    model,
    quant_stats,
    weight_bits,
    prune_ratio
):

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    original_bits = (
        total_params * 32
    )

    total_weights = (
        quant_stats["total_weights"]
    )

    nonzero_weights = (
        quant_stats["nonzero_weights"]
    )

    total_scales = (
        quant_stats["total_scales"]
    )

    # Everything not Conv/Linear weight remains FP32
    fp32_exception_params = (
        total_params - total_weights
    )

    fp32_exception_bits = (
        fp32_exception_params * 32
    )

    scale_bits = (
        total_scales * 32
    )

    # --------------------------------------------------------
    # Dense vs sparse representation
    # --------------------------------------------------------

    if prune_ratio > 0:

        # Sparse representation:
        #
        # 1 bit mask for every original weight
        # +
        # b bits only for nonzero values

        mask_bits = total_weights

        value_bits = (
            nonzero_weights * weight_bits
        )

    else:

        # Dense quantized storage:
        # every weight stored directly

        mask_bits = 0

        value_bits = (
            total_weights * weight_bits
        )

    compressed_bits = (
        value_bits
        + mask_bits
        + scale_bits
        + fp32_exception_bits
    )

    compression_ratio = (
        original_bits / compressed_bits
    )

    sparsity = (
        1
        - nonzero_weights / total_weights
    )

    return {
        "original_model_mb":
            bits_to_mb(original_bits),

        "compressed_model_mb":
            bits_to_mb(compressed_bits),

        "model_compression_ratio":
            compression_ratio,

        "weight_value_mb":
            bits_to_mb(value_bits),

        "mask_overhead_mb":
            bits_to_mb(mask_bits),

        "scale_overhead_mb":
            bits_to_mb(scale_bits),

        "fp32_exception_mb":
            bits_to_mb(fp32_exception_bits),

        "final_sparsity":
            sparsity,
    }


# ============================================================
# ONE EXPERIMENT
# ============================================================

def run_experiment(
    weight_bits,
    activation_bits,
    prune_ratio,
    test_loader,
    criterion
):

    print()
    print("=" * 80)

    print(
        f"W{weight_bits} "
        f"A{activation_bits} "
        f"Prune={prune_ratio:.0%}"
    )

    print("=" * 80)

    start_time = time.time()

    # --------------------------------------------------------
    # Always start from CLEAN FP32 checkpoint
    # --------------------------------------------------------

    model, checkpoint = load_clean_model()

    baseline_accuracy = (
        checkpoint["test_accuracy"]
    )

    # --------------------------------------------------------
    # Pruning
    # --------------------------------------------------------

    if prune_ratio > 0:

        model, prune_stats = (
            global_magnitude_prune(
                model,
                prune_ratio=prune_ratio
            )
        )

    else:

        prune_stats = {
            "actual_sparsity": 0.0
        }

    # --------------------------------------------------------
    # Weight quantization
    # --------------------------------------------------------

    quant_stats = quantize_model_weights(
        model,
        bits=weight_bits
    )

    # --------------------------------------------------------
    # Model storage
    # --------------------------------------------------------

    model_stats = calculate_model_storage(
        model,
        quant_stats,
        weight_bits,
        prune_ratio
    )

    # --------------------------------------------------------
    # Activation compression measurement
    # --------------------------------------------------------

    activation_stats = (
        measure_activation_compression(
            model,
            test_loader,
            activation_bits
        )
    )

    # --------------------------------------------------------
    # Real accuracy with activation quantization
    # --------------------------------------------------------

    activation_hooks = (
        add_activation_quant_hooks(
            model,
            activation_bits
        )
    )

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE
    )

    for handle in activation_hooks:
        handle.remove()

    accuracy_drop = (
        baseline_accuracy
        - test_accuracy
    )

    runtime = time.time() - start_time

    result = {
        "weight_bits": weight_bits,
        "activation_bits": activation_bits,
        "prune_ratio": prune_ratio,

        "baseline_accuracy":
            baseline_accuracy,

        "test_accuracy":
            test_accuracy,

        "accuracy_drop":
            accuracy_drop,

        "test_loss":
            test_loss,

        "model_size_mb":
            model_stats[
                "compressed_model_mb"
            ],

        "original_model_mb":
            model_stats[
                "original_model_mb"
            ],

        "model_compression_ratio":
            model_stats[
                "model_compression_ratio"
            ],

        "activation_compression_ratio":
            activation_stats[
                "activation_compression_ratio"
            ],

        "activation_original_mb":
            activation_stats[
                "activation_original_mb"
            ],

        "activation_compressed_mb":
            activation_stats[
                "activation_compressed_mb"
            ],

        "final_sparsity":
            model_stats[
                "final_sparsity"
            ],

        "weight_value_mb":
            model_stats[
                "weight_value_mb"
            ],

        "mask_overhead_mb":
            model_stats[
                "mask_overhead_mb"
            ],

        "scale_overhead_mb":
            model_stats[
                "scale_overhead_mb"
            ],

        "fp32_exception_mb":
            model_stats[
                "fp32_exception_mb"
            ],

        "runtime_seconds":
            runtime,
    }

    print()
    print(
        f"Accuracy:      "
        f"{test_accuracy:.2f}%"
    )

    print(
        f"Accuracy drop: "
        f"{accuracy_drop:.2f} pp"
    )

    print(
        f"Model size:    "
        f"{result['model_size_mb']:.4f} MB"
    )

    print(
        f"Model CR:      "
        f"{result['model_compression_ratio']:.4f}x"
    )

    print(
        f"Activation CR: "
        f"{result['activation_compression_ratio']:.4f}x"
    )

    print(
        f"Sparsity:      "
        f"{100 * result['final_sparsity']:.2f}%"
    )

    print(
        f"Runtime:       "
        f"{runtime:.1f} sec"
    )

    return result


# ============================================================
# CSV
# ============================================================

def load_completed_configs():

    completed = set()

    if not os.path.exists(RESULTS_CSV):
        return completed

    with open(
        RESULTS_CSV,
        "r",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            key = (
                int(row["weight_bits"]),
                int(row["activation_bits"]),
                round(
                    float(row["prune_ratio"]),
                    4
                )
            )

            completed.add(key)

    return completed


def append_result_csv(result):

    file_exists = os.path.exists(
        RESULTS_CSV
    )

    with open(
        RESULTS_CSV,
        "a",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=result.keys()
        )

        if not file_exists:
            writer.writeheader()

        writer.writerow(result)


# ============================================================
# FULL SWEEP
# ============================================================

def main():

    print(f"Using device: {DEVICE}")

    _, test_loader = get_dataloaders()

    criterion = nn.CrossEntropyLoss()

    completed = load_completed_configs()

    total = (
        len(WEIGHT_BITS_LIST)
        * len(ACTIVATION_BITS_LIST)
        * len(PRUNE_RATIOS)
    )

    experiment_number = 0

    for weight_bits in WEIGHT_BITS_LIST:

        for activation_bits in ACTIVATION_BITS_LIST:

            for prune_ratio in PRUNE_RATIOS:

                experiment_number += 1

                key = (
                    weight_bits,
                    activation_bits,
                    round(prune_ratio, 4)
                )

                # ------------------------------------------------
                # Resume support
                # ------------------------------------------------

                if key in completed:

                    print(
                        f"\nSkipping completed: "
                        f"W{weight_bits} "
                        f"A{activation_bits} "
                        f"P{prune_ratio:.0%}"
                    )

                    continue

                print()
                print(
                    f"Experiment "
                    f"{experiment_number}/{total}"
                )

                run_name = (
                    f"W{weight_bits}"
                    f"A{activation_bits}"
                    f"_P{int(prune_ratio * 100)}"
                )

                # ------------------------------------------------
                # W&B run
                # ------------------------------------------------

                wandb.init(
                    project=WANDB_PROJECT,
                    name=run_name,
                    config={
                        "weight_bits":
                            weight_bits,

                        "activation_bits":
                            activation_bits,

                        "prune_ratio":
                            prune_ratio,

                        "quantization":
                            "symmetric_per_channel_weights",

                        "activation_quantization":
                            "symmetric_per_tensor_dynamic",

                        "pruning":
                            "global_magnitude",

                        "model":
                            "MobileNetV2",

                        "dataset":
                            "CIFAR-10",
                    },
                    reinit=True
                )

                try:

                    result = run_experiment(
                        weight_bits,
                        activation_bits,
                        prune_ratio,
                        test_loader,
                        criterion
                    )

                    wandb.log(result)

                    wandb.summary[
                        "test_accuracy"
                    ] = result[
                        "test_accuracy"
                    ]

                    wandb.summary[
                        "model_compression_ratio"
                    ] = result[
                        "model_compression_ratio"
                    ]

                    wandb.summary[
                        "activation_compression_ratio"
                    ] = result[
                        "activation_compression_ratio"
                    ]

                    append_result_csv(
                        result
                    )

                finally:

                    wandb.finish()

    print()
    print("=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)

    print(
        f"Results saved to: "
        f"{RESULTS_CSV}"
    )


if __name__ == "__main__":
    main()
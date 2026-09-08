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

PRUNE_RATIO = 0.30

WEIGHT_BITS = 6
ACTIVATION_BITS = 8


# ============================================================
# WEIGHT QUANTIZATION
# ============================================================

def quantize_model_weights(model, bits=6):

    total_weights = 0
    zero_weights = 0
    total_scales = 0

    for name, module in model.named_modules():

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

            zero_weights += (
                module.weight.data == 0
            ).sum().item()

            total_scales += scales.numel()

    return {
        "total_weights": total_weights,
        "zero_weights": zero_weights,
        "total_scales": total_scales
    }


# ============================================================
# ACTIVATION QUANTIZATION
# ============================================================

def activation_quant_hook(
    module,
    inputs,
    output
):

    if isinstance(output, torch.Tensor):

        return symmetric_quantize_activation(
            output,
            bits=ACTIVATION_BITS
        )

    return output


def add_activation_quantization(model):

    hooks = []

    for module in model.modules():

        if isinstance(
            module,
            (nn.Conv2d, nn.Linear)
        ):

            handle = (
                module.register_forward_hook(
                    activation_quant_hook
                )
            )

            hooks.append(handle)

    return hooks


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        f"Device:          {DEVICE}"
    )

    print(
        f"Prune ratio:     {PRUNE_RATIO:.0%}"
    )

    print(
        f"Weight bits:     {WEIGHT_BITS}"
    )

    print(
        f"Activation bits: {ACTIVATION_BITS}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load baseline
    # --------------------------------------------------------

    model = get_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    _, test_loader = (
        get_dataloaders()
    )

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # FP32 baseline
    # --------------------------------------------------------

    baseline_loss, baseline_acc = (
        evaluate(
            model,
            test_loader,
            criterion,
            DEVICE
        )
    )

    print()
    print("FP32 baseline")
    print("-" * 70)

    print(
        f"Accuracy: "
        f"{baseline_acc:.2f}%"
    )

    # --------------------------------------------------------
    # Pruning
    # --------------------------------------------------------

    print()
    print("Applying global magnitude pruning...")

    model, prune_stats = (
        global_magnitude_prune(
            model,
            prune_ratio=PRUNE_RATIO
        )
    )

    print(
        f"Total eligible weights: "
        f"{prune_stats['total_weights']:,}"
    )

    print(
        f"Pruned weights: "
        f"{prune_stats['pruned_weights']:,}"
    )

    print(
        f"Remaining weights: "
        f"{prune_stats['remaining_weights']:,}"
    )

    print(
        f"Actual sparsity: "
        f"{100 * prune_stats['actual_sparsity']:.2f}%"
    )

    print(
        f"Threshold: "
        f"{prune_stats['threshold']:.6e}"
    )

    # --------------------------------------------------------
    # Evaluate pruning alone
    # --------------------------------------------------------

    prune_loss, prune_acc = (
        evaluate(
            model,
            test_loader,
            criterion,
            DEVICE
        )
    )

    print()
    print("After pruning only")
    print("-" * 70)

    print(
        f"Accuracy: "
        f"{prune_acc:.2f}%"
    )

    print(
        f"Drop from FP32: "
        f"{baseline_acc - prune_acc:.2f} pp"
    )

    # --------------------------------------------------------
    # Quantize remaining weights
    # --------------------------------------------------------

    print()
    print(
        f"Applying {WEIGHT_BITS}-bit "
        f"weight quantization..."
    )

    quant_stats = (
        quantize_model_weights(
            model,
            bits=WEIGHT_BITS
        )
    )

    # --------------------------------------------------------
    # Evaluate prune + weight quantization
    # --------------------------------------------------------

    pw_loss, pw_acc = (
        evaluate(
            model,
            test_loader,
            criterion,
            DEVICE
        )
    )

    print()
    print(
        "Pruning + weight quantization"
    )

    print("-" * 70)

    print(
        f"Accuracy: "
        f"{pw_acc:.2f}%"
    )

    print(
        f"Drop from FP32: "
        f"{baseline_acc - pw_acc:.2f} pp"
    )

    print(
        f"Zero weights after quantization: "
        f"{quant_stats['zero_weights']:,}"
    )

    print(
        f"Final sparsity: "
        f"{100 * quant_stats['zero_weights'] / quant_stats['total_weights']:.2f}%"
    )

    # --------------------------------------------------------
    # Activation quantization
    # --------------------------------------------------------

    hooks = (
        add_activation_quantization(
            model
        )
    )

    final_loss, final_acc = (
        evaluate(
            model,
            test_loader,
            criterion,
            DEVICE
        )
    )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print()
    print("=" * 70)

    print(
        "Pruning + Weight + "
        "Activation Quantization"
    )

    print("=" * 70)

    print(
        f"Prune ratio:     "
        f"{PRUNE_RATIO:.0%}"
    )

    print(
        f"Weight bits:     "
        f"{WEIGHT_BITS}"
    )

    print(
        f"Activation bits: "
        f"{ACTIVATION_BITS}"
    )

    print(
        f"Test loss:       "
        f"{final_loss:.4f}"
    )

    print(
        f"Test accuracy:   "
        f"{final_acc:.2f}%"
    )

    print(
        f"Total drop:      "
        f"{baseline_acc - final_acc:.2f} pp"
    )

    print(
        f"Final sparsity:  "
        f"{100 * quant_stats['zero_weights'] / quant_stats['total_weights']:.2f}%"
    )

    print("=" * 70)

    # Remove hooks
    for handle in hooks:
        handle.remove()


if __name__ == "__main__":
    main()
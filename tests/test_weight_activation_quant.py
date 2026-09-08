import torch
import torch.nn as nn

from baseline.model import get_model
from baseline.data import get_dataloaders
from baseline.evaluate import evaluate
from baseline.config import DEVICE, MODEL_PATH

from compression.quantization import symmetric_quantize_per_channel
from compression.activation_quantization import symmetric_quantize_activation


WEIGHT_BITS = 6
ACTIVATION_BITS = 7


# ============================================================
# WEIGHT QUANTIZATION
# ============================================================

def quantize_model_weights(model, bits=8):

    quantized_params = 0
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

            quantized_params += weight.numel()
            total_scales += scales.numel()

    print(
        f"Quantized weight parameters: "
        f"{quantized_params:,}"
    )

    print(
        f"Weight scales stored: "
        f"{total_scales:,}"
    )

    return model


# ============================================================
# ACTIVATION QUANTIZATION HOOK
# ============================================================

def activation_quant_hook(module, input, output):
    """
    This function runs automatically after selected layers.

    It takes the output activation and replaces it with
    a quantized-dequantized version.
    """

    if isinstance(output, torch.Tensor):

        return symmetric_quantize_activation(
            output,
            bits=ACTIVATION_BITS
        )

    return output


# ============================================================
# REGISTER ACTIVATION HOOKS
# ============================================================

def add_activation_quantization(model):

    hooks = []

    for name, module in model.named_modules():

        # For first experiment:
        # quantize outputs of Conv2d and Linear layers

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            handle = module.register_forward_hook(
                activation_quant_hook
            )

            hooks.append(handle)

    print(
        f"Activation quantization hooks added: "
        f"{len(hooks)}"
    )

    return hooks


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Weight bits: {WEIGHT_BITS}"
    )

    print(
        f"Activation bits: {ACTIVATION_BITS}"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load trained model
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

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    _, test_loader = get_dataloaders()

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline_loss, baseline_acc = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE
    )

    print()
    print("Baseline")
    print("-" * 70)

    print(
        f"Test accuracy: "
        f"{baseline_acc:.2f}%"
    )

    print(
        f"Test loss: "
        f"{baseline_loss:.4f}"
    )

    # --------------------------------------------------------
    # Weight quantization
    # --------------------------------------------------------

    print()
    print("Applying weight quantization...")

    model = quantize_model_weights(
        model,
        bits=WEIGHT_BITS
    )

    # --------------------------------------------------------
    # Weight-only accuracy
    # --------------------------------------------------------

    weight_loss, weight_acc = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE
    )

    print()
    print("Weight quantization only")
    print("-" * 70)

    print(
        f"Accuracy: "
        f"{weight_acc:.2f}%"
    )

    print(
        f"Accuracy drop: "
        f"{baseline_acc - weight_acc:.2f} pp"
    )

    # --------------------------------------------------------
    # Add activation quantization
    # --------------------------------------------------------

    print()
    print("Adding activation quantization...")

    hooks = add_activation_quantization(
        model
    )

    # --------------------------------------------------------
    # Weight + activation quantization
    # --------------------------------------------------------

    quant_loss, quant_acc = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE
    )

    print()
    print("=" * 70)

    print("Weight + activation quantization")

    print("=" * 70)

    print(
        f"Weight bits: "
        f"{WEIGHT_BITS}"
    )

    print(
        f"Activation bits: "
        f"{ACTIVATION_BITS}"
    )

    print(
        f"Test loss: "
        f"{quant_loss:.4f}"
    )

    print(
        f"Test accuracy: "
        f"{quant_acc:.2f}%"
    )

    print(
        f"Total accuracy drop: "
        f"{baseline_acc - quant_acc:.2f} pp"
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Remove hooks afterwards
    # --------------------------------------------------------

    for handle in hooks:
        handle.remove()


if __name__ == "__main__":
    main()
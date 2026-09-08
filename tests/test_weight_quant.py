import torch
import torch.nn as nn

from baseline.model import get_model
from baseline.data import get_dataloaders
from baseline.evaluate import evaluate
from baseline.config import DEVICE, MODEL_PATH

from compression.quantization import symmetric_quantize_per_channel


WEIGHT_BITS = 8


def quantize_model_weights(model, bits=8):
    """
    Quantize Conv2d and Linear weights using
    symmetric per-output-channel quantization.

    Biases and BatchNorm parameters are left unchanged.
    """

    quantized_params = 0
    total_scales = 0

    for name, module in model.named_modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            if module.weight is None:
                continue

            weight = module.weight.data

            # Per-channel quantization
            quantized_weight, scales, q = (
                symmetric_quantize_per_channel(
                    weight,
                    bits=bits
                )
            )

            # Replace original FP32 weights with
            # quantized-dequantized approximation
            module.weight.data.copy_(quantized_weight)

            num_params = weight.numel()
            num_scales = scales.numel()

            quantized_params += num_params
            total_scales += num_scales

            print(
                f"{name:40s} "
                f"| shape={tuple(weight.shape)} "
                f"| weights={num_params:,} "
                f"| scales={num_scales} "
                f"| scale_min={scales.min().item():.6e} "
                f"| scale_max={scales.max().item():.6e}"
            )

    print()
    print("=" * 70)
    print(f"Quantized weight parameters: {quantized_params:,}")
    print(f"Number of scales stored:     {total_scales:,}")
    print("=" * 70)

    return model


def main():

    print("=" * 70)
    print(f"Using device: {DEVICE}")
    print(f"Weight quantization: {WEIGHT_BITS} bits")
    print("Quantization mode: per-output-channel symmetric")
    print("=" * 70)

    # --------------------------------------------------
    # Load model architecture
    # --------------------------------------------------

    model = get_model()

    # --------------------------------------------------
    # Load trained FP32 checkpoint
    # --------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    print()
    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Stored baseline accuracy: "
        f"{checkpoint['test_accuracy']:.2f}%"
    )

    # --------------------------------------------------
    # Load CIFAR-10 test set
    # --------------------------------------------------

    _, test_loader = get_dataloaders()

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------
    # Evaluate baseline model
    # --------------------------------------------------

    baseline_loss, baseline_acc = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE
    )

    print()
    print("Before quantization")
    print("-" * 70)

    print(
        f"Test loss:     "
        f"{baseline_loss:.4f}"
    )

    print(
        f"Test accuracy: "
        f"{baseline_acc:.2f}%"
    )

    # --------------------------------------------------
    # Quantize weights
    # --------------------------------------------------

    print()
    print("Quantizing model weights...")
    print("-" * 70)

    model = quantize_model_weights(
        model,
        bits=WEIGHT_BITS
    )

    # --------------------------------------------------
    # Evaluate quantized model
    # --------------------------------------------------

    quant_loss, quant_acc = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE
    )

    accuracy_drop = baseline_acc - quant_acc

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("After quantization")
    print("=" * 70)

    print(
        f"Weight bits:    "
        f"{WEIGHT_BITS}"
    )

    print(
        f"Test loss:      "
        f"{quant_loss:.4f}"
    )

    print(
        f"Test accuracy:  "
        f"{quant_acc:.2f}%"
    )

    print(
        f"Accuracy drop:  "
        f"{accuracy_drop:.2f} percentage points"
    )

    # Ideal compression ignoring scale overhead
    ideal_compression = 32 / WEIGHT_BITS

    print(
        f"Ideal weight compression: "
        f"{ideal_compression:.2f}x"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
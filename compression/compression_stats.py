import torch
import torch.nn as nn

from baseline.model import get_model
from baseline.data import get_dataloaders
from baseline.config import DEVICE, MODEL_PATH

from compression.quantization import symmetric_quantize_per_channel
from compression.activation_quantization import symmetric_quantize_activation


WEIGHT_BITS = 6
ACTIVATION_BITS = 7


# ------------------------------------------------------------
# Weight quantization + storage counting
# ------------------------------------------------------------

def quantize_weights_and_count(model, bits=8):
    """
    Quantizes Conv2d / Linear weights using per-channel
    symmetric quantization and computes storage.

    Returns:
        quantized_weight_bits
        fp32_uncompressed_bits
        scale_bits
        number_quantized_weights
        number_scales
    """

    quantized_weight_bits = 0
    fp32_uncompressed_bits = 0
    scale_bits = 0

    number_quantized_weights = 0
    number_scales = 0

    for name, module in model.named_modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            weight = module.weight.data

            quantized_weight, scales, q = (
                symmetric_quantize_per_channel(
                    weight,
                    bits=bits
                )
            )

            module.weight.data.copy_(quantized_weight)

            n_weights = weight.numel()
            n_scales = scales.numel()

            # compressed quantized values
            quantized_weight_bits += n_weights * bits

            # each scale stored as FP32
            scale_bits += n_scales * 32

            number_quantized_weights += n_weights
            number_scales += n_scales

            # bias remains FP32
            if module.bias is not None:
                fp32_uncompressed_bits += module.bias.numel() * 32


    # Count all parameters that were NOT Conv/Linear weights
    for name, param in model.named_parameters():

        # Skip Conv/Linear weights because already counted
        module_name = name.rsplit(".", 1)[0]
        param_name = name.rsplit(".", 1)[-1]

        module = dict(model.named_modules()).get(module_name, None)

        if (
            module is not None
            and isinstance(module, (nn.Conv2d, nn.Linear))
            and param_name == "weight"
        ):
            continue

        # Biases of Conv/Linear were already counted above
        if (
            module is not None
            and isinstance(module, (nn.Conv2d, nn.Linear))
            and param_name == "bias"
        ):
            continue

        fp32_uncompressed_bits += param.numel() * 32


    return {
        "quantized_weight_bits": quantized_weight_bits,
        "fp32_uncompressed_bits": fp32_uncompressed_bits,
        "scale_bits": scale_bits,
        "number_quantized_weights": number_quantized_weights,
        "number_scales": number_scales
    }


# ------------------------------------------------------------
# Activation storage counter
# ------------------------------------------------------------

class ActivationStats:
    def __init__(self):
        self.original_bits = 0
        self.compressed_bits = 0
        self.num_tensors = 0
        self.num_elements = 0


def make_activation_hook(stats, bits):
    """
    Forward hook that:
      1. counts activation size
      2. quantizes/dequantizes activation
    """

    def hook(module, inputs, output):

        if not isinstance(output, torch.Tensor):
            return output

        n = output.numel()

        # Original FP32 activation
        stats.original_bits += n * 32

        # Quantized activation + one FP32 scale
        stats.compressed_bits += n * bits
        stats.compressed_bits += 32

        stats.num_tensors += 1
        stats.num_elements += n

        return symmetric_quantize_activation(
            output,
            bits=bits
        )

    return hook


def add_activation_hooks(model, stats, bits):
    handles = []

    for name, module in model.named_modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            handle = module.register_forward_hook(
                make_activation_hook(
                    stats,
                    bits
                )
            )

            handles.append(handle)

    return handles


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def bits_to_mb(bits):
    return bits / 8 / (1024 ** 2)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print(f"Weight bits:     {WEIGHT_BITS}")
    print(f"Activation bits: {ACTIVATION_BITS}")
    print("=" * 70)

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


    # --------------------------------------------------------
    # Original model parameter storage
    # --------------------------------------------------------

    total_model_params = sum(
        p.numel()
        for p in model.parameters()
    )

    original_model_bits = (
        total_model_params * 32
    )

    print()
    print("Original FP32 model")
    print("-" * 70)

    print(
        f"Total parameters: "
        f"{total_model_params:,}"
    )

    print(
        f"Original parameter size: "
        f"{bits_to_mb(original_model_bits):.4f} MB"
    )


    # --------------------------------------------------------
    # Weight compression
    # --------------------------------------------------------

    weight_stats = quantize_weights_and_count(
        model,
        bits=WEIGHT_BITS
    )

    compressed_model_bits = (
        weight_stats["quantized_weight_bits"]
        +
        weight_stats["fp32_uncompressed_bits"]
        +
        weight_stats["scale_bits"]
    )

    weight_compression_ratio = (
        original_model_bits
        /
        compressed_model_bits
    )

    print()
    print("Weight compression")
    print("-" * 70)

    print(
        f"Quantized weights: "
        f"{weight_stats['number_quantized_weights']:,}"
    )

    print(
        f"Stored scales: "
        f"{weight_stats['number_scales']:,}"
    )

    print(
        f"Quantized weight storage: "
        f"{bits_to_mb(weight_stats['quantized_weight_bits']):.4f} MB"
    )

    print(
        f"Scale overhead: "
        f"{bits_to_mb(weight_stats['scale_bits']):.4f} MB"
    )

    print(
        f"FP32 exceptions "
        f"(BN, biases, etc.): "
        f"{bits_to_mb(weight_stats['fp32_uncompressed_bits']):.4f} MB"
    )

    print(
        f"Final compressed model parameter size: "
        f"{bits_to_mb(compressed_model_bits):.4f} MB"
    )

    print(
        f"Model/weight compression ratio: "
        f"{weight_compression_ratio:.4f}x"
    )


    # --------------------------------------------------------
    # Activation compression
    # --------------------------------------------------------

    stats = ActivationStats()

    handles = add_activation_hooks(
        model,
        stats,
        ACTIVATION_BITS
    )

    _, test_loader = get_dataloaders()

    # Use one batch for activation measurement
    images, labels = next(iter(test_loader))

    images = images.to(DEVICE)

    with torch.no_grad():
        _ = model(images)

    for h in handles:
        h.remove()

    activation_compression_ratio = (
        stats.original_bits
        /
        stats.compressed_bits
    )

    print()
    print("Activation compression")
    print("-" * 70)

    print(
        f"Measured using one inference batch"
    )

    print(
        f"Batch size: "
        f"{images.shape[0]}"
    )

    print(
        f"Activation tensors counted: "
        f"{stats.num_tensors}"
    )

    print(
        f"Activation elements counted: "
        f"{stats.num_elements:,}"
    )

    print(
        f"Original FP32 activation storage: "
        f"{bits_to_mb(stats.original_bits):.4f} MB"
    )

    print(
        f"Compressed activation storage: "
        f"{bits_to_mb(stats.compressed_bits):.4f} MB"
    )

    print(
        f"Activation compression ratio: "
        f"{activation_compression_ratio:.4f}x"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
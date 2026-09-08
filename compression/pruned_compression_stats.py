import torch
import torch.nn as nn

from baseline.model import get_model
from baseline.config import DEVICE, MODEL_PATH
from compression.pruning import global_magnitude_prune
from compression.quantization import symmetric_quantize_per_channel


PRUNE_RATIO = 0.10
WEIGHT_BITS = 6


def bits_to_mb(bits):
    return bits / 8 / (1024 ** 2)


def main():

    print("=" * 70)
    print(f"Prune ratio: {PRUNE_RATIO:.0%}")
    print(f"Weight bits: {WEIGHT_BITS}")
    print("=" * 70)

    # --------------------------------------------------------
    # Load FP32 baseline
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
    model.eval()

    # --------------------------------------------------------
    # Original model storage
    # --------------------------------------------------------

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    original_bits = total_params * 32

    # --------------------------------------------------------
    # Prune
    # --------------------------------------------------------

    model, prune_stats = global_magnitude_prune(
        model,
        prune_ratio=PRUNE_RATIO
    )

    # --------------------------------------------------------
    # Quantize and count
    # --------------------------------------------------------

    total_quantized_weights = 0
    total_nonzero_weights = 0
    total_scales = 0

    for module in model.modules():

        if isinstance(module, (nn.Conv2d, nn.Linear)):

            weight = module.weight.data

            quantized_weight, scales, q = (
                symmetric_quantize_per_channel(
                    weight,
                    bits=WEIGHT_BITS
                )
            )

            module.weight.data.copy_(
                quantized_weight
            )

            total_quantized_weights += weight.numel()

            total_nonzero_weights += (
                quantized_weight != 0
            ).sum().item()

            total_scales += scales.numel()

    # --------------------------------------------------------
    # Count FP32 parameters not quantized
    # --------------------------------------------------------

    fp32_exception_params = (
        total_params - total_quantized_weights
    )

    fp32_exception_bits = (
        fp32_exception_params * 32
    )

    # --------------------------------------------------------
    # Dense quantized representation
    # --------------------------------------------------------

    dense_quantized_bits = (
        total_quantized_weights * WEIGHT_BITS
    )

    # --------------------------------------------------------
    # Sparse representation
    #
    # Bitmap:
    #    1 bit per original quantized weight
    #
    # Nonzero values:
    #    WEIGHT_BITS bits each
    # --------------------------------------------------------

    bitmap_bits = total_quantized_weights

    sparse_value_bits = (
        total_nonzero_weights * WEIGHT_BITS
    )

    # --------------------------------------------------------
    # Scale metadata
    # --------------------------------------------------------

    scale_bits = total_scales * 32

    # --------------------------------------------------------
    # Total compressed model
    # --------------------------------------------------------

    compressed_bits = (
        bitmap_bits
        + sparse_value_bits
        + scale_bits
        + fp32_exception_bits
    )

    compression_ratio = (
        original_bits / compressed_bits
    )

    sparsity = (
        1
        - total_nonzero_weights
        / total_quantized_weights
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("Original model")
    print("-" * 70)

    print(
        f"Parameters: {total_params:,}"
    )

    print(
        f"FP32 size: "
        f"{bits_to_mb(original_bits):.4f} MB"
    )

    print()
    print("Pruned + quantized weights")
    print("-" * 70)

    print(
        f"Eligible weights: "
        f"{total_quantized_weights:,}"
    )

    print(
        f"Nonzero weights: "
        f"{total_nonzero_weights:,}"
    )

    print(
        f"Final sparsity: "
        f"{100*sparsity:.2f}%"
    )

    print(
        f"Nonzero value storage: "
        f"{bits_to_mb(sparse_value_bits):.4f} MB"
    )

    print(
        f"Bitmap overhead: "
        f"{bits_to_mb(bitmap_bits):.4f} MB"
    )

    print(
        f"Scale overhead: "
        f"{bits_to_mb(scale_bits):.4f} MB"
    )

    print(
        f"FP32 exceptions: "
        f"{bits_to_mb(fp32_exception_bits):.4f} MB"
    )

    print()
    print("=" * 70)

    print(
        f"Final compressed model size: "
        f"{bits_to_mb(compressed_bits):.4f} MB"
    )

    print(
        f"Compression ratio: "
        f"{compression_ratio:.4f}x"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
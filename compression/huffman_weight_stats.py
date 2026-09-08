
import csv
import heapq
import math
from collections import Counter

import torch
import torch.nn as nn

from baseline.model import get_model
from baseline.config import DEVICE, MODEL_PATH

from compression.quantization import symmetric_quantize_per_channel


# ============================================================
# CONFIG
# ============================================================

ASSIGNMENT_CSV = "mixed_precision_assignment.csv"

# Canonical Huffman metadata assumption:
#
# For each layer and each distinct quantized symbol:
#   - store symbol value using the layer's assigned bit-width
#   - store code length using CODE_LENGTH_BITS bits
#
# Canonical codes can be reconstructed from symbol + code length,
# so the actual bit pattern does not need to be stored per symbol.
CODE_LENGTH_BITS = 8

OUTPUT_CSV = "huffman_weight_compression_stats.csv"


# ============================================================
# MODEL / ASSIGNMENT
# ============================================================

def load_assignment():
    assignment = {}

    with open(ASSIGNMENT_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            assignment[row["layer_name"]] = int(row["assigned_bits"])

    return assignment


def load_clean_model():
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

    return model


def get_quantizable_layers(model):
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
        and module.weight is not None
    }


# ============================================================
# HUFFMAN CODE LENGTHS
# ============================================================

def huffman_code_lengths(counter):
    """
    Return dict[symbol] = Huffman code length.

    This calculates the exact optimal binary Huffman lengths
    for the observed symbol frequencies.
    """

    if len(counter) == 0:
        return {}

    if len(counter) == 1:
        only_symbol = next(iter(counter))
        return {
            only_symbol: 1
        }

    # Heap entries:
    # (frequency, unique_id, node)
    #
    # Leaf node: ("leaf", symbol)
    # Internal:  ("node", left, right)

    heap = []
    unique = 0

    for symbol, freq in counter.items():
        heapq.heappush(
            heap,
            (
                freq,
                unique,
                ("leaf", symbol)
            )
        )

        unique += 1

    while len(heap) > 1:

        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)

        merged = (
            "node",
            n1,
            n2
        )

        heapq.heappush(
            heap,
            (
                f1 + f2,
                unique,
                merged
            )
        )

        unique += 1

    root = heap[0][2]

    lengths = {}

    stack = [
        (root, 0)
    ]

    while stack:

        node, depth = stack.pop()

        if node[0] == "leaf":

            symbol = node[1]

            lengths[symbol] = max(
                depth,
                1
            )

        else:

            _, left, right = node

            stack.append(
                (
                    left,
                    depth + 1
                )
            )

            stack.append(
                (
                    right,
                    depth + 1
                )
            )

    return lengths


def shannon_entropy(counter):
    total = sum(counter.values())

    if total == 0:
        return 0.0

    entropy = 0.0

    for freq in counter.values():

        p = freq / total

        entropy -= (
            p
            * math.log2(p)
        )

    return entropy


# ============================================================
# MAIN
# ============================================================

def main():

    assignment = load_assignment()

    model = load_clean_model()

    layers = get_quantizable_layers(
        model
    )

    missing = (
        set(layers.keys())
        - set(assignment.keys())
    )

    if missing:
        raise RuntimeError(
            "Assignment CSV missing layers:\n"
            + "\n".join(sorted(missing))
        )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    total_quantized_weights = sum(
        module.weight.numel()
        for module in layers.values()
    )

    total_original_model_bits = (
        total_params * 32
    )

    total_fp32_exception_bits = (
        (total_params - total_quantized_weights)
        * 32
    )

    total_scale_bits = 0

    total_dense_mixed_bits = 0
    total_huffman_stream_bits = 0
    total_huffman_codebook_bits = 0

    output_rows = []

    for name, module in layers.items():

        bits = assignment[name]

        weight = module.weight.data

        q_weight, scales, q = (
            symmetric_quantize_per_channel(
                weight,
                bits=bits
            )
        )

        # q is stored as a float tensor in the simulation,
        # but its values are integer-valued. Convert to Python ints.
        symbols = (
            q.detach()
            .round()
            .to(torch.int32)
            .cpu()
            .flatten()
            .tolist()
        )

        counter = Counter(
            symbols
        )

        lengths = huffman_code_lengths(
            counter
        )

        stream_bits = sum(
            freq
            * lengths[symbol]
            for symbol, freq
            in counter.items()
        )

        num_symbols = len(counter)

        # Canonical Huffman codebook:
        # symbol representation + code length for every distinct symbol.
        codebook_bits = (
            num_symbols
            * (
                bits
                + CODE_LENGTH_BITS
            )
        )

        dense_bits = (
            weight.numel()
            * bits
        )

        scale_bits = (
            scales.numel()
            * 32
        )

        entropy = shannon_entropy(
            counter
        )

        average_code_length = (
            stream_bits
            / weight.numel()
        )

        zero_fraction = (
            counter.get(0, 0)
            / weight.numel()
        )

        total_dense_mixed_bits += (
            dense_bits
        )

        total_huffman_stream_bits += (
            stream_bits
        )

        total_huffman_codebook_bits += (
            codebook_bits
        )

        total_scale_bits += (
            scale_bits
        )

        row = {
            "layer_name":
                name,

            "assigned_bits":
                bits,

            "num_weights":
                weight.numel(),

            "distinct_symbols":
                num_symbols,

            "zero_fraction":
                zero_fraction,

            "entropy_bits_per_weight":
                entropy,

            "avg_huffman_bits_per_weight":
                average_code_length,

            "dense_weight_kb":
                dense_bits
                / 8
                / 1024,

            "huffman_stream_kb":
                stream_bits
                / 8
                / 1024,

            "codebook_kb":
                codebook_bits
                / 8
                / 1024,
        }

        output_rows.append(
            row
        )

        print(
            f"{name:45s} "
            f"W{bits} "
            f"N={weight.numel():8,d} "
            f"symbols={num_symbols:3d} "
            f"H={entropy:5.3f} "
            f"Huff={average_code_length:5.3f} "
            f"zero={100*zero_fraction:5.1f}%"
        )

    # --------------------------------------------------------
    # Dense mixed-precision model
    # --------------------------------------------------------

    dense_mixed_model_bits = (
        total_dense_mixed_bits
        + total_scale_bits
        + total_fp32_exception_bits
    )

    # --------------------------------------------------------
    # Huffman-coded mixed-precision model
    # --------------------------------------------------------

    huffman_model_bits = (
        total_huffman_stream_bits
        + total_huffman_codebook_bits
        + total_scale_bits
        + total_fp32_exception_bits
    )

    dense_model_mb = (
        dense_mixed_model_bits
        / 8
        / (1024 ** 2)
    )

    huffman_model_mb = (
        huffman_model_bits
        / 8
        / (1024 ** 2)
    )

    original_model_mb = (
        total_original_model_bits
        / 8
        / (1024 ** 2)
    )

    dense_cr = (
        total_original_model_bits
        / dense_mixed_model_bits
    )

    huffman_cr = (
        total_original_model_bits
        / huffman_model_bits
    )

    avg_dense_bits = (
        total_dense_mixed_bits
        / total_quantized_weights
    )

    avg_huffman_bits = (
        total_huffman_stream_bits
        / total_quantized_weights
    )

    print()
    print("=" * 90)
    print("HUFFMAN COMPRESSION SUMMARY")
    print("=" * 90)

    print(
        f"Original FP32 model size:       "
        f"{original_model_mb:.4f} MB"
    )

    print(
        f"Dense mixed-precision size:     "
        f"{dense_model_mb:.4f} MB"
    )

    print(
        f"Dense mixed-precision CR:       "
        f"{dense_cr:.4f}x"
    )

    print(
        f"Average assigned bits/weight:   "
        f"{avg_dense_bits:.4f}"
    )

    print()
    print(
        f"Huffman weight stream:          "
        f"{total_huffman_stream_bits / 8 / (1024**2):.4f} MB"
    )

    print(
        f"Huffman codebook overhead:      "
        f"{total_huffman_codebook_bits / 8 / (1024**2):.6f} MB"
    )

    print(
        f"Scale overhead:                 "
        f"{total_scale_bits / 8 / (1024**2):.4f} MB"
    )

    print(
        f"FP32 exceptions:                "
        f"{total_fp32_exception_bits / 8 / (1024**2):.4f} MB"
    )

    print(
        f"Huffman-coded model size:       "
        f"{huffman_model_mb:.4f} MB"
    )

    print(
        f"Huffman-coded model CR:         "
        f"{huffman_cr:.4f}x"
    )

    print(
        f"Average Huffman bits/weight:    "
        f"{avg_huffman_bits:.4f}"
    )

    print(
        f"Extra compression over dense:   "
        f"{dense_mixed_model_bits / huffman_model_bits:.4f}x"
    )

    print("=" * 90)

    # Add global summary as a final row-like CSV record separately.
    with open(
        OUTPUT_CSV,
        "w",
        newline=""
    ) as f:

        fieldnames = (
            list(output_rows[0].keys())
        )

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    print(
        f"Per-layer statistics written to: "
        f"{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()

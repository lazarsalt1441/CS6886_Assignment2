
import csv
import heapq
import math
from collections import Counter

import torch
import torch.nn as nn

from baseline.config import DEVICE, MODEL_PATH
from baseline.model import get_model
from distillation.student_model import get_student_model
from baseline.data import get_dataloaders
from compression.activation_quantization import symmetric_quantize_activation
from compression.quantization import symmetric_quantize_per_channel


# ============================================================
# CONFIG
# ============================================================

# Best width-0.35 distilled student
STUDENT_CHECKPOINT = "checkpoints/student_w0p35_kd_best.pth"

WEIGHT_BITS = 6
ACTIVATION_BITS = 8

# Canonical Huffman metadata:
# symbol value + code length for each distinct symbol in a layer
CODE_LENGTH_BITS = 8

OUTPUT_CSV = "huffman_distilled_w035_stats.csv"


# ============================================================
# HELPERS
# ============================================================

def bits_to_mb(bits):
    return bits / 8 / (1024 ** 2)


def count_params(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


def load_original_teacher_reference():
    teacher = get_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    teacher.load_state_dict(
        checkpoint["model_state_dict"]
    )

    params = count_params(
        teacher
    )

    bits = params * 32

    return {
        "params": params,
        "bits": bits,
        "mb": bits_to_mb(bits),
        "accuracy": checkpoint["test_accuracy"],
    }


def load_student():
    checkpoint = torch.load(
        STUDENT_CHECKPOINT,
        map_location=DEVICE
    )

    width = checkpoint["width_mult"]

    student = get_student_model(
        width_mult=width
    )

    student.load_state_dict(
        checkpoint["student_state_dict"]
    )

    student = student.to(
        DEVICE
    )

    student.eval()

    return student, checkpoint


def get_quantizable_layers(model):
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
        and module.weight is not None
    }


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    total = 0
    correct = 0

    for images, labels in loader:

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        logits = model(
            images
        )

        pred = logits.argmax(
            dim=1
        )

        total += labels.size(0)

        correct += (
            pred == labels
        ).sum().item()

    return (
        100.0
        * correct
        / total
    )


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


def add_activation_hooks(model):
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
# HUFFMAN
# ============================================================

def huffman_code_lengths(counter):
    """
    Exact binary Huffman code lengths.
    """

    if len(counter) == 0:
        return {}

    if len(counter) == 1:
        symbol = next(iter(counter))

        return {
            symbol: 1
        }

    heap = []
    uid = 0

    for symbol, freq in counter.items():

        heapq.heappush(
            heap,
            (
                freq,
                uid,
                ("leaf", symbol)
            )
        )

        uid += 1

    while len(heap) > 1:

        f1, _, n1 = (
            heapq.heappop(
                heap
            )
        )

        f2, _, n2 = (
            heapq.heappop(
                heap
            )
        )

        heapq.heappush(
            heap,
            (
                f1 + f2,
                uid,
                (
                    "node",
                    n1,
                    n2
                )
            )
        )

        uid += 1

    root = heap[0][2]

    lengths = {}

    stack = [
        (
            root,
            0
        )
    ]

    while stack:

        node, depth = (
            stack.pop()
        )

        if node[0] == "leaf":

            lengths[
                node[1]
            ] = max(
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
    total = sum(
        counter.values()
    )

    if total == 0:
        return 0.0

    H = 0.0

    for freq in (
        counter.values()
    ):

        p = (
            freq / total
        )

        H -= (
            p
            * math.log2(p)
        )

    return H


# ============================================================
# MAIN
# ============================================================

def main():

    teacher_ref = (
        load_original_teacher_reference()
    )

    student, student_checkpoint = (
        load_student()
    )

    _, test_loader = (
        get_dataloaders()
    )

    # --------------------------------------------------------
    # FP32 student accuracy
    # --------------------------------------------------------

    fp32_student_acc = evaluate(
        student,
        test_loader
    )

    student_params = (
        count_params(
            student
        )
    )

    student_fp32_bits = (
        student_params * 32
    )

    # --------------------------------------------------------
    # Quantize weights to W6 and collect symbols
    # --------------------------------------------------------

    layers = (
        get_quantizable_layers(
            student
        )
    )

    total_quantized_weights = 0

    total_dense_weight_bits = 0

    total_huffman_stream_bits = 0

    total_huffman_metadata_bits = 0

    total_scale_bits = 0

    layer_rows = []

    for name, module in (
        layers.items()
    ):

        weight = (
            module.weight.data
        )

        q_weight, scales, q = (
            symmetric_quantize_per_channel(
                weight,
                bits=WEIGHT_BITS
            )
        )

        # Replace model weights for real accuracy evaluation
        module.weight.data.copy_(
            q_weight
        )

        q_int = (
            q.detach()
            .round()
            .to(torch.int32)
            .cpu()
            .flatten()
            .tolist()
        )

        counter = Counter(
            q_int
        )

        lengths = (
            huffman_code_lengths(
                counter
            )
        )

        huffman_stream_bits = sum(
            freq
            * lengths[symbol]
            for symbol, freq
            in counter.items()
        )

        distinct_symbols = (
            len(counter)
        )

        # Canonical Huffman codebook:
        #
        # for each symbol:
        #   WEIGHT_BITS bits to identify symbol
        #   CODE_LENGTH_BITS to store code length
        huffman_metadata_bits = (
            distinct_symbols
            * (
                WEIGHT_BITS
                + CODE_LENGTH_BITS
            )
        )

        dense_weight_bits = (
            weight.numel()
            * WEIGHT_BITS
        )

        scale_bits = (
            scales.numel()
            * 32
        )

        total_quantized_weights += (
            weight.numel()
        )

        total_dense_weight_bits += (
            dense_weight_bits
        )

        total_huffman_stream_bits += (
            huffman_stream_bits
        )

        total_huffman_metadata_bits += (
            huffman_metadata_bits
        )

        total_scale_bits += (
            scale_bits
        )

        entropy = (
            shannon_entropy(
                counter
            )
        )

        avg_huff_bits = (
            huffman_stream_bits
            / weight.numel()
        )

        zero_fraction = (
            counter.get(
                0,
                0
            )
            / weight.numel()
        )

        layer_rows.append(
            {
                "layer_name":
                    name,

                "num_weights":
                    weight.numel(),

                "weight_bits":
                    WEIGHT_BITS,

                "distinct_symbols":
                    distinct_symbols,

                "zero_fraction":
                    zero_fraction,

                "entropy_bits_per_weight":
                    entropy,

                "avg_huffman_bits_per_weight":
                    avg_huff_bits,

                "dense_weight_kb":
                    dense_weight_bits
                    / 8
                    / 1024,

                "huffman_stream_kb":
                    huffman_stream_bits
                    / 8
                    / 1024,

                "metadata_kb":
                    huffman_metadata_bits
                    / 8
                    / 1024,

                "scale_kb":
                    scale_bits
                    / 8
                    / 1024,
            }
        )

        print(
            f"{name:45s} "
            f"N={weight.numel():8,d} "
            f"H={entropy:5.3f} "
            f"Huff={avg_huff_bits:5.3f} "
            f"zero={100*zero_fraction:5.1f}%"
        )

    # --------------------------------------------------------
    # W6A8 accuracy
    # --------------------------------------------------------

    act_hooks = (
        add_activation_hooks(
            student
        )
    )

    compressed_acc = evaluate(
        student,
        test_loader
    )

    for handle in act_hooks:
        handle.remove()

    # --------------------------------------------------------
    # FP32 exceptions
    # --------------------------------------------------------

    fp32_exception_params = (
        student_params
        - total_quantized_weights
    )

    fp32_exception_bits = (
        fp32_exception_params
        * 32
    )

    # --------------------------------------------------------
    # Dense W6 model
    # --------------------------------------------------------

    dense_compressed_bits = (
        total_dense_weight_bits
        + total_scale_bits
        + fp32_exception_bits
    )

    # --------------------------------------------------------
    # Huffman W6 model
    # --------------------------------------------------------

    huffman_compressed_bits = (
        total_huffman_stream_bits
        + total_huffman_metadata_bits
        + total_scale_bits
        + fp32_exception_bits
    )

    # --------------------------------------------------------
    # Compression ratios
    # --------------------------------------------------------

    dense_internal_cr = (
        student_fp32_bits
        / dense_compressed_bits
    )

    dense_overall_cr = (
        teacher_ref["bits"]
        / dense_compressed_bits
    )

    huffman_internal_cr = (
        student_fp32_bits
        / huffman_compressed_bits
    )

    huffman_overall_cr = (
        teacher_ref["bits"]
        / huffman_compressed_bits
    )

    avg_huffman_bits = (
        total_huffman_stream_bits
        / total_quantized_weights
    )

    print()
    print("=" * 95)
    print("DISTILLED W0.35 + W6A8 + HUFFMAN SUMMARY")
    print("=" * 95)

    print(
        f"Original teacher FP32 accuracy:     "
        f"{teacher_ref['accuracy']:.2f}%"
    )

    print(
        f"Original teacher FP32 size:         "
        f"{teacher_ref['mb']:.4f} MB"
    )

    print()

    print(
        f"Student FP32 accuracy:              "
        f"{fp32_student_acc:.2f}%"
    )

    print(
        f"Student FP32 size:                  "
        f"{bits_to_mb(student_fp32_bits):.4f} MB"
    )

    print()

    print(
        f"W6A8 accuracy:                      "
        f"{compressed_acc:.2f}%"
    )

    print(
        f"Accuracy drop vs student:           "
        f"{fp32_student_acc - compressed_acc:.2f} pp"
    )

    print(
        f"Accuracy drop vs original teacher:  "
        f"{teacher_ref['accuracy'] - compressed_acc:.2f} pp"
    )

    print()

    print(
        f"Dense W6 student size:              "
        f"{bits_to_mb(dense_compressed_bits):.4f} MB"
    )

    print(
        f"Dense W6 internal CR:               "
        f"{dense_internal_cr:.4f}x"
    )

    print(
        f"Dense W6 overall CR vs original:    "
        f"{dense_overall_cr:.4f}x"
    )

    print()

    print(
        f"Average Huffman bits/weight:        "
        f"{avg_huffman_bits:.4f}"
    )

    print(
        f"Huffman weight stream:              "
        f"{bits_to_mb(total_huffman_stream_bits):.4f} MB"
    )

    print(
        f"Huffman metadata overhead:          "
        f"{bits_to_mb(total_huffman_metadata_bits):.6f} MB"
    )

    print(
        f"Scale overhead:                     "
        f"{bits_to_mb(total_scale_bits):.4f} MB"
    )

    print(
        f"FP32 exceptions:                    "
        f"{bits_to_mb(fp32_exception_bits):.4f} MB"
    )

    print(
        f"Huffman student size:               "
        f"{bits_to_mb(huffman_compressed_bits):.4f} MB"
    )

    print(
        f"Huffman internal CR:                "
        f"{huffman_internal_cr:.4f}x"
    )

    print(
        f"HUFFMAN OVERALL CR vs original:     "
        f"{huffman_overall_cr:.4f}x"
    )

    print(
        f"Extra gain vs dense W6:             "
        f"{dense_compressed_bits / huffman_compressed_bits:.4f}x"
    )

    print("=" * 95)

    # --------------------------------------------------------
    # Save per-layer stats
    # --------------------------------------------------------

    with open(
        OUTPUT_CSV,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=layer_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(
            layer_rows
        )

    print(
        f"Per-layer Huffman stats -> "
        f"{OUTPUT_CSV}"
    )


if __name__ == "__main__":
    main()

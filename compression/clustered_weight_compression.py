
import csv
import heapq
import math
from collections import Counter

import torch
import torch.nn as nn

from baseline.model import get_model
from baseline.data import get_dataloaders
from baseline.evaluate import evaluate
from baseline.config import DEVICE, MODEL_PATH

from compression.activation_quantization import symmetric_quantize_activation


# ============================================================
# CONFIG
# ============================================================

ASSIGNMENT_CSV = "mixed_precision_assignment.csv"

ACTIVATION_BITS = 8

# Map the previous mixed-precision bit assignment to number of
# learned centroids per layer.
#
# W7 -> 64 centroids  (nominal 6-bit indices)
# W6 -> 32 centroids  (nominal 5-bit indices)
# W5 -> 16 centroids  (nominal 4-bit indices)
CENTROIDS_BY_ASSIGNED_BITS = {
    7: 64,
    6: 32,
    5: 16,
}

# Lloyd k-means iterations.
KMEANS_ITERS = 20

# For very large layers, use at most this many weights to update centroids.
# Assignment of ALL weights to centroids is still exact afterwards.
MAX_KMEANS_TRAIN_SAMPLES = 200000

# Canonical Huffman metadata:
# store one code length per distinct index symbol.
CODE_LENGTH_BITS = 8

OUTPUT_LAYER_CSV = "clustered_weight_stats.csv"


# ============================================================
# LOAD MODEL / ASSIGNMENT
# ============================================================

def load_assignment():
    assignment = {}

    with open(ASSIGNMENT_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            assignment[row["layer_name"]] = int(
                row["assigned_bits"]
            )

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

    return model, checkpoint


def get_quantizable_layers(model):
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
        and module.weight is not None
    }


# ============================================================
# MANUAL 1D K-MEANS
# ============================================================

def init_centroids_quantiles(values, k):
    """
    Robust deterministic initialization using quantiles.
    For scalar weights, this is much more stable than random seeds.
    """
    qs = torch.linspace(
        0.0,
        1.0,
        steps=k,
        device=values.device
    )

    centroids = torch.quantile(
        values,
        qs
    )

    return centroids


def assign_to_centroids(values, centroids, chunk_size=200000):
    """
    Assign scalar values to nearest centroid.
    Chunked to avoid large temporary [N, K] tensors.
    """
    assignments = []

    for start in range(0, values.numel(), chunk_size):

        chunk = values[
            start:start + chunk_size
        ]

        distances = torch.abs(
            chunk[:, None]
            - centroids[None, :]
        )

        idx = torch.argmin(
            distances,
            dim=1
        )

        assignments.append(idx)

    return torch.cat(assignments)


def manual_kmeans_1d(values, k, iters=20):
    """
    Manual Lloyd k-means for 1D weight clustering.

    Returns:
        centroids
        assignments for ALL values
    """

    values = values.flatten().float()

    # --------------------------------------------------------
    # Training subset for centroid updates
    # --------------------------------------------------------

    if values.numel() > MAX_KMEANS_TRAIN_SAMPLES:

        # Deterministic roughly-uniform sample through the tensor.
        idx = torch.linspace(
            0,
            values.numel() - 1,
            steps=MAX_KMEANS_TRAIN_SAMPLES,
            device=values.device
        ).long()

        train_values = values[idx]

    else:
        train_values = values

    # Avoid requesting more clusters than unique values.
    unique_count = torch.unique(
        train_values
    ).numel()

    k = min(
        k,
        int(unique_count)
    )

    if k <= 1:
        centroid = train_values.mean().reshape(1)

        assignments = torch.zeros(
            values.numel(),
            dtype=torch.long,
            device=values.device
        )

        return centroid, assignments

    centroids = init_centroids_quantiles(
        train_values,
        k
    )

    # --------------------------------------------------------
    # Lloyd iterations
    # --------------------------------------------------------

    for _ in range(iters):

        train_assignments = assign_to_centroids(
            train_values,
            centroids
        )

        new_centroids = centroids.clone()

        for cluster_id in range(k):

            mask = (
                train_assignments
                == cluster_id
            )

            if mask.any():

                new_centroids[cluster_id] = (
                    train_values[mask]
                    .mean()
                )

        delta = torch.max(
            torch.abs(
                new_centroids
                - centroids
            )
        ).item()

        centroids = new_centroids

        if delta < 1e-7:
            break

    # --------------------------------------------------------
    # Assign every weight using final centroids
    # --------------------------------------------------------

    assignments = assign_to_centroids(
        values,
        centroids
    )

    return centroids, assignments


def cluster_layer_weight(module, k):
    """
    Replace layer weights by nearest learned centroid.
    """
    original_shape = (
        module.weight.data.shape
    )

    values = (
        module.weight.data
        .detach()
        .flatten()
        .float()
    )

    centroids, assignments = (
        manual_kmeans_1d(
            values,
            k=k,
            iters=KMEANS_ITERS
        )
    )

    reconstructed = (
        centroids[
            assignments
        ]
        .reshape(
            original_shape
        )
    )

    module.weight.data.copy_(
        reconstructed.to(
            module.weight.dtype
        )
    )

    return centroids, assignments


# ============================================================
# ACTIVATION QUANTIZATION (A8)
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
# HUFFMAN LENGTHS FOR CLUSTER INDICES
# ============================================================

def huffman_code_lengths(counter):

    if len(counter) == 0:
        return {}

    if len(counter) == 1:
        only = next(iter(counter))

        return {
            only: 1
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

        f1, _, n1 = heapq.heappop(
            heap
        )

        f2, _, n2 = heapq.heappop(
            heap
        )

        merged = (
            "node",
            n1,
            n2
        )

        heapq.heappush(
            heap,
            (
                f1 + f2,
                uid,
                merged
            )
        )

        uid += 1

    root = heap[0][2]

    lengths = {}

    stack = [
        (root, 0)
    ]

    while stack:

        node, depth = (
            stack.pop()
        )

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


def entropy(counter):
    total = sum(
        counter.values()
    )

    result = 0.0

    for freq in counter.values():

        p = (
            freq / total
        )

        result -= (
            p
            * math.log2(p)
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 90)
    print("Layer-wise learned weight clustering + Huffman coding")
    print("=" * 90)

    assignment = (
        load_assignment()
    )

    model, checkpoint = (
        load_clean_model()
    )

    layers = (
        get_quantizable_layers(
            model
        )
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

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    total_quantized_weights = sum(
        module.weight.numel()
        for module in layers.values()
    )

    original_bits = (
        total_params * 32
    )

    fp32_exception_bits = (
        (
            total_params
            - total_quantized_weights
        )
        * 32
    )

    total_index_fixed_bits = 0
    total_huffman_stream_bits = 0
    total_codebook_bits = 0
    total_huffman_metadata_bits = 0

    rows = []

    # --------------------------------------------------------
    # Cluster each layer
    # --------------------------------------------------------

    for idx, (name, module) in enumerate(
        layers.items(),
        start=1
    ):

        assigned_bits = (
            assignment[name]
        )

        requested_k = (
            CENTROIDS_BY_ASSIGNED_BITS[
                assigned_bits
            ]
        )

        print()
        print(
            f"[{idx}/{len(layers)}] "
            f"{name}"
        )

        print(
            f"Previous precision W{assigned_bits} "
            f"-> {requested_k} learned centroids"
        )

        centroids, indices = (
            cluster_layer_weight(
                module,
                k=requested_k
            )
        )

        actual_k = (
            centroids.numel()
        )

        fixed_index_bits = (
            max(
                1,
                math.ceil(
                    math.log2(
                        actual_k
                    )
                )
            )
        )

        counter = Counter(
            indices
            .detach()
            .cpu()
            .tolist()
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

        fixed_stream_bits = (
            module.weight.numel()
            * fixed_index_bits
        )

        # FP32 centroid codebook.
        codebook_bits = (
            actual_k
            * 32
        )

        # Canonical Huffman metadata:
        # index symbol + its code length.
        #
        # Symbol itself requires fixed_index_bits.
        huffman_metadata_bits = (
            len(counter)
            * (
                fixed_index_bits
                + CODE_LENGTH_BITS
            )
        )

        H = entropy(
            counter
        )

        avg_huff = (
            huffman_stream_bits
            / module.weight.numel()
        )

        total_index_fixed_bits += (
            fixed_stream_bits
        )

        total_huffman_stream_bits += (
            huffman_stream_bits
        )

        total_codebook_bits += (
            codebook_bits
        )

        total_huffman_metadata_bits += (
            huffman_metadata_bits
        )

        row = {
            "layer_name":
                name,

            "previous_assigned_bits":
                assigned_bits,

            "num_weights":
                module.weight.numel(),

            "requested_centroids":
                requested_k,

            "actual_centroids":
                actual_k,

            "fixed_index_bits":
                fixed_index_bits,

            "entropy_bits_per_index":
                H,

            "avg_huffman_bits_per_index":
                avg_huff,

            "fixed_index_kb":
                fixed_stream_bits
                / 8
                / 1024,

            "huffman_stream_kb":
                huffman_stream_bits
                / 8
                / 1024,

            "centroid_codebook_kb":
                codebook_bits
                / 8
                / 1024,

            "huffman_metadata_kb":
                huffman_metadata_bits
                / 8
                / 1024,
        }

        rows.append(
            row
        )

        print(
            f"K={actual_k}, "
            f"fixed={fixed_index_bits}b/index, "
            f"H={H:.3f}, "
            f"Huffman={avg_huff:.3f}b/index"
        )

    # --------------------------------------------------------
    # Evaluate clustered model + A8
    # --------------------------------------------------------

    _, test_loader = (
        get_dataloaders()
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    handles = (
        add_activation_hooks(
            model
        )
    )

    test_loss, test_accuracy = (
        evaluate(
            model,
            test_loader,
            criterion,
            DEVICE
        )
    )

    for h in handles:
        h.remove()

    # --------------------------------------------------------
    # Storage calculations
    # --------------------------------------------------------

    fixed_model_bits = (
        total_index_fixed_bits
        + total_codebook_bits
        + fp32_exception_bits
    )

    huffman_model_bits = (
        total_huffman_stream_bits
        + total_codebook_bits
        + total_huffman_metadata_bits
        + fp32_exception_bits
    )

    original_mb = (
        original_bits
        / 8
        / (1024 ** 2)
    )

    fixed_mb = (
        fixed_model_bits
        / 8
        / (1024 ** 2)
    )

    huffman_mb = (
        huffman_model_bits
        / 8
        / (1024 ** 2)
    )

    fixed_cr = (
        original_bits
        / fixed_model_bits
    )

    huffman_cr = (
        original_bits
        / huffman_model_bits
    )

    avg_fixed_bits = (
        total_index_fixed_bits
        / total_quantized_weights
    )

    avg_huffman_bits = (
        total_huffman_stream_bits
        / total_quantized_weights
    )

    print()
    print("=" * 90)
    print("CLUSTERING RESULT")
    print("=" * 90)

    print(
        f"FP32 baseline accuracy:            "
        f"{checkpoint['test_accuracy']:.2f}%"
    )

    print(
        f"Clustered + A8 accuracy:           "
        f"{test_accuracy:.2f}%"
    )

    print(
        f"Accuracy drop:                     "
        f"{checkpoint['test_accuracy'] - test_accuracy:.2f} pp"
    )

    print()
    print(
        f"Original model size:               "
        f"{original_mb:.4f} MB"
    )

    print(
        f"Avg fixed cluster-index bits:      "
        f"{avg_fixed_bits:.4f}"
    )

    print(
        f"Fixed-index clustered model size:  "
        f"{fixed_mb:.4f} MB"
    )

    print(
        f"Fixed-index compression:           "
        f"{fixed_cr:.4f}x"
    )

    print()
    print(
        f"Avg Huffman bits/index:            "
        f"{avg_huffman_bits:.4f}"
    )

    print(
        f"Huffman index stream:              "
        f"{total_huffman_stream_bits / 8 / (1024**2):.4f} MB"
    )

    print(
        f"Centroid codebook overhead:        "
        f"{total_codebook_bits / 8 / (1024**2):.4f} MB"
    )

    print(
        f"Huffman metadata overhead:         "
        f"{total_huffman_metadata_bits / 8 / (1024**2):.6f} MB"
    )

    print(
        f"FP32 exceptions:                   "
        f"{fp32_exception_bits / 8 / (1024**2):.4f} MB"
    )

    print(
        f"Huffman clustered model size:      "
        f"{huffman_mb:.4f} MB"
    )

    print(
        f"Huffman clustered compression:     "
        f"{huffman_cr:.4f}x"
    )

    print("=" * 90)

    # --------------------------------------------------------
    # Save stats
    # --------------------------------------------------------

    with open(
        OUTPUT_LAYER_CSV,
        "w",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(
            rows
        )

    print(
        f"Per-layer stats written to: "
        f"{OUTPUT_LAYER_CSV}"
    )


if __name__ == "__main__":
    main()

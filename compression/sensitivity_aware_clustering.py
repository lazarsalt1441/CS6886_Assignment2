
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

SENSITIVITY_CSV = "layer_sensitivity.csv"
ACTIVATION_BITS = 8
KMEANS_ITERS = 20
MAX_KMEANS_TRAIN_SAMPLES = 200000
CODE_LENGTH_BITS = 8

# Three increasingly aggressive clustering policies.
# Decisions are based on the measured W5 layerwise sensitivity drop.
PROFILES = {
    "conservative": {
        "very_sensitive": 256,  # drop > 0.50 pp
        "sensitive":      128,  # 0.25 < drop <= 0.50
        "medium":          64,  # 0.10 < drop <= 0.25
        "robust":          32,  # drop <= 0.10
    },
    "balanced": {
        "very_sensitive": 128,
        "sensitive":       64,
        "medium":          32,
        "robust":          16,
    },
    "aggressive": {
        "very_sensitive": 64,
        "sensitive":      32,
        "medium":         16,
        "robust":          8,
    },
}

RESULTS_CSV = "sensitivity_cluster_results.csv"
ASSIGNMENT_PREFIX = "cluster_assignment"


# ============================================================
# MODEL HELPERS
# ============================================================

def load_clean_model():
    model = get_model()
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    return model, checkpoint


def get_layers(model):
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
        and module.weight is not None
    }


# ============================================================
# SENSITIVITY MAP
# ============================================================

def load_w5_sensitivity():
    """
    Uses the W5 trial result from layer_sensitivity.csv.
    Negative drops are treated as zero: they mean 'no measurable damage',
    not a real accuracy improvement.
    """
    sensitivity = {}

    with open(SENSITIVITY_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if int(row["trial_bits"]) != 5:
                continue

            sensitivity[row["layer_name"]] = max(
                float(row["accuracy_drop_pp"]),
                0.0
            )

    return sensitivity


def category_from_drop(drop):
    if drop > 0.50:
        return "very_sensitive"
    elif drop > 0.25:
        return "sensitive"
    elif drop > 0.10:
        return "medium"
    else:
        return "robust"


# ============================================================
# 1D K-MEANS
# ============================================================

def init_centroids_quantiles(values, k):
    qs = torch.linspace(0.0, 1.0, steps=k, device=values.device)
    return torch.quantile(values, qs)


def nearest_centroid(values, centroids, chunk_size=200000):
    out = []

    for start in range(0, values.numel(), chunk_size):
        x = values[start:start + chunk_size]
        d = torch.abs(x[:, None] - centroids[None, :])
        out.append(torch.argmin(d, dim=1))

    return torch.cat(out)


def kmeans_1d(values, requested_k):
    values = values.flatten().float()

    if values.numel() > MAX_KMEANS_TRAIN_SAMPLES:
        ids = torch.linspace(
            0,
            values.numel() - 1,
            MAX_KMEANS_TRAIN_SAMPLES,
            device=values.device
        ).long()
        train = values[ids]
    else:
        train = values

    unique = torch.unique(train).numel()
    k = min(requested_k, int(unique))

    if k <= 1:
        c = train.mean().view(1)
        idx = torch.zeros(
            values.numel(),
            dtype=torch.long,
            device=values.device
        )
        return c, idx

    centroids = init_centroids_quantiles(train, k)

    for _ in range(KMEANS_ITERS):
        a = nearest_centroid(train, centroids)
        new = centroids.clone()

        for j in range(k):
            mask = a == j
            if mask.any():
                new[j] = train[mask].mean()

        if torch.max(torch.abs(new - centroids)).item() < 1e-7:
            centroids = new
            break

        centroids = new

    all_idx = nearest_centroid(values, centroids)
    return centroids, all_idx


def cluster_module(module, k):
    shape = module.weight.data.shape
    values = module.weight.data.detach().flatten().float()

    centroids, indices = kmeans_1d(values, k)

    reconstructed = centroids[indices].reshape(shape)
    module.weight.data.copy_(reconstructed.to(module.weight.dtype))

    return centroids, indices


# ============================================================
# HUFFMAN
# ============================================================

def huffman_lengths(counter):
    if len(counter) == 1:
        return {next(iter(counter)): 1}

    heap = []
    uid = 0

    for symbol, freq in counter.items():
        heapq.heappush(heap, (freq, uid, ("leaf", symbol)))
        uid += 1

    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)

        heapq.heappush(
            heap,
            (f1 + f2, uid, ("node", n1, n2))
        )
        uid += 1

    root = heap[0][2]
    lengths = {}
    stack = [(root, 0)]

    while stack:
        node, depth = stack.pop()

        if node[0] == "leaf":
            lengths[node[1]] = max(depth, 1)
        else:
            _, left, right = node
            stack.append((left, depth + 1))
            stack.append((right, depth + 1))

    return lengths


# ============================================================
# ACTIVATIONS
# ============================================================

def make_act_hook(bits):
    def hook(module, inputs, output):
        if not isinstance(output, torch.Tensor):
            return output

        qmax = (2 ** (bits - 1)) - 1
        max_abs = output.abs().max()

        if max_abs.item() == 0:
            return output

        scale = max_abs / qmax
        q = torch.round(output / scale)
        q = torch.clamp(q, -qmax, qmax)
        return q * scale

    return hook


def add_activation_hooks(model):
    handles = []

    for module in get_layers(model).values():
        handles.append(
            module.register_forward_hook(
                make_act_hook(ACTIVATION_BITS)
            )
        )

    return handles


# ============================================================
# ONE PROFILE
# ============================================================

def run_profile(profile_name, policy, sensitivity, test_loader, criterion):
    model, checkpoint = load_clean_model()
    layers = get_layers(model)

    total_params = sum(p.numel() for p in model.parameters())
    total_quant_weights = sum(m.weight.numel() for m in layers.values())

    original_bits = total_params * 32
    fp32_exception_bits = (total_params - total_quant_weights) * 32

    total_huffman_stream_bits = 0
    total_codebook_bits = 0
    total_huffman_metadata_bits = 0

    assignment_rows = []

    print()
    print("=" * 90)
    print(f"PROFILE: {profile_name}")
    print("=" * 90)

    for idx, (name, module) in enumerate(layers.items(), start=1):
        drop = sensitivity[name]
        category = category_from_drop(drop)
        requested_k = policy[category]

        centroids, indices = cluster_module(
            module,
            requested_k
        )

        actual_k = centroids.numel()
        fixed_symbol_bits = max(1, math.ceil(math.log2(actual_k)))

        counter = Counter(
            indices.detach().cpu().tolist()
        )

        lengths = huffman_lengths(counter)

        stream_bits = sum(
            freq * lengths[symbol]
            for symbol, freq in counter.items()
        )

        codebook_bits = actual_k * 32

        metadata_bits = (
            len(counter)
            * (fixed_symbol_bits + CODE_LENGTH_BITS)
        )

        total_huffman_stream_bits += stream_bits
        total_codebook_bits += codebook_bits
        total_huffman_metadata_bits += metadata_bits

        assignment_rows.append({
            "layer_name": name,
            "w5_profile_drop_pp": drop,
            "category": category,
            "requested_centroids": requested_k,
            "actual_centroids": actual_k,
            "num_weights": module.weight.numel(),
            "avg_huffman_bits_per_index":
                stream_bits / module.weight.numel(),
        })

        print(
            f"[{idx:02d}/{len(layers)}] "
            f"{name:45s} "
            f"drop={drop:5.2f} "
            f"{category:14s} "
            f"K={actual_k:3d} "
            f"Huff={stream_bits/module.weight.numel():5.3f}b"
        )

    handles = add_activation_hooks(model)

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE
    )

    for h in handles:
        h.remove()

    huffman_model_bits = (
        total_huffman_stream_bits
        + total_codebook_bits
        + total_huffman_metadata_bits
        + fp32_exception_bits
    )

    model_mb = huffman_model_bits / 8 / (1024 ** 2)
    cr = original_bits / huffman_model_bits

    avg_huff = (
        total_huffman_stream_bits
        / total_quant_weights
    )

    assignment_path = (
        f"{ASSIGNMENT_PREFIX}_{profile_name}.csv"
    )

    with open(assignment_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=assignment_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(assignment_rows)

    result = {
        "profile": profile_name,
        "test_accuracy": test_acc,
        "accuracy_drop_pp":
            checkpoint["test_accuracy"] - test_acc,
        "model_size_mb": model_mb,
        "model_compression_ratio": cr,
        "avg_huffman_bits_per_weight": avg_huff,
        "huffman_stream_mb":
            total_huffman_stream_bits / 8 / (1024 ** 2),
        "codebook_mb":
            total_codebook_bits / 8 / (1024 ** 2),
        "huffman_metadata_mb":
            total_huffman_metadata_bits / 8 / (1024 ** 2),
        "fp32_exception_mb":
            fp32_exception_bits / 8 / (1024 ** 2),
    }

    print()
    print(
        f"{profile_name}: "
        f"accuracy={test_acc:.2f}% | "
        f"CR={cr:.4f}x | "
        f"size={model_mb:.4f} MB | "
        f"avg Huffman={avg_huff:.3f} b/w"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    sensitivity = load_w5_sensitivity()

    _, test_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()

    results = []

    for name, policy in PROFILES.items():
        results.append(
            run_profile(
                name,
                policy,
                sensitivity,
                test_loader,
                criterion
            )
        )

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=results[0].keys()
        )
        writer.writeheader()
        writer.writerows(results)

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)

    for r in results:
        print(
            f"{r['profile']:14s} "
            f"acc={r['test_accuracy']:6.2f}% "
            f"drop={r['accuracy_drop_pp']:5.2f} pp "
            f"CR={r['model_compression_ratio']:6.3f}x "
            f"size={r['model_size_mb']:.4f} MB "
            f"Huff={r['avg_huffman_bits_per_weight']:.3f} b/w"
        )

    print()
    print(f"Results -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()

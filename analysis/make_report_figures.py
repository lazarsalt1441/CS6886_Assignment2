
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


OUTDIR = Path("report_figures")
OUTDIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# File paths
# ------------------------------------------------------------

MIXED_SEARCH = "mixed_precision_search_log.csv"
MIXED_ASSIGNMENT = "mixed_precision_assignment.csv"
MIXED_PRUNING = "mixed_precision_pruning_results.csv"

KD05_SWEEP = "distilled_student_compression_sweep.csv"
KD035_SWEEP = "distilled_student_compression_sweep_0.35.csv"

HUFFMAN_MIXED = "huffman_weight_compression_stats.csv"
HUFFMAN_KD035 = "huffman_distilled_w035_stats.csv"

# Optional files, if you later add them:
LAYER_SENSITIVITY = "layer_sensitivity.csv"
TRAINING_HISTORY = "training_history.csv"


# ============================================================
# Utility
# ============================================================

def savefig(name):
    path = OUTDIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


# ============================================================
# 1. Accuracy vs compression Pareto plot
# ============================================================

def plot_pareto():
    points = [
        {
            "label": "FP32 teacher",
            "accuracy": 95.11,
            "cr": 1.0,
        },
        {
            "label": "Mixed precision + Huffman",
            "accuracy": 94.04,
            "cr": 5.6333,
        },
        {
            "label": "KD 0.5 FP32",
            "accuracy": 91.45,
            "cr": 8.5323 / 2.6721572875976562,
        },
        {
            "label": "KD 0.35 FP32",
            "accuracy": 90.62,
            "cr": 8.5323 / 1.5599746704101562,
        },
        {
            "label": "KD 0.35 + W6A8",
            "accuracy": 90.36,
            "cr": 23.5010,
        },
        {
            "label": "KD 0.35 + W6A8 + Huffman",
            "accuracy": 90.36,
            "cr": 25.2384,
        },
    ]

    # Add all student sweep points if available.
    for path, prefix in [
        (KD05_SWEEP, "KD0.5"),
        (KD035_SWEEP, "KD0.35"),
    ]:
        if os.path.exists(path):
            df = pd.read_csv(path)

            for _, r in df.iterrows():
                points.append({
                    "label":
                        f"{prefix} W{int(r['weight_bits'])}"
                        f"A{int(r['activation_bits'])}"
                        f"P{int(round(100*r['prune_ratio']))}",
                    "accuracy":
                        float(r["compressed_accuracy"]),
                    "cr":
                        float(r["overall_cr_vs_original"]),
                })

    df = pd.DataFrame(points)

    plt.figure(figsize=(9, 6))

    plt.scatter(
        df["cr"],
        df["accuracy"],
        s=45
    )

    # Annotate only the important named operating points.
    highlight = {
        "FP32 teacher",
        "Mixed precision + Huffman",
        "KD 0.35 + W6A8",
        "KD 0.35 + W6A8 + Huffman",
    }

    for _, r in df.iterrows():
        if r["label"] in highlight:
            plt.annotate(
                r["label"],
                (r["cr"], r["accuracy"]),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=9
            )

    plt.xlabel("Overall model compression ratio (×)")
    plt.ylabel("CIFAR-10 test accuracy (%)")
    plt.title("Accuracy–compression trade-off")
    plt.grid(alpha=0.25)

    savefig("fig_pareto_accuracy_vs_compression.png")


# ============================================================
# 2. Distilled student compression sweep
# ============================================================

def plot_student_sweeps():
    if not (
        os.path.exists(KD05_SWEEP)
        and os.path.exists(KD035_SWEEP)
    ):
        print("Skipping student sweep plot: CSV missing.")
        return

    df05 = pd.read_csv(KD05_SWEEP)
    df035 = pd.read_csv(KD035_SWEEP)

    plt.figure(figsize=(9, 6))

    for df, label in [
        (df05, "width 0.50"),
        (df035, "width 0.35"),
    ]:
        plt.scatter(
            df["overall_cr_vs_original"],
            df["compressed_accuracy"],
            label=label,
            s=55
        )

    plt.xlabel("Overall model compression ratio (×)")
    plt.ylabel("CIFAR-10 test accuracy (%)")
    plt.title("Compression sweep for distilled MobileNetV2 students")
    plt.legend()
    plt.grid(alpha=0.25)

    savefig("fig_distilled_student_sweep.png")


# ============================================================
# 3. Mixed-precision bit allocation
# ============================================================

def plot_mixed_bit_allocation():
    if not os.path.exists(MIXED_ASSIGNMENT):
        print("Skipping mixed allocation plot: CSV missing.")
        return

    df = pd.read_csv(MIXED_ASSIGNMENT)

    grouped = (
        df.groupby("assigned_bits")["num_weights"]
        .sum()
        .sort_index()
    )

    total = grouped.sum()

    plt.figure(figsize=(7, 5))

    bars = plt.bar(
        [f"W{int(x)}" for x in grouped.index],
        grouped.values
    )

    for bar, value in zip(bars, grouped.values):
        pct = 100.0 * value / total

        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f"{value:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.ylabel("Number of weights")
    plt.xlabel("Assigned weight precision")
    plt.title("Sensitivity-aware mixed-precision allocation")

    savefig("fig_mixed_precision_weight_allocation.png")


# ============================================================
# 4. Pruning effect after mixed precision
# ============================================================

def plot_mixed_pruning():
    if not os.path.exists(MIXED_PRUNING):
        print("Skipping mixed pruning plot: CSV missing.")
        return

    df = pd.read_csv(MIXED_PRUNING)

    plt.figure(figsize=(8, 5))

    plt.plot(
        100 * df["prune_ratio"],
        df["model_compression_ratio"],
        marker="o"
    )

    plt.xlabel("Global magnitude pruning ratio (%)")
    plt.ylabel("Model compression ratio (×)")
    plt.title("Effect of pruning on mixed-precision storage")
    plt.grid(alpha=0.25)

    savefig("fig_pruning_vs_compression.png")

    plt.figure(figsize=(8, 5))

    plt.plot(
        100 * df["prune_ratio"],
        df["test_accuracy"],
        marker="o"
    )

    plt.xlabel("Global magnitude pruning ratio (%)")
    plt.ylabel("CIFAR-10 test accuracy (%)")
    plt.title("Effect of pruning on mixed-precision accuracy")
    plt.grid(alpha=0.25)

    savefig("fig_pruning_vs_accuracy.png")


# ============================================================
# 5. Huffman benefit per layer
# ============================================================

def plot_huffman_savings(path, filename, title):
    if not os.path.exists(path):
        print(f"Skipping {title}: CSV missing.")
        return

    df = pd.read_csv(path)

    if "dense_weight_kb" not in df.columns:
        return

    df["saving_kb"] = (
        df["dense_weight_kb"]
        - df["huffman_stream_kb"]
    )

    # Largest layers are the most report-friendly.
    df = (
        df.sort_values("num_weights", ascending=False)
        .head(15)
        .copy()
    )

    labels = [
        name.replace("features.", "f.")
        for name in df["layer_name"]
    ]

    x = np.arange(len(df))

    plt.figure(figsize=(11, 6))

    plt.bar(
        x - 0.18,
        df["dense_weight_kb"],
        width=0.36,
        label="Fixed-width"
    )

    plt.bar(
        x + 0.18,
        df["huffman_stream_kb"],
        width=0.36,
        label="Huffman stream"
    )

    plt.xticks(
        x,
        labels,
        rotation=65,
        ha="right"
    )

    plt.ylabel("Weight storage (KB)")
    plt.title(title)
    plt.legend()

    savefig(filename)


# ============================================================
# 6. Full layer sensitivity plot, if layer_sensitivity.csv exists
# ============================================================

def plot_layer_sensitivity():
    if not os.path.exists(LAYER_SENSITIVITY):
        print(
            "Skipping layer sensitivity figure. "
            "Place layer_sensitivity.csv beside this script to enable it."
        )
        return

    df = pd.read_csv(LAYER_SENSITIVITY)

    # Focus on W5 since that was where uniform quantization failed.
    w5 = df[
        df["trial_bits"] == 5
    ].copy()

    w5 = w5.reset_index(drop=True)
    w5["layer_index"] = np.arange(len(w5))

    # Negative measured drops are sampling noise; show them at zero.
    w5["plot_drop"] = w5[
        "accuracy_drop_pp"
    ].clip(lower=0)

    sizes = (
        20
        + 180
        * np.sqrt(
            w5["num_weights"]
            / w5["num_weights"].max()
        )
    )

    plt.figure(figsize=(11, 6))

    plt.scatter(
        w5["layer_index"],
        w5["plot_drop"],
        s=sizes,
        alpha=0.7
    )

    plt.xlabel("Conv/Linear layer index")
    plt.ylabel("Accuracy drop when only this layer is W5 (pp)")
    plt.title(
        "Layer-wise W5 quantization sensitivity\n"
        "Marker area reflects layer parameter count"
    )
    plt.grid(alpha=0.2)

    savefig("fig_layerwise_w5_sensitivity.png")


# ============================================================
# 7. Original baseline training curves, if history exists
# ============================================================

def plot_training_history():
    if not os.path.exists(TRAINING_HISTORY):
        print(
            "Skipping training curves. "
            "Place training_history.csv beside this script."
        )
        return

    df = pd.read_csv(TRAINING_HISTORY)

    required = {
        "epoch",
        "train_loss",
        "test_loss",
        "train_accuracy",
        "test_accuracy",
    }

    missing = required - set(df.columns)

    if missing:
        print(
            f"training_history.csv is missing columns: {missing}"
        )
        return

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["train_loss"],
        label="Train"
    )

    plt.plot(
        df["epoch"],
        df["test_loss"],
        label="Test"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Cross-entropy loss")
    plt.title("Baseline MobileNetV2 training loss")
    plt.legend()
    plt.grid(alpha=0.25)

    savefig("fig_baseline_loss.png")

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["epoch"],
        df["train_accuracy"],
        label="Train"
    )

    plt.plot(
        df["epoch"],
        df["test_accuracy"],
        label="Test"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Baseline MobileNetV2 training accuracy")
    plt.legend()
    plt.grid(alpha=0.25)

    savefig("fig_baseline_accuracy.png")


# ============================================================
# Main
# ============================================================

def main():
    plot_pareto()
    plot_student_sweeps()
    plot_mixed_bit_allocation()
    plot_mixed_pruning()

    plot_huffman_savings(
        HUFFMAN_MIXED,
        "fig_huffman_mixed_layers.png",
        "Huffman coding benefit in the mixed-precision model"
    )

    plot_huffman_savings(
        HUFFMAN_KD035,
        "fig_huffman_kd035_layers.png",
        "Huffman coding benefit in the distilled width-0.35 model"
    )

    plot_layer_sensitivity()
    plot_training_history()

    print()
    print(
        "Done. Figures are in "
        f"{OUTDIR.resolve()}"
    )


if __name__ == "__main__":
    main()

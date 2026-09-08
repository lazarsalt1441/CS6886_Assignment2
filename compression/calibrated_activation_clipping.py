
import csv
import math
import os
from collections import defaultdict

import torch
import torch.nn as nn

from baseline.model import get_model
from baseline.data import get_dataloaders
from baseline.evaluate import evaluate
from baseline.config import DEVICE, MODEL_PATH

from compression.quantization import symmetric_quantize_per_channel


# ============================================================
# CONFIG
# ============================================================

ASSIGNMENT_CSV = "mixed_precision_assignment.csv"

# Percentiles to try for clipping calibration.
# 100.0 corresponds to the old max-based quantizer.
PERCENTILES = [99.0, 99.5, 99.9, 99.95, 99.99, 100.0]

# Evaluate these activation precisions after calibration.
ACTIVATION_BITS_LIST = [7, 6]

# How many batches to use for calibration.
# 8 batches x 128 = ~1024 images with your current batch size.
CALIBRATION_BATCHES = 8

# To avoid storing millions of activations per layer, randomly subsample
# at most this many absolute activation values per layer over calibration.
MAX_SAMPLES_PER_LAYER = 200000

CALIBRATION_CSV = "activation_clipping_thresholds.csv"
RESULTS_CSV = "clipped_activation_results.csv"


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

    return model, checkpoint


def get_quantizable_layers(model):
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
        and module.weight is not None
    }


def apply_mixed_weight_quantization(model, assignment):
    layers = get_quantizable_layers(model)

    missing = set(layers.keys()) - set(assignment.keys())
    if missing:
        raise RuntimeError(
            "Assignment CSV is missing layers:\n" + "\n".join(sorted(missing))
        )

    for name, module in layers.items():
        bits = assignment[name]

        q_weight, scales, q = symmetric_quantize_per_channel(
            module.weight.data,
            bits=bits
        )

        module.weight.data.copy_(q_weight)


# ============================================================
# CALIBRATION DATA COLLECTION
# ============================================================

class ActivationCollector:
    def __init__(self, max_samples_per_layer):
        self.max_samples = max_samples_per_layer
        self.samples = defaultdict(list)
        self.counts = defaultdict(int)

    def add(self, layer_name, tensor):
        """
        Store a random-ish subsample of |activation| values on CPU.
        """
        x = tensor.detach().abs().flatten()

        if x.numel() == 0:
            return

        remaining = self.max_samples - self.counts[layer_name]

        if remaining <= 0:
            return

        # Uniformly subsample from this tensor if needed.
        take = min(remaining, x.numel())

        if take < x.numel():
            idx = torch.randperm(
                x.numel(),
                device=x.device
            )[:take]
            x = x[idx]

        x = x.float().cpu()

        self.samples[layer_name].append(x)
        self.counts[layer_name] += x.numel()

    def finalize(self):
        out = {}

        for name, chunks in self.samples.items():
            if len(chunks) == 0:
                continue

            vals = torch.cat(chunks)

            if vals.numel() > self.max_samples:
                vals = vals[:self.max_samples]

            out[name] = vals

        return out


def make_collect_hook(layer_name, collector):
    def hook(module, inputs, output):
        if isinstance(output, torch.Tensor):
            collector.add(layer_name, output)
        return output
    return hook


def collect_activation_samples(model, calibration_loader):
    collector = ActivationCollector(
        MAX_SAMPLES_PER_LAYER
    )

    handles = []

    for name, module in get_quantizable_layers(model).items():
        handles.append(
            module.register_forward_hook(
                make_collect_hook(name, collector)
            )
        )

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(calibration_loader):

            if batch_idx >= CALIBRATION_BATCHES:
                break

            images = images.to(DEVICE)

            _ = model(images)

            print(
                f"Calibration batch "
                f"{batch_idx + 1}/{CALIBRATION_BATCHES}"
            )

    for h in handles:
        h.remove()

    return collector.finalize()


# ============================================================
# THRESHOLD CALIBRATION
# ============================================================

def quant_dequant_with_alpha(x, bits, alpha):
    """
    Symmetric clipped quantization with fixed alpha.
    """
    if alpha <= 0:
        return x

    qmax = (2 ** (bits - 1)) - 1
    qmin = -qmax

    scale = alpha / qmax

    x_clipped = torch.clamp(
        x,
        -alpha,
        alpha
    )

    q = torch.round(
        x_clipped / scale
    )

    q = torch.clamp(
        q,
        qmin,
        qmax
    )

    return q * scale


def mse_for_candidate(samples, bits, alpha):
    """
    Evaluate clipping+quantization reconstruction MSE on sampled
    activations. We compare against the original unclipped values.
    """
    samples = samples.float()

    reconstructed = quant_dequant_with_alpha(
        samples,
        bits,
        alpha
    )

    mse = torch.mean(
        (samples - reconstructed) ** 2
    ).item()

    return mse


def percentile_value(samples, percentile):
    if percentile >= 100.0:
        return samples.max().item()

    q = percentile / 100.0

    return torch.quantile(
        samples,
        q
    ).item()


def calibrate_thresholds(samples_by_layer, bits):
    """
    Choose, independently per layer, the percentile clipping threshold
    that minimizes reconstruction MSE on calibration samples.
    """
    calibration_rows = []
    chosen = {}

    for layer_name, samples in samples_by_layer.items():

        best = None

        for percentile in PERCENTILES:

            alpha = percentile_value(
                samples,
                percentile
            )

            mse = mse_for_candidate(
                samples,
                bits,
                alpha
            )

            row = {
                "layer_name": layer_name,
                "activation_bits": bits,
                "percentile": percentile,
                "alpha": alpha,
                "mse": mse,
                "num_samples": samples.numel(),
            }

            calibration_rows.append(row)

            if (
                best is None
                or mse < best["mse"]
            ):
                best = row

        chosen[layer_name] = {
            "alpha": best["alpha"],
            "percentile": best["percentile"],
            "mse": best["mse"],
        }

        print(
            f"{layer_name:45s} "
            f"A{bits} "
            f"best={best['percentile']:7.3f}% "
            f"alpha={best['alpha']:.6g} "
            f"mse={best['mse']:.6e}"
        )

    return chosen, calibration_rows


def write_calibration_rows(rows):
    exists = os.path.exists(CALIBRATION_CSV)

    with open(CALIBRATION_CSV, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=rows[0].keys()
        )

        if not exists:
            writer.writeheader()

        writer.writerows(rows)


# ============================================================
# INFERENCE WITH FIXED CLIPPED ACTIVATIONS
# ============================================================

def make_clipped_quant_hook(layer_name, thresholds, bits):
    alpha = thresholds[layer_name]["alpha"]

    def hook(module, inputs, output):

        if isinstance(output, torch.Tensor):
            return quant_dequant_with_alpha(
                output,
                bits,
                alpha
            )

        return output

    return hook


def add_clipped_activation_hooks(model, thresholds, bits):
    handles = []

    for name, module in get_quantizable_layers(model).items():

        if name not in thresholds:
            raise RuntimeError(
                f"No activation threshold found for {name}"
            )

        handles.append(
            module.register_forward_hook(
                make_clipped_quant_hook(
                    name,
                    thresholds,
                    bits
                )
            )
        )

    return handles


# ============================================================
# ACTIVATION STORAGE
# ============================================================

class ActivationStorageCounter:
    def __init__(self, bits):
        self.bits = bits
        self.original_bits = 0
        self.compressed_bits = 0
        self.elements = 0
        self.tensors = 0

    def add(self, output):
        n = output.numel()

        self.original_bits += n * 32

        # Quantized values + one FP32 clipping/scale threshold per tensor.
        self.compressed_bits += n * self.bits
        self.compressed_bits += 32

        self.elements += n
        self.tensors += 1


def make_storage_hook(counter):
    def hook(module, inputs, output):
        if isinstance(output, torch.Tensor):
            counter.add(output)
        return output
    return hook


def measure_activation_storage(model, test_loader, bits):
    counter = ActivationStorageCounter(bits)

    handles = []

    for module in get_quantizable_layers(model).values():
        handles.append(
            module.register_forward_hook(
                make_storage_hook(counter)
            )
        )

    images, _ = next(iter(test_loader))
    images = images.to(DEVICE)

    with torch.no_grad():
        _ = model(images)

    for h in handles:
        h.remove()

    ratio = (
        counter.original_bits
        / counter.compressed_bits
    )

    return {
        "activation_original_mb":
            counter.original_bits
            / 8
            / (1024 ** 2),

        "activation_compressed_mb":
            counter.compressed_bits
            / 8
            / (1024 ** 2),

        "activation_compression_ratio":
            ratio,

        "activation_elements":
            counter.elements,

        "activation_tensors":
            counter.tensors,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # Clear old calibration outputs for a clean run.
    if os.path.exists(CALIBRATION_CSV):
        os.remove(CALIBRATION_CSV)

    if os.path.exists(RESULTS_CSV):
        os.remove(RESULTS_CSV)

    assignment = load_assignment()

    train_loader, test_loader = get_dataloaders()

    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Collect activation samples ONCE using mixed-weight model,
    # before activation quantization.
    # --------------------------------------------------------

    model, checkpoint = load_clean_model()

    apply_mixed_weight_quantization(
        model,
        assignment
    )

    print("=" * 80)
    print("Collecting activation samples for clipping calibration")
    print("=" * 80)

    samples_by_layer = collect_activation_samples(
        model,
        train_loader
    )

    del model

    print()
    print(
        f"Collected calibration samples for "
        f"{len(samples_by_layer)} layers."
    )

    # --------------------------------------------------------
    # Calibrate + evaluate each activation bit width.
    # --------------------------------------------------------

    result_rows = []

    for bits in ACTIVATION_BITS_LIST:

        print()
        print("=" * 80)
        print(
            f"Calibrating per-layer clipping for A{bits}"
        )
        print("=" * 80)

        thresholds, calibration_rows = calibrate_thresholds(
            samples_by_layer,
            bits
        )

        write_calibration_rows(
            calibration_rows
        )

        # Full test evaluation from a fresh clean model.
        eval_model, checkpoint = load_clean_model()

        apply_mixed_weight_quantization(
            eval_model,
            assignment
        )

        # Measure storage BEFORE installing clipped hooks,
        # because storage depends only on bit width / tensor sizes.
        storage = measure_activation_storage(
            eval_model,
            test_loader,
            bits
        )

        handles = add_clipped_activation_hooks(
            eval_model,
            thresholds,
            bits
        )

        test_loss, test_acc = evaluate(
            eval_model,
            test_loader,
            criterion,
            DEVICE
        )

        for h in handles:
            h.remove()

        percentiles_used = [
            thresholds[name]["percentile"]
            for name in thresholds
        ]

        avg_percentile = (
            sum(percentiles_used)
            / len(percentiles_used)
        )

        row = {
            "activation_bits": bits,
            "test_accuracy": test_acc,
            "test_loss": test_loss,
            "accuracy_drop_from_fp32_pp":
                checkpoint["test_accuracy"] - test_acc,
            "activation_compression_ratio":
                storage["activation_compression_ratio"],
            "activation_original_mb":
                storage["activation_original_mb"],
            "activation_compressed_mb":
                storage["activation_compressed_mb"],
            "average_chosen_percentile":
                avg_percentile,
        }

        result_rows.append(row)

        print()
        print("=" * 80)
        print(f"CLIPPED A{bits} RESULT")
        print("=" * 80)

        print(
            f"Accuracy:              "
            f"{test_acc:.2f}%"
        )

        print(
            f"Drop from FP32:        "
            f"{checkpoint['test_accuracy'] - test_acc:.2f} pp"
        )

        print(
            f"Activation CR:         "
            f"{storage['activation_compression_ratio']:.4f}x"
        )

        print(
            f"Average percentile:    "
            f"{avg_percentile:.4f}%"
        )

        print("=" * 80)

        del eval_model

    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=result_rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(result_rows)

    print()
    print(
        f"Calibration details -> "
        f"{CALIBRATION_CSV}"
    )

    print(
        f"Final results -> "
        f"{RESULTS_CSV}"
    )


if __name__ == "__main__":
    main()

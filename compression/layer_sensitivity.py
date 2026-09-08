
import csv
import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from baseline.model import get_model
from baseline.data import get_dataloaders
from baseline.evaluate import evaluate
from baseline.config import DEVICE, MODEL_PATH, BATCH_SIZE, NUM_WORKERS

from compression.quantization import symmetric_quantize_per_channel
from compression.activation_quantization import symmetric_quantize_activation


# ============================================================
# CONFIG
# ============================================================

BASE_WEIGHT_BITS = 7
TRIAL_BITS = [6, 5]
ACTIVATION_BITS = 8

# Use a fixed subset for profiling so the layerwise scan is not too slow.
# The final mixed-precision candidate should later be evaluated on the full test set.
PROFILE_SAMPLES = 2000
PROFILE_SEED = 42

OUTPUT_CSV = "layer_sensitivity.csv"


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# MODEL HELPERS
# ============================================================

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
    """
    Returns a list of (name, module) for Conv2d / Linear layers.
    """
    layers = []

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            if module.weight is not None:
                layers.append((name, module))

    return layers


def quantize_layer(module, bits):
    """
    Quantize exactly one Conv2d / Linear layer in-place.
    """
    weight = module.weight.data

    quantized_weight, scales, q = symmetric_quantize_per_channel(
        weight,
        bits=bits
    )

    module.weight.data.copy_(quantized_weight)

    return {
        "num_weights": weight.numel(),
        "num_scales": scales.numel(),
    }


def quantize_all_except_target(
    model,
    base_bits,
    target_layer_name,
    target_bits
):
    """
    Quantize every Conv2d / Linear layer.

    All layers use base_bits except the target layer, which uses target_bits.
    """
    target_found = False
    target_num_weights = None
    target_num_scales = None

    for name, module in model.named_modules():

        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        if module.weight is None:
            continue

        bits = target_bits if name == target_layer_name else base_bits

        stats = quantize_layer(
            module,
            bits=bits
        )

        if name == target_layer_name:
            target_found = True
            target_num_weights = stats["num_weights"]
            target_num_scales = stats["num_scales"]

    if not target_found:
        raise RuntimeError(
            f"Target layer not found: {target_layer_name}"
        )

    return {
        "target_num_weights": target_num_weights,
        "target_num_scales": target_num_scales,
    }


def quantize_all_weights(model, bits):
    for _, module in get_quantizable_layers(model):
        quantize_layer(module, bits=bits)


# ============================================================
# ACTIVATION QUANTIZATION
# ============================================================

def make_activation_hook(bits):

    def hook(module, inputs, output):

        if isinstance(output, torch.Tensor):
            return symmetric_quantize_activation(
                output,
                bits=bits
            )

        return output

    return hook


def add_activation_hooks(model, bits):
    handles = []

    for _, module in get_quantizable_layers(model):
        handles.append(
            module.register_forward_hook(
                make_activation_hook(bits)
            )
        )

    return handles


# ============================================================
# PROFILE DATALOADER
# ============================================================

def get_profile_loader():
    """
    Build a fixed random subset of the CIFAR-10 test set.
    """
    _, full_test_loader = get_dataloaders()
    dataset = full_test_loader.dataset

    generator = torch.Generator()
    generator.manual_seed(PROFILE_SEED)

    indices = torch.randperm(
        len(dataset),
        generator=generator
    )[:PROFILE_SAMPLES].tolist()

    subset = Subset(
        dataset,
        indices
    )

    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=False
    )

    return loader


# ============================================================
# BASELINE
# ============================================================

def evaluate_uniform_base(profile_loader, criterion):
    """
    Baseline for sensitivity analysis:
        all weights = BASE_WEIGHT_BITS
        activations = ACTIVATION_BITS
    """
    model = load_clean_model()

    quantize_all_weights(
        model,
        bits=BASE_WEIGHT_BITS
    )

    hooks = add_activation_hooks(
        model,
        bits=ACTIVATION_BITS
    )

    loss, acc = evaluate(
        model,
        profile_loader,
        criterion,
        DEVICE
    )

    for h in hooks:
        h.remove()

    return loss, acc


# ============================================================
# RESUME SUPPORT
# ============================================================

def load_completed():
    completed = set()

    if not os.path.exists(OUTPUT_CSV):
        return completed

    with open(OUTPUT_CSV, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            completed.add(
                (
                    row["layer_name"],
                    int(row["trial_bits"])
                )
            )

    return completed


def append_row(row):
    exists = os.path.exists(OUTPUT_CSV)

    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=row.keys()
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(PROFILE_SEED)

    print("=" * 80)
    print("Layer-wise quantization sensitivity profiling")
    print("=" * 80)
    print(f"Device:              {DEVICE}")
    print(f"Base weight bits:    {BASE_WEIGHT_BITS}")
    print(f"Trial bits:          {TRIAL_BITS}")
    print(f"Activation bits:     {ACTIVATION_BITS}")
    print(f"Profiling samples:   {PROFILE_SAMPLES}")
    print(f"Output CSV:          {OUTPUT_CSV}")
    print("=" * 80)

    profile_loader = get_profile_loader()
    criterion = nn.CrossEntropyLoss()

    # --------------------------------------------------------
    # Uniform W7A8 baseline on the SAME profiling subset
    # --------------------------------------------------------

    base_loss, base_acc = evaluate_uniform_base(
        profile_loader,
        criterion
    )

    print()
    print(
        f"Uniform W{BASE_WEIGHT_BITS}A{ACTIVATION_BITS} "
        f"profile accuracy: {base_acc:.2f}%"
    )

    # Get layer names and shapes from a clean model.
    temp_model = load_clean_model()
    layer_info = []

    for name, module in get_quantizable_layers(temp_model):
        layer_info.append(
            {
                "name": name,
                "shape": tuple(module.weight.shape),
                "num_weights": module.weight.numel(),
            }
        )

    del temp_model

    completed = load_completed()

    total = len(layer_info) * len(TRIAL_BITS)
    current = 0

    for info in layer_info:

        layer_name = info["name"]
        shape = info["shape"]
        num_weights = info["num_weights"]

        for trial_bits in TRIAL_BITS:

            current += 1

            key = (
                layer_name,
                trial_bits
            )

            if key in completed:
                print(
                    f"[{current}/{total}] "
                    f"Skipping completed "
                    f"{layer_name} -> W{trial_bits}"
                )
                continue

            print()
            print("-" * 80)
            print(
                f"[{current}/{total}] "
                f"{layer_name}"
            )
            print(
                f"shape={shape}, "
                f"weights={num_weights:,}, "
                f"W{BASE_WEIGHT_BITS} -> W{trial_bits}"
            )

            model = load_clean_model()

            target_stats = quantize_all_except_target(
                model,
                base_bits=BASE_WEIGHT_BITS,
                target_layer_name=layer_name,
                target_bits=trial_bits
            )

            hooks = add_activation_hooks(
                model,
                bits=ACTIVATION_BITS
            )

            loss, acc = evaluate(
                model,
                profile_loader,
                criterion,
                DEVICE
            )

            for h in hooks:
                h.remove()

            accuracy_drop = base_acc - acc

            # Raw bit saving from lowering THIS layer only.
            # Scale count is unchanged by bit-width, so scale metadata
            # does not change for this comparison.
            saved_bits = (
                num_weights
                * (BASE_WEIGHT_BITS - trial_bits)
            )

            saved_kb = (
                saved_bits
                / 8
                / 1024
            )

            # A simple "benefit per damage" score.
            # Higher is better.
            safe_drop = max(
                accuracy_drop,
                0.01
            )

            efficiency = (
                saved_kb / safe_drop
            )

            row = {
                "layer_name": layer_name,
                "weight_shape": str(shape),
                "num_weights": num_weights,
                "base_weight_bits": BASE_WEIGHT_BITS,
                "trial_bits": trial_bits,
                "activation_bits": ACTIVATION_BITS,
                "profile_samples": PROFILE_SAMPLES,
                "base_accuracy": base_acc,
                "trial_accuracy": acc,
                "accuracy_drop_pp": accuracy_drop,
                "saved_kb_vs_base": saved_kb,
                "saving_efficiency_kb_per_pp": efficiency,
                "num_scales": target_stats["target_num_scales"],
            }

            append_row(row)

            print(
                f"Accuracy:      {acc:.2f}%"
            )
            print(
                f"Drop:          {accuracy_drop:.3f} pp"
            )
            print(
                f"Bit saving:    {saved_kb:.2f} KB"
            )
            print(
                f"Efficiency:    {efficiency:.2f} KB/pp"
            )

            del model

    print()
    print("=" * 80)
    print("Sensitivity profiling complete.")
    print(f"Saved to: {OUTPUT_CSV}")
    print("=" * 80)


if __name__ == "__main__":
    main()

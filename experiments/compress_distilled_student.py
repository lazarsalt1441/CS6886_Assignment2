import csv
import os

import torch
import torch.nn as nn

from baseline.config import DEVICE, MODEL_PATH
from baseline.data import get_dataloaders
from baseline.model import get_model
from distillation.student_model import get_student_model

from compression.pruning import global_magnitude_prune
from compression.quantization import symmetric_quantize_per_channel
from compression.activation_quantization import symmetric_quantize_activation


STUDENT_CHECKPOINT = "checkpoints/student_w0p35_kd_best.pth"

WEIGHT_BITS_LIST = [8, 7, 6]
ACTIVATION_BITS_LIST = [8]
PRUNE_RATIOS = [0.0, 0.10, 0.20]

OUTPUT_CSV = "distilled_student_compression_sweep_0.35.csv"


def bits_to_mb(bits):
    return bits / 8 / (1024 ** 2)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def load_teacher_reference_size():
    teacher = get_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    teacher.load_state_dict(
        checkpoint["model_state_dict"]
    )

    total_params = count_params(teacher)
    original_bits = total_params * 32

    return {
        "params": total_params,
        "bits": original_bits,
        "mb": bits_to_mb(original_bits),
        "accuracy": checkpoint["test_accuracy"],
    }


def load_clean_student():
    checkpoint = torch.load(
        STUDENT_CHECKPOINT,
        map_location=DEVICE
    )

    width = checkpoint["width_mult"]

    model = get_student_model(
        width_mult=width
    )

    model.load_state_dict(
        checkpoint["student_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model, checkpoint


@torch.no_grad()
def evaluate(model, dataloader, criterion):
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    for images, labels in dataloader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    return total_loss / total, 100.0 * correct / total


def quantize_student_weights(model, bits):
    layer_stats = {}

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        if module.weight is None:
            continue

        weight = module.weight.data

        q_weight, scales, q = symmetric_quantize_per_channel(
            weight,
            bits=bits
        )

        module.weight.data.copy_(q_weight)

        layer_stats[name] = {
            "bits": bits,
            "total_weights": weight.numel(),
            "nonzero_weights": (q_weight != 0).sum().item(),
            "num_scales": scales.numel(),
        }

    return layer_stats


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

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(
                module.register_forward_hook(
                    make_activation_hook(bits)
                )
            )

    return handles


class ActivationCounter:
    def __init__(self, bits):
        self.bits = bits
        self.original_bits = 0
        self.compressed_bits = 0
        self.elements = 0
        self.tensors = 0

    def add(self, output):
        n = output.numel()

        self.original_bits += n * 32
        self.compressed_bits += n * self.bits + 32
        self.elements += n
        self.tensors += 1


def make_count_hook(counter):
    def hook(module, inputs, output):
        if isinstance(output, torch.Tensor):
            counter.add(output)
        return output

    return hook


def measure_activation_compression(model, loader, bits):
    counter = ActivationCounter(bits)
    handles = []

    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            handles.append(
                module.register_forward_hook(
                    make_count_hook(counter)
                )
            )

    images, _ = next(iter(loader))
    images = images.to(DEVICE)

    with torch.no_grad():
        _ = model(images)

    for handle in handles:
        handle.remove()

    return {
        "original_mb": bits_to_mb(counter.original_bits),
        "compressed_mb": bits_to_mb(counter.compressed_bits),
        "compression_ratio": counter.original_bits / counter.compressed_bits,
        "elements": counter.elements,
        "tensors": counter.tensors,
    }


def calculate_student_storage(
    model,
    layer_stats,
    prune_ratio,
    teacher_reference_bits
):
    total_params = count_params(model)
    student_fp32_bits = total_params * 32

    total_quantized_weights = sum(
        s["total_weights"]
        for s in layer_stats.values()
    )

    total_nonzero_weights = sum(
        s["nonzero_weights"]
        for s in layer_stats.values()
    )

    if prune_ratio > 0:
        value_bits = sum(
            s["nonzero_weights"] * s["bits"]
            for s in layer_stats.values()
        )
        mask_bits = total_quantized_weights
    else:
        value_bits = sum(
            s["total_weights"] * s["bits"]
            for s in layer_stats.values()
        )
        mask_bits = 0

    total_scales = sum(
        s["num_scales"]
        for s in layer_stats.values()
    )

    scale_bits = total_scales * 32

    fp32_exception_params = total_params - total_quantized_weights
    fp32_exception_bits = fp32_exception_params * 32

    compressed_bits = (
        value_bits
        + mask_bits
        + scale_bits
        + fp32_exception_bits
    )

    final_zero_fraction = (
        1
        - total_nonzero_weights / total_quantized_weights
    )

    return {
        "student_params": total_params,
        "student_fp32_mb": bits_to_mb(student_fp32_bits),
        "compressed_student_mb": bits_to_mb(compressed_bits),
        "student_internal_cr": student_fp32_bits / compressed_bits,
        "overall_cr_vs_original": teacher_reference_bits / compressed_bits,
        "weight_values_mb": bits_to_mb(value_bits),
        "mask_mb": bits_to_mb(mask_bits),
        "scale_mb": bits_to_mb(scale_bits),
        "fp32_exception_mb": bits_to_mb(fp32_exception_bits),
        "final_zero_fraction": final_zero_fraction,
    }


def append_csv(row):
    exists = os.path.exists(OUTPUT_CSV)

    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=row.keys()
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def run_experiment(
    weight_bits,
    activation_bits,
    prune_ratio,
    test_loader,
    criterion,
    teacher_ref
):
    model, checkpoint = load_clean_student()

    base_loss, base_acc = evaluate(
        model,
        test_loader,
        criterion
    )

    if prune_ratio > 0:
        model, prune_stats = global_magnitude_prune(
            model,
            prune_ratio=prune_ratio
        )
        actual_pruning_sparsity = prune_stats["actual_sparsity"]
    else:
        actual_pruning_sparsity = 0.0

    layer_stats = quantize_student_weights(
        model,
        weight_bits
    )

    storage = calculate_student_storage(
        model,
        layer_stats,
        prune_ratio,
        teacher_ref["bits"]
    )

    activation_stats = measure_activation_compression(
        model,
        test_loader,
        activation_bits
    )

    handles = add_activation_hooks(
        model,
        activation_bits
    )

    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion
    )

    for handle in handles:
        handle.remove()

    row = {
        "weight_bits": weight_bits,
        "activation_bits": activation_bits,
        "prune_ratio": prune_ratio,
        "student_width": checkpoint["width_mult"],
        "student_fp32_accuracy": base_acc,
        "compressed_accuracy": test_acc,
        "accuracy_drop_vs_student_pp": base_acc - test_acc,
        "accuracy_drop_vs_teacher_pp": teacher_ref["accuracy"] - test_acc,
        "student_params": storage["student_params"],
        "student_fp32_mb": storage["student_fp32_mb"],
        "compressed_student_mb": storage["compressed_student_mb"],
        "student_internal_cr": storage["student_internal_cr"],
        "overall_cr_vs_original": storage["overall_cr_vs_original"],
        "activation_cr": activation_stats["compression_ratio"],
        "activation_original_mb": activation_stats["original_mb"],
        "activation_compressed_mb": activation_stats["compressed_mb"],
        "requested_prune_ratio": prune_ratio,
        "actual_pruning_sparsity": actual_pruning_sparsity,
        "final_zero_fraction": storage["final_zero_fraction"],
        "weight_values_mb": storage["weight_values_mb"],
        "mask_mb": storage["mask_mb"],
        "scale_mb": storage["scale_mb"],
        "fp32_exception_mb": storage["fp32_exception_mb"],
    }

    print()
    print("=" * 90)
    print(
        f"W{weight_bits}"
        f"A{activation_bits} "
        f"P{int(prune_ratio*100)}"
    )
    print("=" * 90)

    print(f"Student FP32 acc:         {base_acc:.2f}%")
    print(f"Compressed acc:           {test_acc:.2f}%")
    print(f"Drop vs student:          {base_acc-test_acc:.2f} pp")
    print(f"Compressed student size:  {storage['compressed_student_mb']:.4f} MB")
    print(f"Student internal CR:      {storage['student_internal_cr']:.4f}x")
    print(f"OVERALL CR vs original:   {storage['overall_cr_vs_original']:.4f}x")
    print(f"Activation CR:            {activation_stats['compression_ratio']:.4f}x")
    print("=" * 90)

    return row


def main():
    if os.path.exists(OUTPUT_CSV):
        os.remove(OUTPUT_CSV)

    teacher_ref = load_teacher_reference_size()

    print("=" * 90)
    print("DISTILLED STUDENT COMPRESSION SWEEP")
    print("=" * 90)
    print(f"Original teacher FP32 size: {teacher_ref['mb']:.4f} MB")
    print(f"Original teacher accuracy:  {teacher_ref['accuracy']:.2f}%")
    print(f"Student checkpoint:         {STUDENT_CHECKPOINT}")
    print("=" * 90)

    _, test_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()

    results = []

    for weight_bits in WEIGHT_BITS_LIST:
        for activation_bits in ACTIVATION_BITS_LIST:
            for prune_ratio in PRUNE_RATIOS:
                row = run_experiment(
                    weight_bits,
                    activation_bits,
                    prune_ratio,
                    test_loader,
                    criterion,
                    teacher_ref
                )

                append_csv(row)
                results.append(row)

    print()
    print("=" * 120)
    print("SUMMARY")
    print("=" * 120)

    print(
        f"{'Config':>12s} "
        f"{'Acc':>9s} "
        f"{'DropStu':>9s} "
        f"{'SizeMB':>9s} "
        f"{'StuCR':>9s} "
        f"{'OverallCR':>10s} "
        f"{'ActCR':>8s}"
    )

    print("-" * 120)

    for r in results:
        config = (
            f"W{r['weight_bits']}"
            f"A{r['activation_bits']}"
            f"P{int(r['prune_ratio']*100)}"
        )

        print(
            f"{config:>12s} "
            f"{r['compressed_accuracy']:8.2f}% "
            f"{r['accuracy_drop_vs_student_pp']:8.2f} "
            f"{r['compressed_student_mb']:9.4f} "
            f"{r['student_internal_cr']:9.3f} "
            f"{r['overall_cr_vs_original']:10.3f} "
            f"{r['activation_cr']:8.3f}"
        )

    print()
    print(f"Results written to: {OUTPUT_CSV}")
    print("=" * 120)


if __name__ == "__main__":
    main()

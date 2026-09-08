
import csv
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from baseline.config import DEVICE, MODEL_PATH
from baseline.data import get_dataloaders
from baseline.model import get_model
from distillation.student_model import get_student_model
from distillation.qat_layers import convert_to_qat


# ============================================================
# CONFIG
# ============================================================

STUDENT_WIDTH = 0.25

# Stage 1: distill a width-0.25 FP32 student
KD_EPOCHS = 100
KD_LR = 0.05
KD_MOMENTUM = 0.9
KD_WEIGHT_DECAY = 5e-4
TEMPERATURE = 4.0
ALPHA = 0.30

# Stage 2: quantization-aware fine-tuning
QAT_CONFIGS = [
    {"weight_bits": 3, "activation_bits": 8},
    {"weight_bits": 2, "activation_bits": 8},
]

QAT_EPOCHS = 30
QAT_LR = 3e-4
QAT_WEIGHT_DECAY = 1e-5
RESUME_QAT = True

OUTPUT_DIR = Path("qat_w025_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

KD_BEST = OUTPUT_DIR / "student_w025_kd_best.pth"
KD_LATEST = OUTPUT_DIR / "student_w025_kd_latest.pth"
RESULTS_CSV = OUTPUT_DIR / "qat_w025_results.csv"

KD_HISTORY_CSV = OUTPUT_DIR / "kd_training_history.csv"

RESUME_KD = True
SEED = 42


# ============================================================
# UTILITIES
# ============================================================

def set_seed(seed):
    torch.manual_seed(seed)


def evaluate(model, loader):
    model.eval()

    total = 0
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = F.cross_entropy(
                logits,
                labels
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            pred = logits.argmax(dim=1)

            total += labels.size(0)
            correct += (
                pred == labels
            ).sum().item()

    return (
        total_loss / total,
        100.0 * correct / total
    )


def kd_loss(
    student_logits,
    teacher_logits,
    labels
):
    hard = F.cross_entropy(
        student_logits,
        labels
    )

    T = TEMPERATURE

    student_log_probs = F.log_softmax(
        student_logits / T,
        dim=1
    )

    teacher_probs = F.softmax(
        teacher_logits / T,
        dim=1
    )

    soft = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean"
    ) * (T * T)

    total = (
        ALPHA * hard
        + (1 - ALPHA) * soft
    )

    return total, hard, soft


def cosine_lr(
    base_lr,
    epoch,
    total_epochs,
    warmup=5
):
    e = epoch + 1

    if e <= warmup:
        return base_lr * e / warmup

    progress = (
        e - warmup
    ) / max(
        1,
        total_epochs - warmup
    )

    return (
        0.5
        * base_lr
        * (
            1
            + math.cos(
                math.pi * progress
            )
        )
    )


def set_lr(
    optimizer,
    lr
):
    for group in optimizer.param_groups:
        group["lr"] = lr


def append_csv(row):
    exists = RESULTS_CSV.exists()

    with open(
        RESULTS_CSV,
        "a",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=row.keys()
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


def append_history(path, row):
    """
    Append one epoch of training history.
    Existing files are preserved so resumed runs continue logging.
    """
    path = Path(path)
    exists = path.exists()

    with open(
        path,
        "a",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=row.keys()
        )

        if not exists:
            writer.writeheader()

        writer.writerow(row)


# ============================================================
# TEACHER
# ============================================================

def load_teacher():
    teacher = get_model()

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    teacher.load_state_dict(
        checkpoint["model_state_dict"]
    )

    teacher = teacher.to(DEVICE)
    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad = False

    return teacher, checkpoint


# ============================================================
# STAGE 1: WIDTH-0.25 DISTILLATION
# ============================================================

def train_kd_student(
    teacher,
    train_loader,
    test_loader
):
    student = get_student_model(
        width_mult=STUDENT_WIDTH
    ).to(DEVICE)

    optimizer = optim.SGD(
        student.parameters(),
        lr=KD_LR,
        momentum=KD_MOMENTUM,
        weight_decay=KD_WEIGHT_DECAY
    )

    start_epoch = 0
    best_acc = 0.0

    if (
        RESUME_KD
        and KD_LATEST.exists()
    ):
        ckpt = torch.load(
            KD_LATEST,
            map_location=DEVICE
        )

        student.load_state_dict(
            ckpt["student_state_dict"]
        )

        optimizer.load_state_dict(
            ckpt["optimizer_state_dict"]
        )

        start_epoch = ckpt["epoch"]
        best_acc = ckpt["best_test_accuracy"]

        print(
            f"Resuming KD from epoch "
            f"{start_epoch}"
        )

    for epoch in range(
        start_epoch,
        KD_EPOCHS
    ):
        student.train()
        teacher.eval()

        lr = cosine_lr(
            KD_LR,
            epoch,
            KD_EPOCHS
        )

        set_lr(
            optimizer,
            lr
        )

        total = 0
        correct = 0
        running = 0.0
        running_hard = 0.0
        running_soft = 0.0

        epoch_start = time.time()

        for batch_idx, (
            images,
            labels
        ) in enumerate(
            train_loader
        ):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            student_logits = student(
                images
            )

            with torch.no_grad():
                teacher_logits = teacher(
                    images
                )

            loss, hard, soft = kd_loss(
                student_logits,
                teacher_logits,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                5.0
            )

            optimizer.step()

            running += (
                loss.item()
                * labels.size(0)
            )

            running_hard += (
                hard.item()
                * labels.size(0)
            )

            running_soft += (
                soft.item()
                * labels.size(0)
            )

            pred = student_logits.argmax(
                dim=1
            )

            total += labels.size(0)
            correct += (
                pred == labels
            ).sum().item()

            if batch_idx % 100 == 0:
                print(
                    f"[KD] epoch {epoch+1}/{KD_EPOCHS} "
                    f"batch {batch_idx}/{len(train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        test_loss, test_acc = evaluate(
            student,
            test_loader
        )

        train_acc = (
            100.0
            * correct
            / total
        )

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        train_loss = (
            running / total
        )

        hard_loss_epoch = (
            running_hard / total
        )

        soft_loss_epoch = (
            running_soft / total
        )

        print(
            f"[KD] epoch {epoch+1} "
            f"train={train_acc:.2f}% "
            f"test={test_acc:.2f}% "
            f"lr={lr:.6f} "
            f"time={epoch_seconds:.1f}s"
        )

        append_history(
            KD_HISTORY_CSV,
            {
                "epoch":
                    epoch + 1,
                "train_loss":
                    train_loss,
                "train_hard_loss":
                    hard_loss_epoch,
                "train_soft_loss":
                    soft_loss_epoch,
                "train_accuracy":
                    train_acc,
                "test_loss":
                    test_loss,
                "test_accuracy":
                    test_acc,
                "learning_rate":
                    lr,
                "runtime_seconds":
                    epoch_seconds,
            }
        )

        latest = {
            "width_mult":
                STUDENT_WIDTH,
            "epoch":
                epoch + 1,
            "student_state_dict":
                student.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "best_test_accuracy":
                best_acc,
        }

        torch.save(
            latest,
            KD_LATEST
        )

        if test_acc > best_acc:
            best_acc = test_acc

            torch.save(
                {
                    "width_mult":
                        STUDENT_WIDTH,
                    "epoch":
                        epoch + 1,
                    "student_state_dict":
                        student.state_dict(),
                    "best_test_accuracy":
                        best_acc,
                },
                KD_BEST
            )

            print(
                f"Saved KD best: "
                f"{best_acc:.2f}%"
            )

    best = torch.load(
        KD_BEST,
        map_location=DEVICE
    )

    student.load_state_dict(
        best["student_state_dict"]
    )

    _, final_acc = evaluate(
        student,
        test_loader
    )

    print(
        f"Best width-0.25 KD accuracy: "
        f"{final_acc:.2f}%"
    )

    return student, final_acc


# ============================================================
# STAGE 2: QAT
# ============================================================

def run_qat(
    fp32_student_state,
    weight_bits,
    activation_bits,
    train_loader,
    test_loader,
    teacher
):
    model = get_student_model(
        width_mult=STUDENT_WIDTH
    )

    model.load_state_dict(
        fp32_student_state
    )

    model = convert_to_qat(
        model,
        weight_bits=weight_bits,
        activation_bits=activation_bits
    ).to(DEVICE)

    optimizer = optim.Adam(
        model.parameters(),
        lr=QAT_LR,
        weight_decay=QAT_WEIGHT_DECAY
    )

    best_acc = 0.0
    start_epoch = 0

    out_path = (
        OUTPUT_DIR
        / f"student_w025_W{weight_bits}A{activation_bits}_qat_best.pth"
    )

    latest_path = (
        OUTPUT_DIR
        / f"student_w025_W{weight_bits}A{activation_bits}_qat_latest.pth"
    )

    history_path = (
        OUTPUT_DIR
        / f"W{weight_bits}A{activation_bits}_qat_training_history.csv"
    )

    if (
        RESUME_QAT
        and latest_path.exists()
    ):
        ckpt = torch.load(
            latest_path,
            map_location=DEVICE
        )

        model.load_state_dict(
            ckpt["model_state_dict"]
        )

        optimizer.load_state_dict(
            ckpt["optimizer_state_dict"]
        )

        start_epoch = ckpt["epoch"]
        best_acc = ckpt["best_test_accuracy"]

        print(
            f"Resuming QAT W{weight_bits}A{activation_bits} "
            f"from epoch {start_epoch}"
        )

    print()
    print("=" * 90)
    print(
        f"QAT W{weight_bits}A{activation_bits}"
    )
    print("=" * 90)

    for epoch in range(
        start_epoch,
        QAT_EPOCHS
    ):
        model.train()
        teacher.eval()

        lr = (
            0.5
            * QAT_LR
            * (
                1
                + math.cos(
                    math.pi
                    * epoch
                    / max(
                        1,
                        QAT_EPOCHS - 1
                    )
                )
            )
        )

        set_lr(
            optimizer,
            lr
        )

        total = 0
        correct = 0
        running_loss = 0.0
        running_hard = 0.0
        running_soft = 0.0
        epoch_start = time.time()

        for batch_idx, (
            images,
            labels
        ) in enumerate(
            train_loader
        ):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            student_logits = model(
                images
            )

            with torch.no_grad():
                teacher_logits = teacher(
                    images
                )

            loss, hard, soft = kd_loss(
                student_logits,
                teacher_logits,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                5.0
            )

            optimizer.step()

            batch_size = labels.size(0)

            running_loss += (
                loss.item()
                * batch_size
            )

            running_hard += (
                hard.item()
                * batch_size
            )

            running_soft += (
                soft.item()
                * batch_size
            )

            total += batch_size

            pred = student_logits.argmax(
                dim=1
            )

            correct += (
                pred == labels
            ).sum().item()

            if batch_idx % 100 == 0:
                print(
                    f"[QAT W{weight_bits}] "
                    f"epoch {epoch+1}/{QAT_EPOCHS} "
                    f"batch {batch_idx}/{len(train_loader)} "
                    f"loss={loss.item():.4f}"
                )

        test_loss, test_acc = evaluate(
            model,
            test_loader
        )

        train_acc = (
            100.0
            * correct
            / total
        )

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        train_loss = (
            running_loss / total
        )

        hard_loss_epoch = (
            running_hard / total
        )

        soft_loss_epoch = (
            running_soft / total
        )

        print(
            f"[QAT W{weight_bits}] "
            f"epoch {epoch+1}: "
            f"train={train_acc:.2f}% "
            f"test={test_acc:.2f}% "
            f"lr={lr:.7f}"
        )

        append_history(
            history_path,
            {
                "epoch":
                    epoch + 1,
                "train_loss":
                    train_loss,
                "train_hard_loss":
                    hard_loss_epoch,
                "train_soft_loss":
                    soft_loss_epoch,
                "train_accuracy":
                    train_acc,
                "test_loss":
                    test_loss,
                "test_accuracy":
                    test_acc,
                "learning_rate":
                    lr,
                "runtime_seconds":
                    epoch_seconds,
            }
        )

        # Save latest checkpoint EVERY epoch so the run can be interrupted safely.
        torch.save(
            {
                "width_mult":
                    STUDENT_WIDTH,
                "weight_bits":
                    weight_bits,
                "activation_bits":
                    activation_bits,
                "epoch":
                    epoch + 1,
                "model_state_dict":
                    model.state_dict(),
                "optimizer_state_dict":
                    optimizer.state_dict(),
                "best_test_accuracy":
                    best_acc,
            },
            latest_path
        )

        if test_acc > best_acc:
            best_acc = test_acc

            torch.save(
                {
                    "width_mult":
                        STUDENT_WIDTH,
                    "weight_bits":
                        weight_bits,
                    "activation_bits":
                        activation_bits,
                    "epoch":
                        epoch + 1,
                    "model_state_dict":
                        model.state_dict(),
                    "optimizer_state_dict":
                        optimizer.state_dict(),
                    "best_test_accuracy":
                        best_acc,
                    "test_accuracy":
                        best_acc,
                },
                out_path
            )

            print(
                f"Saved QAT best "
                f"{best_acc:.2f}%"
            )

    # --------------------------------------------
    # Storage estimate
    # --------------------------------------------

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    quant_weight_params = sum(
        module.weight.numel()
        for module in model.modules()
        if isinstance(
            module,
            (nn.Conv2d, nn.Linear)
        )
    )

    num_scales = sum(
        module.weight.shape[0]
        for module in model.modules()
        if isinstance(
            module,
            (nn.Conv2d, nn.Linear)
        )
    )

    fp32_exceptions = (
        total_params
        - quant_weight_params
    )

    compressed_bits = (
        quant_weight_params
        * weight_bits
        + num_scales
        * 16   # assume FP16 deployment scales
        + fp32_exceptions
        * 32
    )

    # Original teacher size, fixed from original model.
    teacher_params = sum(
        p.numel()
        for p in teacher.parameters()
    )

    original_bits = (
        teacher_params * 32
    )

    model_mb = (
        compressed_bits
        / 8
        / (1024 ** 2)
    )

    overall_cr = (
        original_bits
        / compressed_bits
    )

    result = {
        "width_mult":
            STUDENT_WIDTH,
        "weight_bits":
            weight_bits,
        "activation_bits":
            activation_bits,
        "best_accuracy":
            best_acc,
        "model_size_mb":
            model_mb,
        "overall_cr_vs_original":
            overall_cr,
        "activation_cr":
            32.0 / activation_bits,
    }

    append_csv(
        result
    )

    print()
    print(
        f"QAT W{weight_bits}A{activation_bits}: "
        f"best={best_acc:.2f}% | "
        f"size={model_mb:.4f} MB | "
        f"overallCR={overall_cr:.2f}x"
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(
        SEED
    )

    teacher, teacher_checkpoint = (
        load_teacher()
    )

    train_loader, test_loader = (
        get_dataloaders()
    )

    # --------------------------------------------
    # Stage 1
    # --------------------------------------------

    student, kd_acc = train_kd_student(
        teacher,
        train_loader,
        test_loader
    )

    fp32_student_state = {
        k: v.detach().cpu().clone()
        for k, v in student.state_dict().items()
    }

    # --------------------------------------------
    # Stage 2
    # --------------------------------------------

    print()
    print("=" * 100)
    print("STARTING QAT SWEEP")
    print("=" * 100)

    for cfg in QAT_CONFIGS:
        run_qat(
            fp32_student_state,
            cfg["weight_bits"],
            cfg["activation_bits"],
            train_loader,
            test_loader,
            teacher
        )

    print()
    print("=" * 100)
    print(
        f"All results written to "
        f"{RESULTS_CSV}"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()

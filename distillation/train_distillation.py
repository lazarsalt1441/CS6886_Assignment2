
import csv
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from baseline.config import (
    DEVICE,
    MODEL_PATH,
)

from baseline.data import get_dataloaders
from baseline.model import get_model

from distillation.student_model import (
    get_student_model,
    count_parameters,
    parameter_size_mb_fp32,
)

from distillation.distillation_utils import (
    set_seed,
    distillation_loss,
    hard_label_loss,
    evaluate_model,
    WarmupCosineScheduler,
    current_lr,
)

from distillation.kd_config import (
    STUDENT_WIDTHS,
    TEMPERATURE,
    ALPHA,
    NUM_EPOCHS,
    LEARNING_RATE,
    MOMENTUM,
    WEIGHT_DECAY,
    USE_COSINE_SCHEDULER,
    WARMUP_EPOCHS,
    MAX_GRAD_NORM,
    KD_OUTPUT_DIR,
    KD_RESULTS_CSV,
    RESUME,
    SAVE_EVERY_EPOCH,
    RUN_HARD_LABEL_BASELINE,
    KD_SEED,
)


# ============================================================
# IO
# ============================================================

OUTPUT_DIR = Path(KD_OUTPUT_DIR)
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def width_tag(width):
    return str(width).replace(".", "p")


def result_csv_path():
    return (
        OUTPUT_DIR
        / KD_RESULTS_CSV
    )


def epoch_log_path(
    width,
    mode
):
    return (
        OUTPUT_DIR
        / (
            f"student_w{width_tag(width)}"
            f"_{mode}_epochs.csv"
        )
    )


def latest_checkpoint_path(
    width,
    mode
):
    return (
        OUTPUT_DIR
        / (
            f"student_w{width_tag(width)}"
            f"_{mode}_latest.pth"
        )
    )


def best_checkpoint_path(
    width,
    mode
):
    return (
        OUTPUT_DIR
        / (
            f"student_w{width_tag(width)}"
            f"_{mode}_best.pth"
        )
    )


def append_csv(
    path,
    row
):
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

        writer.writerow(
            row
        )


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
        checkpoint[
            "model_state_dict"
        ]
    )

    teacher = teacher.to(
        DEVICE
    )

    teacher.eval()

    for parameter in teacher.parameters():
        parameter.requires_grad = False

    print(
        f"Teacher checkpoint accuracy: "
        f"{checkpoint['test_accuracy']:.2f}%"
    )

    return (
        teacher,
        checkpoint
    )


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    student,
    teacher,
    train_loader,
    optimizer,
    mode,
):
    student.train()
    teacher.eval()

    running_total_loss = 0.0
    running_hard_loss = 0.0
    running_soft_loss = 0.0

    total = 0
    correct = 0

    start_time = time.time()

    for batch_idx, (
        images,
        labels
    ) in enumerate(
        train_loader
    ):

        images = images.to(
            DEVICE
        )

        labels = labels.to(
            DEVICE
        )

        optimizer.zero_grad()

        student_logits = (
            student(
                images
            )
        )

        if mode == "kd":

            with torch.no_grad():
                teacher_logits = (
                    teacher(
                        images
                    )
                )

            (
                loss,
                hard_loss,
                soft_loss,
            ) = distillation_loss(
                student_logits,
                teacher_logits,
                labels,
                temperature=TEMPERATURE,
                alpha=ALPHA,
            )

        elif mode == "hard":

            loss = hard_label_loss(
                student_logits,
                labels
            )

            hard_loss = loss

            soft_loss = torch.tensor(
                0.0,
                device=DEVICE
            )

        else:
            raise ValueError(
                f"Unknown mode: {mode}"
            )

        loss.backward()

        if MAX_GRAD_NORM is not None:

            torch.nn.utils.clip_grad_norm_(
                student.parameters(),
                MAX_GRAD_NORM
            )

        optimizer.step()

        batch_size = (
            labels.size(0)
        )

        running_total_loss += (
            loss.item()
            * batch_size
        )

        running_hard_loss += (
            hard_loss.item()
            * batch_size
        )

        running_soft_loss += (
            soft_loss.item()
            * batch_size
        )

        total += batch_size

        predictions = (
            student_logits.argmax(
                dim=1
            )
        )

        correct += (
            predictions == labels
        ).sum().item()

        if batch_idx % 50 == 0:

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"Batch "
                f"{batch_idx:03d}/"
                f"{len(train_loader)} "
                f"| total={loss.item():.4f} "
                f"| hard={hard_loss.item():.4f} "
                f"| soft={soft_loss.item():.4f} "
                f"| {elapsed:.1f}s"
            )

    return {
        "train_loss":
            running_total_loss
            / total,

        "train_hard_loss":
            running_hard_loss
            / total,

        "train_soft_loss":
            running_soft_loss
            / total,

        "train_accuracy":
            100.0
            * correct
            / total,
    }


# ============================================================
# CHECKPOINTING
# ============================================================

def save_checkpoint(
    path,
    width,
    mode,
    epoch,
    student,
    optimizer,
    best_accuracy,
):
    torch.save(
        {
            "width_mult":
                width,

            "mode":
                mode,

            "epoch":
                epoch,

            "student_state_dict":
                student.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),

            "best_test_accuracy":
                best_accuracy,

            "temperature":
                TEMPERATURE,

            "alpha":
                ALPHA,
        },
        path,
    )


def maybe_resume(
    width,
    mode,
    student,
    optimizer,
):
    path = latest_checkpoint_path(
        width,
        mode
    )

    if (
        not RESUME
        or not path.exists()
    ):
        return (
            0,
            0.0
        )

    print(
        f"Resuming from: "
        f"{path}"
    )

    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

    student.load_state_dict(
        checkpoint[
            "student_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    start_epoch = (
        checkpoint["epoch"]
    )

    best_accuracy = (
        checkpoint[
            "best_test_accuracy"
        ]
    )

    return (
        start_epoch,
        best_accuracy
    )


# ============================================================
# TRAIN ONE STUDENT
# ============================================================

def train_student(
    width,
    mode,
    teacher,
    train_loader,
    test_loader,
):
    print()
    print("=" * 90)
    print(
        f"STUDENT width={width} "
        f"| mode={mode}"
    )
    print("=" * 90)

    student = get_student_model(
        width_mult=width
    ).to(
        DEVICE
    )

    total_params = (
        count_parameters(
            student
        )
    )

    fp32_mb = (
        parameter_size_mb_fp32(
            student
        )
    )

    print(
        f"Student parameters: "
        f"{total_params:,}"
    )

    print(
        f"Student FP32 size: "
        f"{fp32_mb:.4f} MB"
    )

    optimizer = optim.SGD(
        student.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = None

    if USE_COSINE_SCHEDULER:

        scheduler = (
            WarmupCosineScheduler(
                optimizer,
                base_lr=LEARNING_RATE,
                total_epochs=NUM_EPOCHS,
                warmup_epochs=WARMUP_EPOCHS,
            )
        )

    (
        start_epoch,
        best_accuracy,
    ) = maybe_resume(
        width,
        mode,
        student,
        optimizer,
    )

    # Ensure the resumed run gets a sensible LR
    # corresponding to the current epoch.
    if (
        scheduler is not None
        and start_epoch > 0
    ):

        for e in range(
            start_epoch
        ):
            scheduler.step(e)

    # Evaluate random/resumed student before training.
    initial_loss, initial_accuracy = (
        evaluate_model(
            student,
            test_loader,
            DEVICE
        )
    )

    print(
        f"Starting test accuracy: "
        f"{initial_accuracy:.2f}%"
    )

    run_start = time.time()

    for epoch in range(
        start_epoch,
        NUM_EPOCHS
    ):

        print()
        print(
            f"Epoch "
            f"{epoch + 1}/"
            f"{NUM_EPOCHS}"
        )

        print("-" * 90)

        epoch_start = time.time()

        train_stats = (
            train_one_epoch(
                student,
                teacher,
                train_loader,
                optimizer,
                mode,
            )
        )

        test_loss, test_accuracy = (
            evaluate_model(
                student,
                test_loader,
                DEVICE
            )
        )

        if scheduler is not None:

            new_lr = scheduler.step(
                epoch
            )

        else:

            new_lr = current_lr(
                optimizer
            )

        epoch_seconds = (
            time.time()
            - epoch_start
        )

        print()
        print(
            f"Train loss:     "
            f"{train_stats['train_loss']:.4f}"
        )

        print(
            f"Hard loss:      "
            f"{train_stats['train_hard_loss']:.4f}"
        )

        print(
            f"Soft KD loss:   "
            f"{train_stats['train_soft_loss']:.4f}"
        )

        print(
            f"Train accuracy: "
            f"{train_stats['train_accuracy']:.2f}%"
        )

        print(
            f"Test loss:      "
            f"{test_loss:.4f}"
        )

        print(
            f"Test accuracy:  "
            f"{test_accuracy:.2f}%"
        )

        print(
            f"LR:             "
            f"{new_lr:.7f}"
        )

        print(
            f"Epoch time:     "
            f"{epoch_seconds:.1f}s"
        )

        epoch_row = {
            "width_mult":
                width,

            "mode":
                mode,

            "epoch":
                epoch + 1,

            "train_loss":
                train_stats[
                    "train_loss"
                ],

            "train_hard_loss":
                train_stats[
                    "train_hard_loss"
                ],

            "train_soft_loss":
                train_stats[
                    "train_soft_loss"
                ],

            "train_accuracy":
                train_stats[
                    "train_accuracy"
                ],

            "test_loss":
                test_loss,

            "test_accuracy":
                test_accuracy,

            "learning_rate":
                new_lr,

            "epoch_seconds":
                epoch_seconds,
        }

        append_csv(
            epoch_log_path(
                width,
                mode
            ),
            epoch_row
        )

        # ----------------------------------------------
        # Best checkpoint
        # ----------------------------------------------

        if (
            test_accuracy
            > best_accuracy
        ):

            best_accuracy = (
                test_accuracy
            )

            save_checkpoint(
                best_checkpoint_path(
                    width,
                    mode
                ),
                width,
                mode,
                epoch + 1,
                student,
                optimizer,
                best_accuracy,
            )

            print(
                f"Saved BEST student "
                f"({best_accuracy:.2f}%)"
            )

        # ----------------------------------------------
        # Latest checkpoint for resume
        # ----------------------------------------------

        if SAVE_EVERY_EPOCH:

            save_checkpoint(
                latest_checkpoint_path(
                    width,
                    mode
                ),
                width,
                mode,
                epoch + 1,
                student,
                optimizer,
                best_accuracy,
            )

    total_seconds = (
        time.time()
        - run_start
    )

    # --------------------------------------------------------
    # Reload BEST checkpoint for final evaluation.
    # --------------------------------------------------------

    best_path = best_checkpoint_path(
        width,
        mode
    )

    best_checkpoint = torch.load(
        best_path,
        map_location=DEVICE
    )

    student.load_state_dict(
        best_checkpoint[
            "student_state_dict"
        ]
    )

    final_loss, final_accuracy = (
        evaluate_model(
            student,
            test_loader,
            DEVICE
        )
    )

    summary = {
        "width_mult":
            width,

        "mode":
            mode,

        "parameters":
            total_params,

        "fp32_size_mb":
            fp32_mb,

        "best_test_accuracy":
            final_accuracy,

        "teacher_accuracy":
            95.11,

        "accuracy_gap_to_teacher_pp":
            95.11
            - final_accuracy,

        "temperature":
            TEMPERATURE
            if mode == "kd"
            else 0.0,

        "alpha":
            ALPHA
            if mode == "kd"
            else 1.0,

        "epochs":
            NUM_EPOCHS,

        "runtime_seconds":
            total_seconds,

        "best_checkpoint":
            str(best_path),
    }

    append_csv(
        result_csv_path(),
        summary
    )

    print()
    print("=" * 90)

    print(
        f"FINAL width={width}, "
        f"mode={mode}"
    )

    print("=" * 90)

    print(
        f"Best accuracy:  "
        f"{final_accuracy:.2f}%"
    )

    print(
        f"Teacher gap:    "
        f"{95.11-final_accuracy:.2f} pp"
    )

    print(
        f"FP32 size:      "
        f"{fp32_mb:.4f} MB"
    )

    print(
        f"Checkpoint:     "
        f"{best_path}"
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():
    set_seed(
        KD_SEED
    )

    print("=" * 90)
    print("KNOWLEDGE DISTILLATION OVERNIGHT RUN")
    print("=" * 90)

    print(
        f"Device:          {DEVICE}"
    )

    print(
        f"Student widths:  {STUDENT_WIDTHS}"
    )

    print(
        f"Temperature:     {TEMPERATURE}"
    )

    print(
        f"Alpha:           {ALPHA}"
    )

    print(
        f"Epochs/student:  {NUM_EPOCHS}"
    )

    print(
        f"Output dir:      {OUTPUT_DIR}"
    )

    print("=" * 90)

    teacher, teacher_checkpoint = (
        load_teacher()
    )

    train_loader, test_loader = (
        get_dataloaders()
    )

    # Verify teacher on the exact current test pipeline.
    teacher_loss, teacher_accuracy = (
        evaluate_model(
            teacher,
            test_loader,
            DEVICE
        )
    )

    print(
        f"Teacher re-evaluated accuracy: "
        f"{teacher_accuracy:.2f}%"
    )

    all_summaries = []

    for width in STUDENT_WIDTHS:

        # ----------------------------------------------
        # Main KD run
        # ----------------------------------------------

        summary = train_student(
            width,
            "kd",
            teacher,
            train_loader,
            test_loader,
        )

        all_summaries.append(
            summary
        )

        # ----------------------------------------------
        # Optional hard-label-only comparison
        # ----------------------------------------------

        if RUN_HARD_LABEL_BASELINE:

            baseline_summary = (
                train_student(
                    width,
                    "hard",
                    teacher,
                    train_loader,
                    test_loader,
                )
            )

            all_summaries.append(
                baseline_summary
            )

    print()
    print("=" * 110)
    print("ALL DISTILLATION RUNS COMPLETE")
    print("=" * 110)

    print(
        f"{'width':>8s} "
        f"{'mode':>8s} "
        f"{'params':>12s} "
        f"{'FP32 MB':>10s} "
        f"{'accuracy':>10s} "
        f"{'teacher gap':>12s}"
    )

    print("-" * 110)

    for s in all_summaries:

        print(
            f"{s['width_mult']:8.2f} "
            f"{s['mode']:>8s} "
            f"{s['parameters']:12,d} "
            f"{s['fp32_size_mb']:10.4f} "
            f"{s['best_test_accuracy']:9.2f}% "
            f"{s['accuracy_gap_to_teacher_pp']:11.2f} pp"
        )

    print()
    print(
        f"Summary CSV: "
        f"{result_csv_path()}"
    )

    print("=" * 110)


if __name__ == "__main__":
    main()


import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def distillation_loss(
    student_logits,
    teacher_logits,
    labels,
    temperature=4.0,
    alpha=0.3
):
    """
    Hinton-style knowledge distillation.

    Hard-label component:
        CE(student_logits, labels)

    Soft teacher component:
        T^2 * KL(
            softmax(teacher/T)
            ||
            softmax(student/T)
        )

    PyTorch KLDiv expects log probabilities as the first argument
    and probabilities as the second.
    """

    hard_loss = F.cross_entropy(
        student_logits,
        labels
    )

    T = temperature

    student_log_probs = F.log_softmax(
        student_logits / T,
        dim=1
    )

    teacher_probs = F.softmax(
        teacher_logits / T,
        dim=1
    )

    soft_loss = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="batchmean"
    ) * (T * T)

    total_loss = (
        alpha * hard_loss
        + (1.0 - alpha) * soft_loss
    )

    return (
        total_loss,
        hard_loss,
        soft_loss
    )


def hard_label_loss(
    student_logits,
    labels
):
    return F.cross_entropy(
        student_logits,
        labels
    )


@torch.no_grad()
def evaluate_model(
    model,
    dataloader,
    device
):
    model.eval()

    total_loss = 0.0
    total = 0
    correct = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        loss = F.cross_entropy(
            logits,
            labels
        )

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        predictions = logits.argmax(
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

    return (
        total_loss / total,
        100.0 * correct / total
    )


class WarmupCosineScheduler:
    """
    Small self-contained warmup + cosine LR scheduler.

    For the first warmup_epochs:
        lr rises approximately linearly from base_lr/warmup_epochs
        to base_lr.

    Afterwards:
        cosine decay to zero.
    """

    def __init__(
        self,
        optimizer,
        base_lr,
        total_epochs,
        warmup_epochs=5
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs

    def step(self, epoch):
        # epoch is zero-indexed and called AFTER the epoch.
        next_epoch = epoch + 1

        if (
            self.warmup_epochs > 0
            and next_epoch <= self.warmup_epochs
        ):
            scale = (
                next_epoch
                / self.warmup_epochs
            )

            lr = (
                self.base_lr
                * scale
            )

        else:
            denom = max(
                1,
                self.total_epochs
                - self.warmup_epochs
            )

            progress = (
                next_epoch
                - self.warmup_epochs
            ) / denom

            progress = min(
                max(progress, 0.0),
                1.0
            )

            lr = (
                0.5
                * self.base_lr
                * (
                    1.0
                    + math.cos(
                        math.pi * progress
                    )
                )
            )

        for group in self.optimizer.param_groups:
            group["lr"] = lr

        return lr


def current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]

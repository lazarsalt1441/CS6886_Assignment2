import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from baseline.config import (
    SEED,
    DEVICE,
    NUM_EPOCHS,
    LEARNING_RATE,
    MOMENTUM,
    WEIGHT_DECAY,
    MODEL_PATH
)

from baseline.data import get_dataloaders
from baseline.model import get_model
from baseline.evaluate import evaluate


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0.0

    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(dataloader):

        images = images.to(device)
        labels = labels.to(device)

        # -------------------------
        # Clear previous gradients
        # -------------------------
        optimizer.zero_grad()

        # -------------------------
        # Forward pass
        # -------------------------
        outputs = model(images)

        # -------------------------
        # Compute loss
        # -------------------------
        loss = criterion(
            outputs,
            labels
        )

        # -------------------------
        # Backpropagation
        # -------------------------
        loss.backward()

        # -------------------------
        # Update weights
        # -------------------------
        optimizer.step()

        # -------------------------
        # Statistics
        # -------------------------
        total_loss += (
            loss.item() * images.size(0)
        )

        predicted = outputs.argmax(dim=1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        if batch_idx % 50 == 0:

            print(
                f"Batch "
                f"{batch_idx:03d}/"
                f"{len(dataloader)} "
                f"| Loss: "
                f"{loss.item():.4f}"
            )

    avg_loss = total_loss / total

    accuracy = (
        100.0 * correct / total
    )

    return avg_loss, accuracy


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print(
        f"Using device: {DEVICE}"
    )

    # -------------------------
    # Data
    # -------------------------

    train_loader, test_loader = (
        get_dataloaders()
    )

    print(
        f"Training samples: "
        f"{len(train_loader.dataset)}"
    )

    print(
        f"Test samples: "
        f"{len(test_loader.dataset)}"
    )

    # -------------------------
    # Model
    # -------------------------

    model = get_model()

    model = model.to(DEVICE)

    # -------------------------
    # Loss
    # -------------------------

    criterion = nn.CrossEntropyLoss()

    # -------------------------
    # Optimizer
    # -------------------------

    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # -------------------------
    # LR scheduler
    # -------------------------

    scheduler = (
        optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=NUM_EPOCHS
        )
    )

    # -------------------------
    # Statistics
    # -------------------------

    best_test_acc = 0.0

    train_losses = []
    train_accuracies = []

    test_losses = []
    test_accuracies = []

    # ========================================================
    # TRAIN
    # ========================================================

    for epoch in range(NUM_EPOCHS):

        print()
        print("=" * 60)

        print(
            f"Epoch "
            f"{epoch + 1}/"
            f"{NUM_EPOCHS}"
        )

        print("=" * 60)

        # -------------------------
        # Training
        # -------------------------

        train_loss, train_acc = (
            train_one_epoch(
                model,
                train_loader,
                criterion,
                optimizer,
                DEVICE
            )
        )

        # -------------------------
        # Testing
        # -------------------------

        test_loss, test_acc = evaluate(
            model,
            test_loader,
            criterion,
            DEVICE
        )

        # -------------------------
        # Learning-rate update
        # -------------------------

        scheduler.step()

        # -------------------------
        # Store results
        # -------------------------

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # -------------------------
        # Print results
        # -------------------------

        print()

        print(
            f"Train loss: "
            f"{train_loss:.4f}"
        )

        print(
            f"Train accuracy: "
            f"{train_acc:.2f}%"
        )

        print(
            f"Test loss: "
            f"{test_loss:.4f}"
        )

        print(
            f"Test accuracy: "
            f"{test_acc:.2f}%"
        )

        print(
            f"Learning rate: "
            f"{optimizer.param_groups[0]['lr']:.6f}"
        )

        # -------------------------
        # Save best model
        # -------------------------

        if test_acc > best_test_acc:

            best_test_acc = test_acc

            torch.save(
                {
                    "epoch": epoch + 1,

                    "model_state_dict":
                        model.state_dict(),

                    "optimizer_state_dict":
                        optimizer.state_dict(),

                    "test_accuracy":
                        test_acc,

                    "train_losses":
                        train_losses,

                    "train_accuracies":
                        train_accuracies,

                    "test_losses":
                        test_losses,

                    "test_accuracies":
                        test_accuracies
                },
                MODEL_PATH
            )

            print(
                f"Saved best model! "
                f"Accuracy = "
                f"{best_test_acc:.2f}%"
            )

    print()
    print("=" * 60)

    print("Training complete.")

    print(
        f"Best test accuracy: "
        f"{best_test_acc:.2f}%"
    )


if __name__ == "__main__":
    main()
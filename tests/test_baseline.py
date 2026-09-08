import torch
import torch.nn as nn

from baseline.model import get_model
from baseline.data import get_dataloaders
from baseline.evaluate import evaluate
from baseline.config import DEVICE, MODEL_PATH


def main():

    print(f"Using device: {DEVICE}")

    # Build the same MobileNetV2 architecture
    model = get_model()

    # Load our trained checkpoint
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    # Put the trained weights into MobileNet
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    print(
        f"Checkpoint saved at epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Stored test accuracy: "
        f"{checkpoint['test_accuracy']:.2f}%"
    )

    # Get CIFAR-10
    _, test_loader = get_dataloaders()

    criterion = nn.CrossEntropyLoss()

    # Actually run inference again
    test_loss, test_acc = evaluate(
        model,
        test_loader,
        criterion,
        DEVICE
    )

    print()
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.2f}%")


if __name__ == "__main__":
    main()
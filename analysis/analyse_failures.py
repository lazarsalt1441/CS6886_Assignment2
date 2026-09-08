import torch
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from baseline.model import get_model
from baseline.data import get_dataloaders
from baseline.config import DEVICE, MODEL_PATH


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def main():
    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Test loader
    # --------------------------------------------------------
    _, test_loader = get_dataloaders()

    confusion = Counter()

    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    misclassified = []

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            preds = outputs.argmax(dim=1)

            for i in range(len(labels)):

                true_label = labels[i].item()
                pred_label = preds[i].item()

                class_total[true_label] += 1

                if true_label == pred_label:
                    class_correct[true_label] += 1

                else:
                    confusion[(true_label, pred_label)] += 1

                    if len(misclassified) < 25:
                        misclassified.append(
                            (
                                images[i].cpu(),
                                true_label,
                                pred_label
                            )
                        )

    # --------------------------------------------------------
    # Per-class accuracy
    # --------------------------------------------------------
    print("\nPER-CLASS ACCURACY")
    print("=" * 50)

    for i, name in enumerate(CIFAR10_CLASSES):

        acc = (
            100.0
            * class_correct[i]
            / class_total[i]
        )

        print(
            f"{name:12s}: "
            f"{acc:.2f}%"
        )

    # --------------------------------------------------------
    # Most common confusion pairs
    # --------------------------------------------------------
    print("\nMOST COMMON MISCLASSIFICATIONS")
    print("=" * 50)

    for (true_label, pred_label), count in confusion.most_common(15):

        print(
            f"{CIFAR10_CLASSES[true_label]:12s}"
            f" -> "
            f"{CIFAR10_CLASSES[pred_label]:12s}"
            f": {count}"
        )

    # --------------------------------------------------------
    # Plot example failures
    # --------------------------------------------------------

    # Undo ImageNet normalization
    mean = torch.tensor(
        [0.485, 0.456, 0.406]
    ).view(3, 1, 1)

    std = torch.tensor(
        [0.229, 0.224, 0.225]
    ).view(3, 1, 1)

    fig, axes = plt.subplots(
        5,
        5,
        figsize=(12, 12)
    )

    for ax, (img, true_label, pred_label) in zip(
        axes.flat,
        misclassified
    ):

        img = img * std + mean
        img = img.clamp(0, 1)

        ax.imshow(
            img.permute(1, 2, 0)
        )

        ax.set_title(
            f"T: {CIFAR10_CLASSES[true_label]}\n"
            f"P: {CIFAR10_CLASSES[pred_label]}",
            fontsize=8
        )

        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        "misclassified_examples.png",
        dpi=300,
        bbox_inches="tight"
    )

    print(
        "\nSaved figure as "
        "misclassified_examples.png"
    )


if __name__ == "__main__":
    main()
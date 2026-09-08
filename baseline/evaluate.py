import torch


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0

    correct = 0
    total = 0

    for images, labels in dataloader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        total_loss += (
            loss.item() * images.size(0)
        )

        predicted = outputs.argmax(dim=1)

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    avg_loss = total_loss / total

    accuracy = (
        100.0 * correct / total
    )

    return avg_loss, accuracy
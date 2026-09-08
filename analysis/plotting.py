import torch
import matplotlib.pyplot as plt

checkpoint = torch.load("best_mobilenetv2_cifar10.pth", map_location="cpu")

train_losses = checkpoint["train_losses"]
test_losses = checkpoint["test_losses"]
train_acc = checkpoint["train_accuracies"]
test_acc = checkpoint["test_accuracies"]

epochs = range(1, len(train_losses) + 1)

plt.figure()
plt.plot(epochs, train_losses, label="Train Loss")
plt.plot(epochs, test_losses, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.savefig("loss_curve.png", dpi=300, bbox_inches="tight")
plt.show()

plt.figure()
plt.plot(epochs, train_acc, label="Train Accuracy")
plt.plot(epochs, test_acc, label="Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.grid()
plt.savefig("accuracy_curve.png", dpi=300, bbox_inches="tight")
plt.show()
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from baseline.config import (
    DATA_DIR,
    IMAGE_SIZE,
    BATCH_SIZE,
    NUM_WORKERS
)


# ImageNet normalization
# because MobileNetV2 was pretrained on ImageNet
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


train_transform = transforms.Compose([

    # CIFAR images are originally 32x32
    transforms.Resize(IMAGE_SIZE),

    # Data augmentation
    transforms.RandomCrop(
        IMAGE_SIZE,
        padding=4
    ),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])


test_transform = transforms.Compose([

    transforms.Resize(IMAGE_SIZE),

    transforms.ToTensor(),

    transforms.Normalize(
        IMAGENET_MEAN,
        IMAGENET_STD
    )
])


def get_dataloaders():

    train_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, test_loader
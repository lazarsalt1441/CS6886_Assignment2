
import torch.nn as nn
from torchvision.models import mobilenet_v2


def get_student_model(
    width_mult=0.5,
    num_classes=10,
    dropout=0.2
):
    """
    Smaller MobileNetV2 student.

    We use torchvision's MobileNetV2 implementation with a reduced
    width multiplier. Official ImageNet pretrained weights are not
    loaded here because torchvision's standard pretrained MobileNetV2
    checkpoint is for width_mult=1.0, and tensor shapes differ.

    The student learns from:
      - CIFAR-10 hard labels
      - teacher soft targets
    """

    model = mobilenet_v2(
        weights=None,
        width_mult=width_mult,
        dropout=dropout
    )

    model.classifier[1] = nn.Linear(
        model.last_channel,
        num_classes
    )

    return model


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


def parameter_size_mb_fp32(model):
    total_params = count_parameters(model)

    return (
        total_params
        * 32
        / 8
        / (1024 ** 2)
    )

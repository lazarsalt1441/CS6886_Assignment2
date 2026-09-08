import torch.nn as nn

from torchvision.models import (
    mobilenet_v2,
    MobileNet_V2_Weights
)

from baseline.config import NUM_CLASSES


def get_model():

    # Load pretrained ImageNet weights
    weights = MobileNet_V2_Weights.DEFAULT

    model = mobilenet_v2(
        weights=weights
    )

    # Original final layer:
    #
    # Linear(1280, 1000)
    #
    # because ImageNet has 1000 classes.
    #
    # Replace it with CIFAR-10 classifier.

    model.classifier[1] = nn.Linear(
        model.last_channel,
        NUM_CLASSES
    )

    return model
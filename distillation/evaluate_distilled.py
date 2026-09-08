
import sys
import torch

from baseline.config import DEVICE
from baseline.data import get_dataloaders
from distillation.student_model import (
    get_student_model,
    count_parameters,
    parameter_size_mb_fp32,
)
from distillation.distillation_utils import evaluate_model


def main():
    if len(sys.argv) != 2:
        print(
            "Usage:\n"
            "python3 evaluate_distilled.py "
            "kd_outputs/student_w0p5_kd_best.pth"
        )
        raise SystemExit(1)

    path = sys.argv[1]

    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

    width = checkpoint[
        "width_mult"
    ]

    model = get_student_model(
        width_mult=width
    ).to(
        DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "student_state_dict"
        ]
    )

    _, test_loader = (
        get_dataloaders()
    )

    loss, accuracy = (
        evaluate_model(
            model,
            test_loader,
            DEVICE
        )
    )

    print(
        f"Width multiplier: "
        f"{width}"
    )

    print(
        f"Parameters: "
        f"{count_parameters(model):,}"
    )

    print(
        f"FP32 size: "
        f"{parameter_size_mb_fp32(model):.4f} MB"
    )

    print(
        f"Test loss: "
        f"{loss:.4f}"
    )

    print(
        f"Test accuracy: "
        f"{accuracy:.2f}%"
    )


if __name__ == "__main__":
    main()

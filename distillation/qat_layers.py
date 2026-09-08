
import torch
import torch.nn as nn
import torch.nn.functional as F


def ste_symmetric_quantize_per_channel(weight, bits):
    """
    Symmetric per-output-channel fake quantization with STE.

    Returns a float tensor whose forward values lie on the quantization
    grid, but whose backward gradient wrt weight is approximately identity.
    """
    if bits >= 32:
        return weight

    qmax = (2 ** (bits - 1)) - 1

    # Per-output-channel scale.
    reduce_dims = tuple(range(1, weight.dim()))

    max_abs = weight.detach().abs().amax(
        dim=reduce_dims,
        keepdim=True
    )

    scale = max_abs / qmax
    scale = torch.where(
        scale == 0,
        torch.ones_like(scale),
        scale
    )

    q = torch.round(weight / scale)
    q = torch.clamp(q, -qmax, qmax)

    dequant = q * scale

    # Straight-through estimator:
    # forward = dequant
    # backward d(output)/d(weight) ~= 1
    return weight + (dequant - weight).detach()


def ste_symmetric_quantize_activation(x, bits):
    """
    Dynamic symmetric per-tensor activation fake quantization with STE.
    """
    if bits >= 32:
        return x

    qmax = (2 ** (bits - 1)) - 1

    max_abs = x.detach().abs().max()

    if max_abs.item() == 0:
        return x

    scale = max_abs / qmax

    q = torch.round(x / scale)
    q = torch.clamp(q, -qmax, qmax)

    dequant = q * scale

    return x + (dequant - x).detach()


class QATConv2d(nn.Conv2d):
    def __init__(self, *args, weight_bits=3, activation_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits

    @classmethod
    def from_conv(cls, conv, weight_bits, activation_bits):
        new = cls(
            in_channels=conv.in_channels,
            out_channels=conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            groups=conv.groups,
            bias=(conv.bias is not None),
            padding_mode=conv.padding_mode,
            weight_bits=weight_bits,
            activation_bits=activation_bits,
        )

        new.weight.data.copy_(conv.weight.data)

        if conv.bias is not None:
            new.bias.data.copy_(conv.bias.data)

        return new

    def forward(self, x):
        qw = ste_symmetric_quantize_per_channel(
            self.weight,
            self.weight_bits
        )

        y = F.conv2d(
            x,
            qw,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )

        return ste_symmetric_quantize_activation(
            y,
            self.activation_bits
        )


class QATLinear(nn.Linear):
    def __init__(self, *args, weight_bits=3, activation_bits=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_bits = weight_bits
        self.activation_bits = activation_bits

    @classmethod
    def from_linear(cls, linear, weight_bits, activation_bits):
        new = cls(
            linear.in_features,
            linear.out_features,
            bias=(linear.bias is not None),
            weight_bits=weight_bits,
            activation_bits=activation_bits,
        )

        new.weight.data.copy_(linear.weight.data)

        if linear.bias is not None:
            new.bias.data.copy_(linear.bias.data)

        return new

    def forward(self, x):
        qw = ste_symmetric_quantize_per_channel(
            self.weight,
            self.weight_bits
        )

        y = F.linear(
            x,
            qw,
            self.bias
        )

        return ste_symmetric_quantize_activation(
            y,
            self.activation_bits
        )


def convert_to_qat(module, weight_bits=3, activation_bits=8):
    """
    Recursively replace Conv2d/Linear layers with manual QAT versions.
    BatchNorm and all other modules are untouched.
    """
    for name, child in list(module.named_children()):

        if isinstance(child, nn.Conv2d):
            replacement = QATConv2d.from_conv(
                child,
                weight_bits,
                activation_bits
            )
            setattr(module, name, replacement)

        elif isinstance(child, nn.Linear):
            replacement = QATLinear.from_linear(
                child,
                weight_bits,
                activation_bits
            )
            setattr(module, name, replacement)

        else:
            convert_to_qat(
                child,
                weight_bits,
                activation_bits
            )

    return module

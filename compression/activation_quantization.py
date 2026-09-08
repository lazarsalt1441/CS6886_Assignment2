import torch


def symmetric_quantize_activation(x, bits=8):
    """
    Symmetric per-tensor activation quantization.

    Simulates low-bit activation storage:
        FP32 activation
            -> quantized integer levels
            -> dequantized FP32 activation
    """

    if bits < 2:
        raise ValueError("bits must be >= 2")

    # Signed integer range
    # 8 bit -> [-127, 127]
    # 6 bit -> [-31, 31]
    # 4 bit -> [-7, 7]
    qmax = (2 ** (bits - 1)) - 1
    qmin = -qmax

    # Find largest activation magnitude
    max_abs = x.abs().max()

    # Handle an all-zero activation tensor
    if max_abs.item() == 0:
        return x

    # Calculate scale
    scale = max_abs / qmax

    # Quantize
    q = torch.round(x / scale)

    # Make sure integer value fits in our bit range
    q = torch.clamp(q, qmin, qmax)

    # Dequantize
    x_dequantized = q * scale

    return x_dequantized
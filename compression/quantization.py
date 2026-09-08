import torch


def symmetric_quantize_tensor(x, bits=8):
    """
    Symmetric per-tensor quantization.

    Args:
        x: torch.Tensor
        bits: number of quantization bits

    Returns:
        x_dequant: dequantized tensor
        scale: scale factor
        q: integer-valued quantized tensor
    """

    if bits < 2:
        raise ValueError("bits must be >= 2")

    # Signed symmetric integer range
    qmax = (2 ** (bits - 1)) - 1
    qmin = -qmax

    max_abs = x.abs().max()

    # Avoid division by zero for all-zero tensors
    if max_abs == 0:
        scale = torch.tensor(
            1.0,
            device=x.device,
            dtype=x.dtype
        )

        q = torch.zeros_like(x)

        return x.clone(), scale, q

    # Scale maps max absolute value -> qmax
    scale = max_abs / qmax

    # Quantize
    q = torch.round(x / scale)

    # Clip to valid integer range
    q = torch.clamp(q, qmin, qmax)

    # Dequantize
    x_dequant = q * scale

    return x_dequant, scale, q

def symmetric_quantize_per_channel(weight, bits=8):
    """
    Per-output-channel symmetric quantization.

    For Conv2d:
        weight shape = [C_out, C_in, K, K]

    For Linear:
        weight shape = [C_out, C_in]
    """

    if bits < 2:
        raise ValueError("bits must be >= 2")

    qmax = (2 ** (bits - 1)) - 1
    qmin = -qmax

    # Flatten everything except output-channel dimension
    flat = weight.reshape(weight.shape[0], -1)

    # Maximum magnitude separately for each output channel
    max_abs = flat.abs().max(dim=1).values

    # Prevent division by zero
    max_abs = torch.where(
        max_abs == 0,
        torch.ones_like(max_abs),
        max_abs
    )

    scales = max_abs / qmax

    # Reshape scales so broadcasting works
    shape = [weight.shape[0]] + [1] * (weight.dim() - 1)
    scales_view = scales.reshape(shape)

    q = torch.round(weight / scales_view)
    q = torch.clamp(q, qmin, qmax)

    dequantized = q * scales_view

    return dequantized, scales, q
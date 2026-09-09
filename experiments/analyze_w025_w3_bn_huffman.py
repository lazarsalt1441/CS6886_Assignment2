
import copy
import heapq
from collections import Counter

import torch
import torch.nn as nn

from baseline.config import DEVICE, MODEL_PATH
from baseline.data import get_dataloaders
from baseline.model import get_model
from distillation.student_model import get_student_model
from distillation.qat_layers import convert_to_qat


KD025_BEST = "checkpoints/student_w025_kd_best.pth"
W3_QAT_BEST = "checkpoints/student_w025_W3A8_qat_best.pth"

STUDENT_WIDTH = 0.25
WEIGHT_BITS = 3
ACTIVATION_BITS = 8

WEIGHT_SCALE_BITS = 16
HUFFMAN_CODELEN_BITS = 8
BIAS_BITS_AGGRESSIVE = 16
TRY_INT16_BIASES = True


def bits_to_mb(bits):
    return bits / 8 / (1024 ** 2)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total = 0
    correct = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        logits = model(images)
        pred = logits.argmax(dim=1)

        total += labels.numel()
        correct += (pred == labels).sum().item()

    return 100.0 * correct / total


def load_teacher_reference():
    teacher = get_model()
    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    teacher.load_state_dict(ckpt["model_state_dict"])

    params = count_params(teacher)
    bits = params * 32

    return {
        "accuracy": float(ckpt["test_accuracy"]),
        "params": params,
        "bits": bits,
        "mb": bits_to_mb(bits),
    }


def load_kd025():
    ckpt = torch.load(KD025_BEST, map_location=DEVICE)
    width = float(ckpt.get("width_mult", STUDENT_WIDTH))

    model = get_student_model(width_mult=width).to(DEVICE)
    model.load_state_dict(ckpt["student_state_dict"])
    model.eval()

    return model, ckpt


def load_w3_qat():
    ckpt = torch.load(W3_QAT_BEST, map_location=DEVICE)

    width = float(ckpt.get("width_mult", STUDENT_WIDTH))
    wbits = int(ckpt.get("weight_bits", WEIGHT_BITS))
    abits = int(ckpt.get("activation_bits", ACTIVATION_BITS))

    model = get_student_model(width_mult=width)

    model = convert_to_qat(
        model,
        weight_bits=wbits,
        activation_bits=abits
    )

    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    return model, ckpt


def fuse_conv_bn_inplace(conv, bn):
    W = conv.weight.detach()

    if conv.bias is None:
        b = torch.zeros(
            conv.out_channels,
            device=W.device,
            dtype=W.dtype
        )
    else:
        b = conv.bias.detach()

    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    eps = bn.eps

    invstd = torch.rsqrt(var + eps)
    multiplier = gamma * invstd

    reshape = [conv.out_channels] + [1] * (W.dim() - 1)

    W_fused = W * multiplier.reshape(reshape)
    b_fused = beta + (b - mean) * multiplier

    conv.weight.data.copy_(W_fused)

    if conv.bias is None:
        conv.bias = nn.Parameter(b_fused.clone())
    else:
        conv.bias.data.copy_(b_fused)


def fold_bn_recursive(module):
    folded = 0

    for child in module.children():
        folded += fold_bn_recursive(child)

    if isinstance(module, nn.Sequential):
        names = list(module._modules.keys())

        for i in range(len(names) - 1):
            a = module._modules[names[i]]
            b = module._modules[names[i + 1]]

            if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                fuse_conv_bn_inplace(a, b)
                module._modules[names[i + 1]] = nn.Identity()
                folded += 1

    return folded


def quantize_weight_symbols_per_channel(weight, bits):
    qmax = (2 ** (bits - 1)) - 1

    reduce_dims = tuple(range(1, weight.dim()))

    max_abs = weight.detach().abs().amax(
        dim=reduce_dims,
        keepdim=True
    )

    scales = max_abs / qmax
    scales = torch.where(
        scales == 0,
        torch.ones_like(scales),
        scales
    )

    q = torch.round(weight.detach() / scales)
    q = torch.clamp(q, -qmax, qmax)

    return q.to(torch.int32), scales.squeeze()


def huffman_code_lengths(counter):
    if len(counter) == 0:
        return {}

    if len(counter) == 1:
        symbol = next(iter(counter))
        return {symbol: 1}

    heap = []
    uid = 0

    for symbol, freq in counter.items():
        heapq.heappush(
            heap,
            (freq, uid, ("leaf", symbol))
        )
        uid += 1

    while len(heap) > 1:
        f1, _, n1 = heapq.heappop(heap)
        f2, _, n2 = heapq.heappop(heap)

        heapq.heappush(
            heap,
            (f1 + f2, uid, ("node", n1, n2))
        )
        uid += 1

    root = heap[0][2]
    lengths = {}
    stack = [(root, 0)]

    while stack:
        node, depth = stack.pop()

        if node[0] == "leaf":
            lengths[node[1]] = max(depth, 1)
        else:
            _, left, right = node
            stack.append((left, depth + 1))
            stack.append((right, depth + 1))

    return lengths


def huffman_storage_for_model(model):
    stream_bits = 0
    metadata_bits = 0
    scale_bits = 0
    bias_fp32_bits = 0
    quant_weight_count = 0
    bias_count = 0

    for module in model.modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        if module.weight is None:
            continue

        q, scales = quantize_weight_symbols_per_channel(
            module.weight.data,
            WEIGHT_BITS
        )

        symbols = q.flatten().cpu().tolist()
        counter = Counter(symbols)
        lengths = huffman_code_lengths(counter)

        layer_stream_bits = sum(
            freq * lengths[symbol]
            for symbol, freq in counter.items()
        )

        distinct = len(counter)

        layer_metadata_bits = (
            distinct
            * (WEIGHT_BITS + HUFFMAN_CODELEN_BITS)
        )

        stream_bits += layer_stream_bits
        metadata_bits += layer_metadata_bits
        scale_bits += scales.numel() * WEIGHT_SCALE_BITS
        quant_weight_count += module.weight.numel()

        if module.bias is not None:
            bias_count += module.bias.numel()
            bias_fp32_bits += module.bias.numel() * 32

    return {
        "stream_bits": stream_bits,
        "metadata_bits": metadata_bits,
        "scale_bits": scale_bits,
        "bias_fp32_bits": bias_fp32_bits,
        "quant_weight_count": quant_weight_count,
        "bias_count": bias_count,
    }


def quantize_biases_int16_inplace(model):
    qmax = (2 ** (BIAS_BITS_AGGRESSIVE - 1)) - 1

    total_bias_bits = 0
    total_bias_scale_bits = 0
    bias_count = 0

    for module in model.modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        if module.bias is None:
            continue

        b = module.bias.data
        max_abs = b.detach().abs().max()

        if max_abs.item() == 0:
            scale = torch.tensor(
                1.0,
                device=b.device,
                dtype=b.dtype
            )
        else:
            scale = max_abs / qmax

        q = torch.round(b / scale)
        q = torch.clamp(q, -qmax, qmax)

        module.bias.data.copy_(q * scale)

        bias_count += b.numel()
        total_bias_bits += b.numel() * BIAS_BITS_AGGRESSIVE
        total_bias_scale_bits += WEIGHT_SCALE_BITS

    return {
        "bias_bits": total_bias_bits,
        "bias_scale_bits": total_bias_scale_bits,
        "bias_count": bias_count,
    }


def main():
    teacher_ref = load_teacher_reference()
    _, test_loader = get_dataloaders()

    # Best KD-0.25
    kd_model, kd_ckpt = load_kd025()
    kd_acc = evaluate(kd_model, test_loader)
    kd_params = count_params(kd_model)
    kd_bits = kd_params * 32
    kd_mb = bits_to_mb(kd_bits)
    kd_cr = teacher_ref["bits"] / kd_bits

    print("=" * 100)
    print("BEST WIDTH-0.25 KD RESULT")
    print("=" * 100)
    print(f"Teacher accuracy:          {teacher_ref['accuracy']:.2f}%")
    print(f"Teacher size:              {teacher_ref['mb']:.4f} MB")
    print(f"KD-0.25 accuracy:          {kd_acc:.2f}%")
    print(f"KD-0.25 parameters:        {kd_params:,}")
    print(f"KD-0.25 FP32 size:         {kd_mb:.4f} MB")
    print(f"Architecture-only CR:      {kd_cr:.4f}x")

    # W3 QAT
    qat_model, qat_ckpt = load_w3_qat()
    qat_acc = evaluate(qat_model, test_loader)

    print()
    print("=" * 100)
    print("W3A8 QAT BEFORE BN FOLDING")
    print("=" * 100)
    print(
        f"Stored best checkpoint acc: "
        f"{float(qat_ckpt.get('test_accuracy', qat_acc)):.2f}%"
    )
    print(f"Re-evaluated accuracy:      {qat_acc:.2f}%")

    # BN folding
    folded_model = copy.deepcopy(qat_model)
    folded_count = fold_bn_recursive(folded_model)
    folded_model = folded_model.to(DEVICE)
    folded_model.eval()
    folded_acc = evaluate(folded_model, test_loader)

    print()
    print("=" * 100)
    print("AFTER BN FOLDING")
    print("=" * 100)
    print(f"BN layers folded:          {folded_count}")
    print(f"Accuracy after folding:    {folded_acc:.2f}%")
    print(f"Accuracy change:           {folded_acc - qat_acc:+.2f} pp")

    # Huffman with FP32 biases
    hstats = huffman_storage_for_model(folded_model)

    conservative_bits = (
        hstats["stream_bits"]
        + hstats["metadata_bits"]
        + hstats["scale_bits"]
        + hstats["bias_fp32_bits"]
    )

    conservative_mb = bits_to_mb(conservative_bits)
    conservative_cr = teacher_ref["bits"] / conservative_bits

    avg_huff = (
        hstats["stream_bits"]
        / hstats["quant_weight_count"]
    )

    print()
    print("=" * 100)
    print("FOLDED W3 + HUFFMAN, FP32 BIASES")
    print("=" * 100)
    print(f"Avg Huffman bits/weight:   {avg_huff:.4f}")
    print(f"Huffman weight stream:     {bits_to_mb(hstats['stream_bits']):.6f} MB")
    print(f"Huffman metadata:          {bits_to_mb(hstats['metadata_bits']):.6f} MB")
    print(f"FP16 weight scales:        {bits_to_mb(hstats['scale_bits']):.6f} MB")
    print(f"FP32 folded biases:        {bits_to_mb(hstats['bias_fp32_bits']):.6f} MB")
    print(f"Total model size:          {conservative_mb:.6f} MB")
    print(f"Overall CR vs teacher:     {conservative_cr:.2f}x")
    print(f"Accuracy:                  {folded_acc:.2f}%")

    # Optional INT16 bias quantization
    if TRY_INT16_BIASES:
        aggressive_model = copy.deepcopy(folded_model)

        bstats = quantize_biases_int16_inplace(
            aggressive_model
        )

        aggressive_acc = evaluate(
            aggressive_model,
            test_loader
        )

        aggressive_bits = (
            hstats["stream_bits"]
            + hstats["metadata_bits"]
            + hstats["scale_bits"]
            + bstats["bias_bits"]
            + bstats["bias_scale_bits"]
        )

        aggressive_mb = bits_to_mb(aggressive_bits)
        aggressive_cr = teacher_ref["bits"] / aggressive_bits

        print()
        print("=" * 100)
        print("FOLDED W3 + HUFFMAN + INT16 BIASES")
        print("=" * 100)
        print(f"INT16 biases:              {bits_to_mb(bstats['bias_bits']):.6f} MB")
        print(f"FP16 bias scales:          {bits_to_mb(bstats['bias_scale_bits']):.6f} MB")
        print(f"Final model size:          {aggressive_mb:.6f} MB")
        print(f"FINAL OVERALL CR:          {aggressive_cr:.2f}x")
        print(f"Accuracy:                  {aggressive_acc:.2f}%")
        print(f"Accuracy change vs W3 QAT: {aggressive_acc - qat_acc:+.2f} pp")

    print()
    print("=" * 100)
    print(
        f"100x target size = "
        f"{teacher_ref['mb']/100:.6f} MB"
    )
    print("=" * 100)


if __name__ == "__main__":
    main()

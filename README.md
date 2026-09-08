# CS6886 Assignment 2 — MobileNetV2 Compression

Implementation and evaluation of weight and activation compression techniques for MobileNetV2 on CIFAR-10.

## Results

| Configuration | Test Accuracy | Model Size | Compression |
|---|---:|---:|---:|
| FP32 MobileNetV2 | 95.11% | 8.5323 MB | 1.00x |
| Mixed W5/W6/W7 + A8 | 94.04% | 1.7073 MB | ~5.00x |
| KD-0.35 + W6A8 + Huffman | 90.36% | 0.3381 MB | 25.2384x |
| KD-0.25 + W3A8 QAT + BN Folding + Huffman | 88.08% | 0.083518 MB | 102.16x |

The **KD-0.35 + W6A8 + Huffman** configuration is selected as the final operating point, providing 25.2384x model compression at 90.36% CIFAR-10 test accuracy.

### Notation

- **W6A8**: 6-bit weights and 8-bit activations.
- **KD**: Knowledge Distillation.
- **KD-0.35**: distilled MobileNetV2 student with width multiplier 0.35.
- **KD-0.25**: distilled MobileNetV2 student with width multiplier 0.25.
- **QAT**: Quantization-Aware Training.
- **BN**: Batch Normalization.
- **FP32**: 32-bit floating-point representation.

## Repository Structure

```text
.
├── baseline/
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── model.py
│   └── train.py
│
├── compression/
│   ├── activation_quantization.py
│   ├── calibrated_activation_clipping.py
│   ├── clustered_weight_compression.py
│   ├── compression_stats.py
│   ├── huffman_weight_stats.py
│   ├── layer_sensitivity.py
│   ├── mixed_precision_search.py
│   ├── pruned_compression_stats.py
│   ├── pruning.py
│   ├── quantization.py
│   └── sensitivity_aware_clustering.py
│
├── distillation/
│   ├── distillation_utils.py
│   ├── evaluate_distilled.py
│   ├── kd_config.py
│   ├── qat_layers.py
│   ├── student_model.py
│   └── train_distillation.py
│
├── experiments/
│   ├── analyze_w025_w3_bn_huffman.py
│   ├── compress_distilled_student.py
│   ├── huffman_distilled_w035.py
│   ├── run_sweep.py
│   └── train_w025_kd_qat.py
│
├── tests/
│   ├── test_baseline.py
│   ├── test_mixed_precision_pruning.py
│   ├── test_prune_quant.py
│   ├── test_weight_activation_quant.py
│   └── test_weight_quant.py
│
├── analysis/
│   ├── analyse_failures.py
│   ├── analyze_layer_sensitivity.py
│   └── make_report_figures.py
│
├── results/
├── figures/
├── requirements.txt
├── .gitignore
└── README.md
```

The codebase separates baseline training/evaluation, reusable compression methods, knowledge distillation/QAT, complete experiments, tests, and analysis.

## Environment

Experiments were run using:

- Python 3.12.2
- PyTorch 2.14.0
- torchvision 0.29.0
- NumPy 2.5.2
- pandas 3.0.5
- Matplotlib 3.11.1
- Weights & Biases 0.29.0

Create the environment from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducibility

A fixed random seed of **42** is used for the experiments:

```python
SEED = 42
```

The seed is configured in `baseline/config.py`. Python, NumPy, and PyTorch random number generators are initialized using this seed during training.

All commands below should be executed from the repository root.

## Running the Code

### 1. Train the FP32 baseline

```bash
python -m baseline.train
```

This trains the MobileNetV2 CIFAR-10 baseline and saves the best checkpoint according to `MODEL_PATH` in `baseline/config.py`.

### 2. Evaluate the FP32 baseline

```bash
python -m tests.test_baseline
```

Expected test accuracy:

```text
95.11%
```

### 3. Weight quantization

```bash
python -m tests.test_weight_quant
```

### 4. Weight and activation quantization

```bash
python -m tests.test_weight_activation_quant
```

### 5. Pruning + weight + activation quantization

```bash
python -m tests.test_prune_quant
```

### 6. Mixed-precision pruning test

```bash
python -m tests.test_mixed_precision_pruning
```

### 7. Layer-wise sensitivity analysis

```bash
python -m compression.layer_sensitivity
```

This evaluates the sensitivity of individual convolutional and linear layers to reduced weight precision.

### 8. Sensitivity-aware mixed-precision search

```bash
python -m compression.mixed_precision_search
```

This searches for a layer-wise W5/W6/W7 configuration while using A8 activations.

The resulting high-accuracy configuration achieves approximately:

```text
Accuracy:       94.04%
Model size:     1.7073 MB
Compression:    ~5.00x
```

### 9. Run the compression sweep

```bash
python -m experiments.run_sweep
```

The controlled sweep varies:

- weight bit-width: W5, W6, W7, W8
- activation bit-width: A7, A8
- global magnitude-pruning ratio: 0%, 10%, 20%, 30%

This gives a total of 32 configurations.

The script evaluates test accuracy, model compression ratio, activation compression ratio, and sparsity. The configurations are also logged to Weights & Biases for generation of the Parallel Coordinates plot.

### 10. Train knowledge-distilled students

```bash
python -m distillation.train_distillation
```

Reduced-width MobileNetV2 student models are trained using the FP32 baseline as the teacher.

### 11. Evaluate KD-0.35 + W6A8 + Huffman

After generating the required KD-0.35 checkpoint, run:

```bash
python -m experiments.huffman_distilled_w035
```

Expected result:

```text
Original teacher accuracy:      95.11%
Student FP32 accuracy:          90.62%
W6A8 accuracy:                  90.36%
Dense W6 model size:            0.3631 MB
Dense W6 compression:           23.5010x
Average Huffman bits/weight:    5.3528
Huffman model size:             0.3381 MB
Final compression:              25.2384x
```

The final size includes the Huffman-coded weight stream, Huffman metadata, quantization scales, and remaining FP32 parameters.

This is the selected final compression operating point.

### 12. Train KD-0.25 + W3A8 QAT

```bash
python -m experiments.train_w025_kd_qat
```

This first trains a width-0.25 knowledge-distilled student and then performs W3A8 quantization-aware training.

### 13. Analyze BN folding and extreme compression

```bash
python -m experiments.analyze_w025_w3_bn_huffman
```

Expected result:

```text
KD-0.25 accuracy:               89.55%
W3A8 QAT accuracy:              88.15%
Accuracy after BN folding:      88.12%
Final accuracy:                 88.08%
Final model size:               0.083518 MB
Final compression:              102.16x
```

The final representation uses:

- W3 quantized weights
- A8 activations
- BatchNorm folding
- Huffman-coded weight symbols
- FP16 per-channel weight scales
- INT16 folded biases
- FP16 bias scales

### 14. Failure analysis

```bash
python -m analysis.analyse_failures
```

This evaluates class-wise accuracy and common CIFAR-10 misclassifications.

## Quick Run Reference

| Task | Command |
|---|---|
| Train baseline | `python -m baseline.train` |
| Evaluate baseline | `python -m tests.test_baseline` |
| Weight quantization | `python -m tests.test_weight_quant` |
| Weight + activation quantization | `python -m tests.test_weight_activation_quant` |
| Pruning + quantization | `python -m tests.test_prune_quant` |
| Layer sensitivity | `python -m compression.layer_sensitivity` |
| Mixed-precision search | `python -m compression.mixed_precision_search` |
| 32-run compression sweep | `python -m experiments.run_sweep` |
| Train KD students | `python -m distillation.train_distillation` |
| KD-0.35 W6A8 + Huffman | `python -m experiments.huffman_distilled_w035` |
| Train KD-0.25 W3A8 QAT | `python -m experiments.train_w025_kd_qat` |
| Analyze 102.16x model | `python -m experiments.analyze_w025_w3_bn_huffman` |
| Failure analysis | `python -m analysis.analyse_failures` |

## Activation Compression

Activations are quantized per tensor. A8 activations use approximately one quarter of the storage required by FP32 activations, giving approximately **4x activation compression**.

Activation storage is measured from the outputs of `Conv2d` and `Linear` layers during inference, including the quantization-scale overhead.

## Weights & Biases

The 32-run compression sweep logs the following parameters and metrics to W&B:

- weight bit-width
- activation bit-width
- pruning ratio
- model compression ratio
- test accuracy

These runs are used to generate the Parallel Coordinates chart included in the report.

## Checkpoints

PyTorch checkpoint (`.pth`) files are excluded from GitHub because they are generated training artifacts and can be large.

Evaluation scripts requiring checkpoints should therefore be executed after running the corresponding training scripts.

## Dataset

CIFAR-10 is used for all experiments. The downloaded dataset is excluded from Git and is prepared through the data-loading pipeline.

## Main Selected Configuration

The final configuration selected for the assignment is:

**KD-0.35 + W6A8 + Huffman**

with:

- Test accuracy: **90.36%**
- Model size: **0.3381 MB**
- Model compression: **25.2384x**
- Activation compression: approximately **4x**
- Average Huffman weight representation: **5.3528 bits/weight**
# CS6886 Assignment 2 — MobileNetV2 Compression on CIFAR-10

**Course:** CS6886 — Systems Engineering for Deep Learning  
**Author:** Nishanth Senthil Kumar  
**Roll No.:** EE23B049  
**Institute:** Indian Institute of Technology Madras

---

## Overview

This repository contains the implementation for Assignment 2 of CS6886.

The project trains and compresses a MobileNetV2 model on CIFAR-10. The implemented compression techniques include:

- Weight quantization
- Activation quantization
- Mixed-precision quantization
- Magnitude pruning
- Layer-sensitivity analysis
- Knowledge distillation (KD)
- Quantization-aware training (QAT)
- Batch-normalization (BN) folding
- Huffman coding

The FP32 MobileNetV2 baseline achieves **95.11% test accuracy**.

Three representative compression operating points obtained in the experiments are:

| Method | Accuracy | Compression Ratio |
|---|---:|---:|
| Sensitivity-aware mixed precision | 94.04% | 5.00x |
| KD-0.35 + W6A8 + Huffman | 90.36% | 25.2384x |
| KD-0.25 + W3A8 QAT + BN folding + Huffman | 88.08% | 102.16x |

The **KD-0.35 + W6A8 + Huffman** configuration is selected as the practical operating point because it provides a strong trade-off between compression and classification accuracy.

---

## Repository Structure

```text
CS6886_Assignment2/
├── README.md
├── requirements.txt
├── .gitignore
│
├── baseline/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── model.py
│   └── train.py
│
├── compression/
│   ├── __init__.py
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
│   ├── __init__.py
│   ├── distillation_utils.py
│   ├── evaluate_distilled.py
│   ├── kd_config.py
│   ├── qat_layers.py
│   ├── student_model.py
│   └── train_distillation.py
│
├── experiments/
│   ├── __init__.py
│   ├── analyze_w025_w3_bn_huffman.py
│   ├── compress_distilled_student.py
│   ├── huffman_distilled_w035.py
│   ├── run_sweep.py
│   └── train_w025_kd_qat.py
│
├── analysis/
│   ├── __init__.py
│   ├── analyse_failures.py
│   ├── analyze_layer_sensitivity.py
│   ├── make_report_figures.py
│   └── plotting.py
│
├── tests/
│   ├── __init__.py
│   ├── test_baseline.py
│   ├── test_mixed_precision_pruning.py
│   ├── test_prune_quant.py
│   ├── test_weight_activation_quant.py
│   └── test_weight_quant.py
│
├── checkpoints/
│   ├── best_mobilenetv2_cifar10.pth
│   ├── student_w0p35_kd_best.pth
│   ├── student_w025_kd_best.pth
│   └── student_w025_W3A8_qat_best.pth
│
├── results/
│   └── CSV files containing experimental results
│
└── figures/
    └── Figures used in the report
```

---

## Environment

The experiments were run using:

```text
Python       3.12.2
PyTorch      2.14.0
torchvision  0.29.0
NumPy        2.5.2
pandas       3.0.5
matplotlib   3.11.1
wandb        0.29.0
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Reproducibility

The random seed used for the experiments is:

```python
SEED = 42
```

The seed is configured in:

```text
baseline/config.py
```

Seeds are set for Python, NumPy, and PyTorch.

The implementation automatically selects the configured PyTorch device. The experiments were run using the available accelerator through the device configuration in `baseline/config.py`.

---

## Dataset

CIFAR-10 is used for all experiments.

The raw dataset is **not stored in this repository**. It is downloaded automatically using `torchvision` when a training or evaluation command requiring CIFAR-10 is executed for the first time.

No manual dataset download is therefore required.

### Training preprocessing

The training pipeline uses:

- Resize to 96x96
- Random crop with padding
- Random horizontal flip
- Conversion to tensor
- ImageNet normalization

### Test preprocessing

The test pipeline is deterministic and uses:

- Resize to 96x96
- Conversion to tensor
- ImageNet normalization

The images are resized because the model uses an ImageNet-pretrained MobileNetV2 backbone. ImageNet normalization is used to keep the input distribution consistent with the pretrained weights.

---

## Included Checkpoints

The trained checkpoints required to reproduce the main reported results are included directly in:

```text
checkpoints/
```

The provided checkpoints are:

```text
checkpoints/
├── best_mobilenetv2_cifar10.pth
├── student_w0p35_kd_best.pth
├── student_w025_kd_best.pth
└── student_w025_W3A8_qat_best.pth
```

They are used as follows:

### `best_mobilenetv2_cifar10.pth`

FP32 MobileNetV2 teacher/baseline.

```text
Test accuracy: 95.11%
```

### `student_w0p35_kd_best.pth`

Width-0.35 knowledge-distilled student used for the 25.2384x compression pipeline.

### `student_w025_kd_best.pth`

Width-0.25 knowledge-distilled student used as part of the extreme-compression pipeline.

### `student_w025_W3A8_qat_best.pth`

Width-0.25 W3A8 quantization-aware-trained checkpoint used for the 102.16x compression result.

The main reported models can therefore be evaluated **without retraining**.

Training scripts are also included if regeneration of the checkpoints is required.

---

# Running the Code

All commands below should be executed from the repository root.

## Fresh Clone Setup

Clone the repository:

```bash
git clone https://github.com/lazarsalt1441/CS6886_Assignment2.git
cd CS6886_Assignment2
```

Create a Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

CIFAR-10 will be downloaded automatically the first time it is required.

The provided model checkpoints are already contained in `checkpoints/`, so retraining is not required for the main evaluation commands.

---

## 1. Evaluate the FP32 Baseline

Run:

```bash
python -m tests.test_baseline
```

Expected main result:

```text
Test accuracy: 95.11%
```

The baseline checkpoint corresponds to the best checkpoint selected by test accuracy.

---

## 2. Train the FP32 Baseline

To regenerate the baseline model:

```bash
python -m baseline.train
```

The baseline uses:

- MobileNetV2
- ImageNet pretrained initialization
- SGD
- Momentum
- Weight decay
- Cosine-annealing learning-rate schedule
- Cross-entropy loss

---

## 3. Weight Quantization

To evaluate weight-only quantization:

```bash
python -m tests.test_weight_quant
```

Weights are quantized using symmetric per-output-channel quantization.

---

## 4. Weight and Activation Quantization

Run:

```bash
python -m tests.test_weight_activation_quant
```

Weights use per-output-channel symmetric quantization, while activations use symmetric per-tensor quantization.

---

## 5. Pruning + Quantization

Run:

```bash
python -m tests.test_prune_quant
```

One tested configuration uses:

```text
Pruning:        30%
Weight bits:    6
Activation bits: 8
```

Representative output:

```text
FP32 baseline accuracy: 95.11%
After pruning:          92.24%
After W6 quantization:  91.80%
After W6A8:             91.26%
Final sparsity:         30.00%
```

---

## 6. Mixed-Precision Pruning

Run:

```bash
python -m tests.test_mixed_precision_pruning
```

This evaluates combinations of pruning and mixed-precision quantization.

---

## 7. Compression Sweep

Run:

```bash
python -m experiments.run_sweep
```

The sweep evaluates multiple combinations of:

- Weight bit-width
- Activation bit-width
- Pruning ratio

The resulting experimental data is stored as CSV output and was used to generate the Weights & Biases parallel-coordinates visualization included in the report.

---

## 8. Layer-Sensitivity Analysis

Run:

```bash
python -m analysis.analyze_layer_sensitivity
```

The sensitivity analysis measures the accuracy degradation caused by quantizing individual layers.

These measurements are used to guide the sensitivity-aware mixed-precision assignment.

Results are stored under:

```text
results/
```

---

## 9. Knowledge Distillation

Reduced-width MobileNetV2 students are trained using knowledge distillation from the full-width FP32 teacher.

To train the standard distilled students:

```bash
python -m distillation.train_distillation
```

The student architecture is defined in:

```text
distillation/student_model.py
```

The student learns using both:

- CIFAR-10 hard labels
- Teacher soft targets

---

# Main Compressed Models

## 10. KD-0.35 + W6A8 + Huffman

This is the selected practical operating point.

The required width-0.35 student checkpoint is already provided in `checkpoints/`.

Run:

```bash
python -m experiments.huffman_distilled_w035
```

Expected main results:

```text
Original teacher FP32 accuracy:     95.11%
Original teacher FP32 size:         8.5323 MB

Student FP32 accuracy:              90.62%
Student FP32 size:                  1.5600 MB

W6A8 accuracy:                      90.36%
Accuracy drop vs student:           0.26 pp
Accuracy drop vs teacher:           4.75 pp

Dense W6 student size:              0.3631 MB
Dense W6 overall CR:                23.5010x

Average Huffman bits/weight:        5.3528
Huffman weight stream:              0.2520 MB
Huffman metadata overhead:          0.005469 MB
Scale overhead:                     0.0269 MB
FP32 exceptions:                    0.0537 MB

Huffman student size:               0.3381 MB
Overall CR vs original:             25.2384x
```

Thus, the original 8.5323 MB model is reduced to approximately 0.3381 MB while retaining 90.36% test accuracy.

---

## 11. Train Width-0.25 KD + QAT Models

To regenerate the width-0.25 knowledge-distilled and QAT checkpoints:

```bash
python -m experiments.train_w025_kd_qat
```

This performs:

```text
Teacher
   ↓
Width-0.25 student
   ↓
Knowledge distillation
   ↓
Quantization-aware training
```

The provided checkpoints mean that this training step is **not required** for evaluation.

---

## 12. Extreme Compression: W3A8 + BN Folding + Huffman

The required KD and QAT checkpoints are already included in `checkpoints/`.

Run:

```bash
python -m experiments.analyze_w025_w3_bn_huffman
```

Expected main results:

```text
W3A8 QAT accuracy:             88.15%

BN layers folded:             52
Accuracy after folding:       88.12%
Accuracy change:              -0.03 pp

Average Huffman bits/weight:   2.1846

Final model size:              0.083518 MB
Final compression ratio:       102.16x
Final accuracy:                88.08%
```

The full pipeline is:

```text
Width reduction
    ↓
Knowledge distillation
    ↓
W3A8 quantization-aware training
    ↓
Batch-normalization folding
    ↓
Huffman coding
    ↓
INT16 bias quantization
```

This produces the most aggressive compression result in the project.

---

# Compression Implementation

## Weight Quantization

Weights are compressed using symmetric per-output-channel quantization.

For a bit-width \(b\),

```text
qmax = 2^(b-1) - 1
```

For each output channel, a scale is determined from the maximum absolute weight value.

The quantized integer is approximately:

```text
q = round(w / scale)
```

followed by clipping to the representable integer range.

A separate scale is stored for each output channel.

---

## Activation Quantization

Activations use symmetric per-tensor quantization.

The primary configurations use:

```text
A8 = 8-bit activations
```

Relative to FP32 storage, the ideal activation compression is:

```text
32 / 8 = 4x
```

The reported activation compression includes the small scale-storage overhead.

---

## Mixed-Precision Quantization

Layer-sensitivity analysis is used to determine which layers tolerate aggressive quantization.

Sensitive layers are assigned higher weight precision, while robust layers can use lower bit-widths.

The sensitivity-aware mixed-precision configuration obtains approximately:

```text
Accuracy:          94.04%
Compression ratio: 5.00x
Activation CR:     ~4.00x
```

---

## Knowledge Distillation

Knowledge distillation is used to reduce the MobileNetV2 width while recovering part of the accuracy lost through architectural compression.

The smaller student learns from:

1. Ground-truth CIFAR-10 labels.
2. Soft probability targets generated by the FP32 teacher.

Two reduced-width students are used in the final experiments:

```text
KD-0.35 → width multiplier 0.35
KD-0.25 → width multiplier 0.25
```

---

## Quantization-Aware Training

For aggressive low-bit quantization, particularly W3A8, quantization-aware training is used.

Fake quantization is introduced during training so that the student network can adapt its parameters to the quantization error.

This substantially improves the robustness of the width-0.25 model under 3-bit weight quantization.

---

## Batch-Normalization Folding

Batch-normalization layers are folded into the preceding convolution layers before final storage.

For a convolution followed by batch normalization, the convolution weights and biases are transformed so that the BN operation no longer needs to be represented as a separate layer.

For the extreme-compression model:

```text
BN layers folded:       52
Before folding:         88.15%
After folding:          88.12%
Accuracy change:        -0.03 pp
```

Thus, BN folding removes the need to store the separate BN parameters while producing negligible accuracy degradation.

---

## Huffman Coding

Huffman coding is applied to the quantized weight symbols.

Because quantized values do not occur with equal frequency, Huffman coding assigns shorter codes to common symbols and longer codes to less common symbols.

For the KD-0.35 W6 model:

```text
Nominal precision:          6 bits/weight
Average Huffman precision:  5.3528 bits/weight
```

For the extreme W3 model:

```text
Nominal precision:          3 bits/weight
Average Huffman precision:  2.1846 bits/weight
```

The Huffman metadata required for decoding is explicitly included in the final storage calculations.

---

# Storage Overheads

The reported model sizes include the additional storage required to reconstruct the compressed models.

For the selected **KD-0.35 + W6A8 + Huffman** model:

```text
Huffman weight stream:       0.2520 MB
Huffman metadata:            0.005469 MB
Quantization scales:         0.0269 MB
FP32 exceptions:             0.0537 MB
---------------------------------------
Total compressed size:       0.3381 MB
```

Therefore:

```text
Original model size:         8.5323 MB
Compressed model size:       0.3381 MB
Compression ratio:           25.2384x
```

For the extreme model, BN folding and INT16 bias quantization are additionally applied.

The final representation occupies:

```text
Final model size:            0.083518 MB
Compression ratio:           102.16x
Accuracy:                    88.08%
```

---

# Main Results

| Configuration | Accuracy | Model Size | Compression Ratio |
|---|---:|---:|---:|
| FP32 MobileNetV2 | 95.11% | 8.5323 MB | 1.00x |
| Sensitivity-aware mixed precision | 94.04% | ~1.707 MB | 5.00x |
| KD-0.35 + W6A8 + Huffman | 90.36% | 0.3381 MB | 25.2384x |
| KD-0.25 + W3A8 QAT + BN folding + Huffman | 88.08% | 0.083518 MB | 102.16x |

The **KD-0.35 + W6A8 + Huffman** model is selected as the final practical operating point.

It provides:

```text
Test accuracy:               90.36%
Model size:                  0.3381 MB
Model compression ratio:     25.2384x
Activation compression:      ~4.00x
```

relative to the original FP32 teacher.

---

# Figures

Figures used in the report are stored under:

```text
figures/
```

These include:

- Baseline training/test loss curves
- Baseline training/test accuracy curves
- Misclassified examples
- Layer-wise sensitivity
- Mixed-precision weight allocation
- Pruning versus accuracy
- Pruning versus compression
- Distilled-student compression results
- Huffman layer statistics
- Accuracy/compression Pareto plot
- W&B parallel-coordinates plot

---

# Results

Numerical results are stored under:

```text
results/
```

The CSV files contain results from:

- Quantization sweeps
- Pruning experiments
- Layer-sensitivity analysis
- Mixed-precision search
- Distilled-student compression
- Huffman compression
- Activation clipping
- Clustering experiments

---

# Notes

- CIFAR-10 is automatically downloaded by `torchvision` on first use.
- Raw CIFAR-10 data is not committed to the repository.
- Four trained checkpoints required for direct evaluation of the main reported models are included under `checkpoints/`.
- Retraining is not required to reproduce the main evaluation results.
- Training scripts are included to regenerate the models if required.
- Python virtual environments are excluded from Git.
- Python cache files are excluded from Git.
- Local W&B run data is excluded from Git.
- Generated training-output directories are excluded where appropriate.
- All final compressed model sizes include the stated metadata and quantization overheads.

---

# GitHub Repository

https://github.com/lazarsalt1441/CS6886_Assignment2
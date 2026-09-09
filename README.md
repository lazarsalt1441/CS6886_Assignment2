# CS6886 Assignment 2 — MobileNetV2 Compression on CIFAR-10

**Course:** CS6886 — Systems Engineering for Deep Learning  
**Author:** Nishanth Senthil Kumar  
**Roll No.:** EE23B049  
**Institute:** Indian Institute of Technology Madras

## Overview

This repository contains the implementation for Assignment 2 of CS6886.

The project trains and compresses a MobileNetV2 model on CIFAR-10. The implemented compression pipeline includes:

- Weight quantization
- Activation quantization
- Mixed-precision quantization
- Magnitude pruning
- Layer-sensitivity analysis
- Knowledge distillation (KD)
- Quantization-aware training (QAT)
- Batch-normalization (BN) folding
- Huffman coding

The FP32 teacher model achieves **95.11% test accuracy**.

Three representative compression operating points obtained in the experiments are:

| Method | Accuracy | Compression Ratio |
|---|---:|---:|
| Sensitivity-aware mixed precision | 94.04% | 5.00x |
| KD-0.35 + W6A8 + Huffman | 90.36% | 25.2384x |
| KD-0.25 + W3A8 QAT + BN folding + Huffman | 88.08% | 102.16x |

The **KD-0.35 + W6A8 + Huffman** configuration is used as the final practical operating point because it provides a strong trade-off between model size and classification accuracy.

---

## Repository Structure

```text
CS6886_Assignment2/
├── README.md
├── requirements.txt
├── .gitignore
│
├── baseline/
│   ├── config.py
│   ├── data.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
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
├── analysis/
│   ├── analyse_failures.py
│   ├── analyze_layer_sensitivity.py
│   ├── make_report_figures.py
│   └── plotting.py
│
├── tests/
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

Install the required packages using:

```bash
pip install -r requirements.txt
```

---

## Reproducibility

The random seed used for the experiments is:

```python
SEED = 42
```

The seed configuration is defined in:

```text
baseline/config.py
```

Random seeds are set for Python, NumPy, and PyTorch.

The code automatically uses an available accelerator when configured. The reported experiments were executed using the PyTorch device configuration in `baseline/config.py`.

---

## Dataset

CIFAR-10 is used for all experiments.

The raw CIFAR-10 dataset is **not stored in this repository**. It is downloaded/prepared through the data-loading pipeline.

The baseline preprocessing includes resizing CIFAR-10 images to 96x96 to provide a larger spatial resolution for the ImageNet-pretrained MobileNetV2.

Training-time augmentation includes:

- Resize to 96x96
- Random crop with padding
- Random horizontal flip
- Conversion to tensor
- ImageNet normalization

The test pipeline is deterministic and uses resizing, tensor conversion, and the same normalization.

---

## Included Checkpoints

The trained checkpoints required to directly reproduce the main reported evaluation results are included in:

```text
checkpoints/
```

The included models are:

```text
checkpoints/
├── best_mobilenetv2_cifar10.pth
├── student_w0p35_kd_best.pth
├── student_w025_kd_best.pth
└── student_w025_W3A8_qat_best.pth
```

They correspond to:

- `best_mobilenetv2_cifar10.pth`  
  FP32 MobileNetV2 teacher/baseline with **95.11%** test accuracy.

- `student_w0p35_kd_best.pth`  
  Width-0.35 knowledge-distilled student used for the **25.2384x** compression result.

- `student_w025_kd_best.pth`  
  Width-0.25 knowledge-distilled FP32 student used in the extreme-compression pipeline.

- `student_w025_W3A8_qat_best.pth`  
  Width-0.25 W3A8 quantization-aware-trained model used for the **102.16x** result.

Therefore, retraining is **not required** to evaluate the main reported models.

Training scripts are also included if regeneration of the checkpoints is desired.

---

## Running the Code

All commands below should be executed from the repository root.

### 1. Evaluate the FP32 Baseline

```bash
python -m tests.test_baseline
```

Expected result:

```text
Test accuracy: 95.11%
```

---

### 2. Train the FP32 Baseline

To train the baseline MobileNetV2:

```bash
python -m baseline.train
```

The baseline training uses SGD with momentum, weight decay, and cosine-annealing learning-rate scheduling.

---

### 3. Weight Quantization

To evaluate weight-only quantization:

```bash
python -m tests.test_weight_quant
```

---

### 4. Weight and Activation Quantization

To evaluate weight and activation quantization:

```bash
python -m tests.test_weight_activation_quant
```

---

### 5. Pruning + Quantization

To evaluate pruning followed by quantization:

```bash
python -m tests.test_prune_quant
```

For example, the tested 30% pruning + W6A8 configuration produces approximately:

```text
FP32 baseline accuracy: 95.11%
Pruned accuracy:        92.24%
W6 accuracy:            91.80%
W6A8 accuracy:          91.26%
Final sparsity:         30.00%
```

---

### 6. Mixed-Precision Pruning

```bash
python -m tests.test_mixed_precision_pruning
```

---

### 7. Compression Sweep

The compression sweep used to evaluate different combinations of weight precision, activation precision, and pruning can be run using:

```bash
python -m experiments.run_sweep
```

The resulting data can be used for the Weights & Biases parallel-coordinates visualization.

The report includes the W&B parallel-coordinates plot generated from the experimental sweep.

---

### 8. Layer-Sensitivity Analysis

```bash
python -m analysis.analyze_layer_sensitivity
```

Layer-sensitivity results are stored under:

```text
results/
```

These measurements are used to determine which layers are more sensitive to low-bit weight quantization and therefore guide mixed-precision assignment.

---

### 9. Knowledge Distillation

The smaller MobileNetV2 students use knowledge distillation from the full-width FP32 teacher.

To train the standard distilled students:

```bash
python -m distillation.train_distillation
```

The student architecture is implemented in:

```text
distillation/student_model.py
```

---

### 10. Evaluate KD-0.35 + W6A8 + Huffman

The required width-0.35 distilled checkpoint is already included in `checkpoints/`.

Run:

```bash
python -m experiments.huffman_distilled_w035
```

The reported result is:

```text
Original teacher FP32 accuracy:     95.11%
Original teacher FP32 size:         8.5323 MB

Student FP32 accuracy:              90.62%
Student FP32 size:                  1.5600 MB

W6A8 accuracy:                      90.36%

Average Huffman bits/weight:        5.3528
Huffman weight stream:              0.2520 MB
Huffman metadata overhead:          0.005469 MB
Scale overhead:                     0.0269 MB
FP32 exceptions:                    0.0537 MB

Huffman student size:               0.3381 MB
Overall compression ratio:          25.2384x
```

This is the configuration selected as the final practical operating point.

---

### 11. Train the Width-0.25 KD + QAT Model

To regenerate the width-0.25 knowledge-distilled and quantization-aware-trained models:

```bash
python -m experiments.train_w025_kd_qat
```

This performs knowledge distillation followed by QAT for the low-bit student configurations.

Generated training checkpoints are stored separately from the provided reproducibility checkpoints.

---

### 12. Evaluate the Extreme-Compression Model

The required KD-0.25 and W3A8 QAT checkpoints are included in `checkpoints/`.

Run:

```bash
python -m experiments.analyze_w025_w3_bn_huffman
```

The final reported result is approximately:

```text
Teacher accuracy:              95.11%

W3A8 QAT accuracy:             88.15%
Accuracy after BN folding:     88.12%

Average Huffman bits/weight:   2.1846

Final model size:              0.083518 MB
Final compression ratio:       102.16x
Final accuracy:                88.08%
```

This pipeline combines:

```text
Width reduction
    -> Knowledge distillation
    -> W3A8 quantization-aware training
    -> Batch-normalization folding
    -> Huffman coding
    -> INT16 bias quantization
```

---

## Compression Method

### Weight Quantization

Weights are quantized using symmetric per-output-channel quantization.

For a bit-width `b`:

```text
qmax = 2^(b-1) - 1
```

A separate scale is stored for each output channel.

---

### Activation Quantization

Activations are quantized using symmetric per-tensor quantization.

For the main reported configurations:

```text
A8 = 8-bit activations
```

Compared with FP32 activation storage, this gives approximately:

```text
32 / 8 = 4x
```

activation compression, with a small scale-storage overhead.

---

### Mixed Precision

Layer-sensitivity analysis is used to identify layers that are more sensitive to aggressive quantization.

Sensitive layers receive higher weight precision, while more robust layers can use fewer bits.

The sensitivity-aware mixed-precision configuration achieves approximately:

```text
Accuracy:             94.04%
Weight/model CR:      5.00x
Activation CR:        4.00x
```

---

### Knowledge Distillation

Knowledge distillation is used to train reduced-width MobileNetV2 student networks from the full-width FP32 teacher.

The students learn from both:

- CIFAR-10 ground-truth labels
- Soft targets produced by the teacher network

This allows substantial architectural compression while retaining more accuracy than training the small model independently.

---

### Quantization-Aware Training

For the aggressive W3A8 configuration, quantization-aware training is used to expose the model to simulated low-precision quantization during training.

This allows the student to adapt its parameters to the quantization error before deployment.

---

### Batch-Normalization Folding

For the extreme-compression model, batch-normalization parameters are folded into the preceding convolution layers before final storage.

For a convolution followed by batch normalization, the convolution weights and biases are transformed so that the BN operation no longer needs to be stored or executed separately.

In the reported W3A8 model:

```text
BN layers folded: 52
```

The accuracy changes only from:

```text
88.15% -> 88.12%
```

after folding.

---

### Huffman Coding

Huffman coding is applied to the quantized weight symbols to exploit the non-uniform distribution of quantized values.

For the selected KD-0.35 + W6A8 model:

```text
Nominal weight precision:     6 bits
Average Huffman length:       5.3528 bits/weight
```

For the extreme W3 model:

```text
Nominal weight precision:     3 bits
Average Huffman length:       2.1846 bits/weight
```

All reported final model sizes include the relevant coding and quantization overheads rather than considering only the idealized weight bit-width.

---

## Storage Overheads

The reported compressed model sizes account for additional information required to reconstruct and execute the compressed model.

For the selected KD-0.35 + W6A8 + Huffman model:

```text
Huffman weight stream:       0.2520 MB
Huffman metadata:            0.005469 MB
Quantization scales:         0.0269 MB
FP32 exceptions:             0.0537 MB
---------------------------------------
Total compressed size:       0.3381 MB
```

Compared with the original FP32 teacher:

```text
Original model size:         8.5323 MB
Compressed model size:       0.3381 MB
Compression ratio:           25.2384x
```

For the extreme W3A8 model, the final representation additionally quantizes the folded biases to INT16 and stores FP16 bias scales, producing:

```text
Final model size:            0.083518 MB
Compression ratio:           102.16x
Accuracy:                    88.08%
```

---

## Main Results

| Configuration | Accuracy | Model Size | Compression Ratio |
|---|---:|---:|---:|
| FP32 MobileNetV2 | 95.11% | 8.5323 MB | 1.00x |
| Sensitivity-aware mixed precision | 94.04% | ~1.707 MB | 5.00x |
| KD-0.35 + W6A8 + Huffman | 90.36% | 0.3381 MB | 25.2384x |
| KD-0.25 + W3A8 QAT + BN folding + Huffman | 88.08% | 0.083518 MB | 102.16x |

The KD-0.35 + W6A8 + Huffman model is selected as the preferred practical configuration because it reduces the original model from **8.5323 MB to 0.3381 MB** while retaining **90.36% test accuracy**.

Its activation representation uses A8 quantization, corresponding to approximately **4x activation compression** relative to FP32.

---

## Figures and Results

Report figures are stored in:

```text
figures/
```

This includes:

- Baseline loss curve
- Baseline accuracy curve
- Misclassified examples
- Layer-sensitivity results
- Mixed-precision allocation
- Pruning results
- Distilled-student compression results
- Huffman statistics
- Accuracy/compression Pareto plot
- W&B parallel-coordinates plot

Numerical experimental results are stored as CSV files under:

```text
results/
```

---

## Notes

- Raw CIFAR-10 data is not committed to the repository.
- Python virtual environments and cache files are excluded.
- W&B local run data is excluded.
- Training-output directories are excluded where appropriate.
- The four checkpoints required for direct evaluation of the principal reported models are included under `checkpoints/`.
- All reported compression sizes include the stated storage overheads.

---

## GitHub Repository

https://github.com/lazarsalt1441/CS6886_Assignment2

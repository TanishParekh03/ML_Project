# Phase 1: Cross-Dataset Zero-Shot & Normalization Adaptation Benchmark

Empirical evaluation of German GTSRB-trained models evaluated across international road sign benchmarks.

## 1. Downstream Linear Probe Representation Performance

| Dataset | Paradigm | Normalization Regime | Linear Probe Top-1 Acc | Linear Probe Macro-F1 |
| :--- | :--- | :--- | :---: | :---: |
| **Belgium (BelgiumTSC)** | Baseline ResNet-18 | Static BN (Frozen German stats) | `89.15%` | `70.43%` |
| **Belgium (BelgiumTSC)** | Baseline ResNet-18 | BN-Adapt (Test-time shift) | `43.76%` | `34.60%` |
| **Belgium (BelgiumTSC)** | Concept Bottleneck Model | Static BN (Frozen German stats) | `88.91%` | `68.73%` |
| **Belgium (BelgiumTSC)** | Concept Bottleneck Model | BN-Adapt (Test-time shift) | `41.52%` | `32.95%` |
| **China (TSRD)** | Baseline ResNet-18 | Static BN (Frozen German stats) | `93.48%` | `89.64%` |
| **China (TSRD)** | Baseline ResNet-18 | BN-Adapt (Test-time shift) | `53.66%` | `61.14%` |
| **China (TSRD)** | Concept Bottleneck Model | Static BN (Frozen German stats) | `94.28%` | `94.73%` |
| **China (TSRD)** | Concept Bottleneck Model | BN-Adapt (Test-time shift) | `51.50%` | `58.81%` |

## 2. Concept Bottleneck Model: Zero-Shot Perceptual Generality

Evaluation of CBM's 4-axis concept heads (trained in Germany) predicting physical concepts across Belgium and China with **zero retraining**:

| Dataset | Normalization | Shape Acc | Colour Acc | Border Acc | Symbol Acc | Mean Concept Acc | Exact 4-Concept Match | Zero-Shot Class Acc |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Belgium (BelgiumTSC)** | Static BN | `85.1%` | `94.2%` | `91.0%` | `72.0%` | **`85.59%`** | `68.11%` | `37.45%` |
| **Belgium (BelgiumTSC)** | BN-Adapt (Shift) | `49.9%` | `60.4%` | `62.3%` | `28.6%` | **`50.30%`** | `19.61%` | `14.72%` |
| **China (TSRD)** | Static BN | `86.1%` | `60.3%` | `61.9%` | `55.8%` | **`66.00%`** | `37.61%` | `1.50%` |
| **China (TSRD)** | BN-Adapt (Shift) | `62.5%` | `44.3%` | `47.6%` | `32.1%` | **`46.64%`** | `19.16%` | `1.35%` |

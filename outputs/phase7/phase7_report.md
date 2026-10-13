# Phase 7 Report — Subject-Independent Evaluation & Generalization Benchmark

**Date**: 2026-10-07  
**Status**: COMPLETE (PASS)  
**Task**: Strict Subject-Independent Evaluation & LOSO Cross-Validation  

---

## 1. Executive Summary
This evaluation rigorously benchmarks the multimodal physiological strain engine across held-out subjects never seen during training:
1. **Held-Out Test Subjects** (`S15`, `S16`, `S17`): Evaluated against the frozen models trained in Phase 6.
2. **Leave-One-Subject-Out (LOSO) Cross-Validation**: Iterated across all 15 WESAD subjects to measure real-world person-to-person transferability.

Zero subject overlap was permitted between training partitions and evaluation partitions.

---

## 2. Held-Out Test Evaluation Results
Evaluated on **1850** windows across unseen subjects **S15, S16, S17**:

| Model | Accuracy | Bal. Accuracy | Precision | Recall | Macro F1 | ROC-AUC | PR-AUC | Brier |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A (NormWear 768-D)** | 0.8443 | 0.8556 | 0.7387 | 0.8997 | 0.8113 | 0.9400 | 0.8857 | 0.0990 |
| **Model B (Physio 20-D)** | 0.7195 | 0.6228 | 1.0000 | 0.2456 | 0.3944 | 0.8977 | 0.8519 | 0.2026 |
| **Model C (Fusion 788-D)** | **0.7319** | **0.6437** | **0.9364** | **0.2994** | **0.4537** | **0.8708** | **0.8118** | **0.1762** |

---

## 3. Held-Out Subject Breakdown (Model C)

| Subject ID | Total Windows | Baseline | Stress | Accuracy | Bal. Accuracy | F1 | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **S15** | 617 | 389 | 228 | 0.6580 | 0.5473 | 0.2097 | 0.6853 |
| **S16** | 612 | 390 | 222 | 0.8775 | 0.8340 | 0.8000 | 0.9351 |
| **S17** | 621 | 383 | 238 | 0.6618 | 0.5588 | 0.2105 | 1.0000 |

---

## 4. Full Leave-One-Subject-Out (LOSO) Cross-Validation Benchmark
Conducted across all 15 WESAD subjects:

- **Mean Macro-F1**: `0.7754 ± 0.2402`
- **Mean Balanced Accuracy**: `0.8412 ± 0.1296`
- **Mean ROC-AUC**: `0.9406 ± 0.0903`

---

## 5. Artifact Verification
- Model Comparison: [`outputs/phase7/model_comparison_table.csv`](model_comparison_table.csv)
- Test Breakdown: [`outputs/phase7/test_subject_breakdown.csv`](test_subject_breakdown.csv)
- LOSO Benchmark: [`outputs/phase7/loso_benchmark_results.csv`](loso_benchmark_results.csv)
- Confusion Matrix: [`outputs/phase7/plots/confusion_matrix_test.png`](plots/confusion_matrix_test.png)
- ROC Curves: [`outputs/phase7/plots/roc_curves_comparison.png`](plots/roc_curves_comparison.png)
- Calibration Curves: [`outputs/phase7/plots/calibration_curves.png`](plots/calibration_curves.png)
- LOSO F1 Chart: [`outputs/phase7/plots/loso_subject_f1.png`](plots/loso_subject_f1.png)

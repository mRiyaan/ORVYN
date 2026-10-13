# Phase 5 Report — Real NormWear Embedding Extraction & Dataset Creation

**Date**: 2026-10-07  
**Status**: COMPLETE (PASS)  
**Execution**: Executed natively on MPS hardware with frozen foundation model.

---

## 1. Foundation Model Specification
- **Model Architecture**: Official 12-layer ViT-Base Masked Autoencoder backbone (`NormWearModel`).
- **Foundation Checkpoint**: `checkpoints/normwear_pretrain_ckpt.pth`
- **Checkpoint SHA-256**: `36d0bca18356ccfc8e8916058bf838f26f1212a646f5780b487ad78581a92561`
- **Frozen Parameters**: `136,112,328` total parameters (`0` trainable parameters, `requires_grad = False` across all layers).
- **Execution Device**: `MPS` (`Apple Silicon Metal Performance Shaders`).
- **Throughput**: `7.8` windows/second (`1158.13s` total for 9078 windows).

---

## 2. Generated Representations
Generated from real WESAD wrist physiological windows `[N, 3, 384]` (order: `wrist_BVP`, `wrist_EDA`, `wrist_TEMP`):
1. **Primary Representation (Channel-Mean CLS)**:
   - File: [`data/processed/embeddings/normwear_embeddings.npy`](../../data/processed/embeddings/normwear_embeddings.npy)
   - Shape: `[9078, 768]` (`9078` windows $\times$ `768` dimensions)
   - Size: `26.6 MB`
2. **Channel-Wise CLS Representation**:
   - File: [`data/processed/embeddings/cls_channel_embeddings.npy`](../../data/processed/embeddings/cls_channel_embeddings.npy)
   - Shape: `[9078, 3, 768]` (`[9078, 3, 768]`)
3. **Channel-Concatenated Representation**:
   - File: [`data/processed/embeddings/cls_concat_embeddings.npy`](../../data/processed/embeddings/cls_concat_embeddings.npy)
   - Shape: `[9078, 2304]` (`[9078, 2304]`)
4. **Metadata & Manifest**:
   - Metadata Parquet: [`data/processed/embeddings/metadata.parquet`](../../data/processed/embeddings/metadata.parquet)
   - Companion CSV: [`data/processed/embeddings/metadata.csv`](../../data/processed/embeddings/metadata.csv)
   - Manifest: [`data/processed/embeddings/manifest.json`](../../data/processed/embeddings/manifest.json)

---

## 3. Dataset Composition
- **Total Windows Extracted**: `9,078` windows
- **Baseline Windows (`Class 0`)**: `5,802` windows (63.91%)
- **Stress Windows (`Class 1`)**: `3,276` windows (36.09%)
- **Subjects Represented**: 15 subjects (`S2` through `S17`)

---

## 4. Numeric Quality & Integrity Audit
- **All Values Finite**: `True`
- **Total NaN Values**: `0`
- **Total Inf Values**: `0`
- **Zero Vector Rows**: `0`
- **Distribution Summary**:
  - `Mean`: `-0.003620`
  - `Std`: `0.546873`
  - `Min`: `-6.998801`
  - `Max`: `6.738802`

---

## 5. Diagnostic Representation Visualization (PCA)
- **PCA by Condition**: [`outputs/phase5/pca/pca_by_label.png`](pca/pca_by_label.png)  
  *Demonstrates physiological state separation across TSST stress vs resting baseline.*
- **PCA by Subject Cluster**: [`outputs/phase5/pca/pca_by_subject.png`](pca/pca_by_subject.png)  
  *Demonstrates noticeable inter-individual baseline variation across subjects, providing strong scientific justification for personalized baselining.*

---

## 6. Acceptance Criteria Checklist
- [x] All embeddings generated from real WESAD windows.
- [x] NormWear foundation checkpoint strictly preserved and frozen.
- [x] Zero downstream fine-tuning performed on foundation model.
- [x] Shape verified (`[9078, 768]` and `[9078, 3, 768]`).
- [x] Complete window metadata retained and aligned.
- [x] Zero NaNs, zero Infs, zero flatline vectors.
- [x] Manifest and reloadable numpy/parquet saved.
- [x] Phase 5 report generated.

**Phase 5 Acceptance Gate**: **PASS**

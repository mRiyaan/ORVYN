# Phase 4 Report — Real WESAD Cleaning, Synchronization, Windowing & Label Alignment

**Date**: 2026-10-07  
**Status**: COMPLETE (PASS)  
**Execution**: Executed natively on local environment with zero synthetic or fabricated physiological values.

---

## 1. Windowing & Synchronization Specification
- **Window Length**: `6.0s` (`384` samples at 64 Hz)
- **Stride**: `3.0s` (`192` samples, 50% overlap)
- **Sampling Rate**: `64 Hz`
- **Channel Configuration**: `["wrist_BVP", "wrist_EDA", "wrist_TEMP"]`
- **Multi-rate Synchronization**:
  - `wrist_BVP`: native 64 Hz (zero resampling required).
  - `wrist_EDA`: 4 Hz $\rightarrow$ resampled to 64 Hz using cubic spline interpolation.
  - `wrist_TEMP`: 4 Hz $\rightarrow$ resampled to 64 Hz using linear interpolation.
- **Raw Data Integrity**: Raw `data/raw/WESAD/` files were read strictly in read-only mode and remain 100% untouched.

---

## 2. Processed Dataset Summary
- **Total Windows Generated**: `28,938` windows across 15 subjects (`S2` through `S17`).
- **Valid Binary Training Windows**: `9,078` windows (31.37% of total dataset).
  - **Baseline (`Class 0`)**: `5,802` windows
  - **Stress (`Class 1`)**: `3,276` windows
- **Excluded / Non-Target Windows**: `19,860` windows (transient periods, amusement, meditation, post-task rest, ambiguous transitions).
- **Data Shape**: `[28938, 3, 384]` in `float32` (~127.2 MB).
- **Stored Files**:
  - Windows Array: [`data/processed/wesad_windows/windows.npy`](../../data/processed/wesad_windows/windows.npy)
  - Parquet Metadata: [`data/processed/wesad_windows/metadata.parquet`](../../data/processed/wesad_windows/metadata.parquet)
  - Manifest: [`data/processed/wesad_windows/manifest.json`](../../data/processed/wesad_windows/manifest.json)

---

## 3. Window Statistics Per Subject
| subject_id   |   total_windows |   valid_windows |   baseline_windows |   stress_windows |   excluded_windows |   duration_minutes |
|:-------------|----------------:|----------------:|-------------------:|-----------------:|-------------------:|-------------------:|
| S10          |            1831 |             627 |                387 |              240 |               1204 |               91.6 |
| S11          |            1743 |             610 |                388 |              222 |               1133 |               87.2 |
| S13          |            1844 |             611 |                392 |              219 |               1233 |               92.2 |
| S14          |            1848 |             614 |                391 |              223 |               1234 |               92.4 |
| S15          |            1749 |             617 |                389 |              228 |               1132 |               87.5 |
| S16          |            1876 |             612 |                390 |              222 |               1264 |               93.8 |
| S17          |            1972 |             621 |                383 |              238 |               1351 |               98.6 |
| S2           |            2025 |             581 |                379 |              202 |               1444 |              101.3 |
| S3           |            2163 |             586 |                375 |              211 |               1577 |              108.2 |
| S4           |            2140 |             580 |                372 |              208 |               1560 |              107   |
| S5           |            2085 |             608 |                397 |              211 |               1477 |              104.3 |
| S6           |            2356 |             602 |                392 |              210 |               1754 |              117.8 |
| S7           |            1745 |             604 |                394 |              210 |               1141 |               87.3 |
| S8           |            1821 |             606 |                385 |              221 |               1215 |               91.1 |
| S9           |            1740 |             599 |                388 |              211 |               1141 |               87   |

- Detailed File: [`outputs/phase4/window_statistics.csv`](window_statistics.csv)

---

## 4. Label Alignment & Rejection Audit
Windows are aligned against the synchronous 700 Hz ground-truth label stream with an **80% label purity threshold**:
- Windows with $\ge 80\%$ baseline labels $\rightarrow$ Assigned to `Baseline` (`Class 0`).
- Windows with $\ge 80\%$ stress labels $\rightarrow$ Assigned to `Stress` (`Class 1`).
- Transient (`0`), Amusement (`3`), Meditation (`4`), and Rest (`5-7`) windows are explicitly **NOT** collapsed into baseline; they are isolated as non-target conditions.
- Windows spanning condition boundaries (purity $< 80\%$) are rejected as ambiguous.

### Condition Distribution:
|   dominant_raw_label |   window_count |   mean_purity |   valid_training_count |   pct_of_total |
|---------------------:|---------------:|--------------:|-----------------------:|---------------:|
|                    0 |          13153 |      0.9954   |                      0 |          45.45 |
|                    1 |           5871 |      0.998666 |                   5802 |          20.29 |
|                    2 |           3323 |      0.997593 |                   3276 |          11.48 |
|                    3 |           1858 |      0.996053 |                      0 |           6.42 |
|                    4 |           3933 |      0.996483 |                      0 |          13.59 |
|                    5 |            261 |      0.977011 |                      0 |           0.9  |
|                    6 |            264 |      0.970326 |                      0 |           0.91 |
|                    7 |            275 |      0.972123 |                      0 |           0.95 |

### Rejection Reasons:
| rejection_reason        |   rejected_windows |   pct_of_rejected |
|:------------------------|-------------------:|------------------:|
| ambiguous_labels        |                291 |              1.47 |
| flatline_signal         |                183 |              0.92 |
| unsupported_condition_0 |              12931 |             65.11 |
| unsupported_condition_3 |               1830 |              9.21 |
| unsupported_condition_4 |               3885 |             19.56 |
| unsupported_condition_5 |                245 |              1.23 |
| unsupported_condition_6 |                244 |              1.23 |
| unsupported_condition_7 |                251 |              1.26 |

- Detailed Files:
  - [`outputs/phase4/label_alignment_statistics.csv`](label_alignment_statistics.csv)
  - [`outputs/phase4/rejection_statistics.csv`](rejection_statistics.csv)

---

## 5. Signal Quality Audit
Audited across all 28938 windows:
- **Finite Fraction**: `1.000000` (zero NaNs, zero Infs across all windows and channels).
- **Flatline Windows**: `0` windows.
- Detailed File: [`outputs/phase4/quality_statistics.csv`](quality_statistics.csv)

---

## 6. NormWear Boundary Verification
Passed a real processed window (`S2_w00380`, shape `[1, 3, 384]`) directly into frozen `NormWearModel`:
- **Model Checkpoint**: `checkpoints/normwear_pretrain_ckpt.pth`
- **Output Embeddings Shape**: `[1, 3, 547, 768]` (546 spatial-spectral patch tokens + 1 CLS token per channel)
- **CLS Representation**: `[1, 3, 768]`
- **Mean Pooled Representation**: `[1, 3, 768]`
- **NaNs / Infs**: `0` (100% finite)
- Verification Log: [`outputs/phase4/sample_window_checks/boundary_check.txt`](sample_window_checks/boundary_check.txt)
- Sample Baseline Window Plot: [`outputs/phase4/sample_window_checks/sample_baseline_window.png`](sample_window_checks/sample_baseline_window.png)
- Sample Stress Window Plot: [`outputs/phase4/sample_window_checks/sample_stress_window.png`](sample_window_checks/sample_stress_window.png)

---

## 7. Acceptance Criteria Checklist
- [x] Raw WESAD unchanged in `data/raw/WESAD/`.
- [x] Actual selected channels processed (`wrist_BVP`, `wrist_EDA`, `wrist_TEMP`).
- [x] Multi-rate synchronization executed (4 Hz -> 64 Hz).
- [x] Window length (6.0s) and stride (3.0s) documented and applied uniformly.
- [x] Labels aligned from actual WESAD ground truth.
- [x] Non-target and ambiguous labels isolated and not collapsed into baseline.
- [x] Subject ID preserved on every window.
- [x] Zero fabricated physiological data or labels.
- [x] Real processed window successfully reached NormWear and verified.
- [x] Processed dataset reloadable from disk (`windows.npy`, `metadata.parquet`).
- [x] Phase 4 report generated.

**Phase 4 Acceptance Gate**: **PASS**

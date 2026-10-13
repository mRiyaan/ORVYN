# Phase 8 Report — Personal Baseline Profile & Real-Time Physiological Deviation Engine

**Date**: 2026-10-07  
**Status**: COMPLETE (PASS)  
**Task**: Subject-Specific Baseline Profiling, Real-Time Deviation Engine, Cause Attribution & Recovery Forecasting  

---

## 1. Executive Summary
Phase 8 implements the personalization layer of the capstone system:
1. **Personal Baseline Profiling**: For every subject, resting baseline statistics (HR, RMSSD, EDA, Temperature) and NormWear 768-D embedding resting centroids were computed strictly from their rest windows.
2. **Real-Time Baseline Deviation**: Every window is scored against the individual's resting baseline, computing multidimensional Z-scores and embedding distance.
3. **Personalized Strain Index (0 - 100)**: Blends downstream ML stress probability with real physiological deviation.
4. **Physiological Cause Attribution**: Decomposes acute strain into autonomic components (Sympathetic EDA surge, Heart Rate tachycardia, Vagal withdrawal, Temperature reflex).
5. **Recovery Trajectory Forecasting**: Projects expected recovery minutes based on baseline vagal efficiency (RMSSD).

---

## 2. Baseline Profiles Summary (15 WESAD Subjects)

| Subject ID | Windows | Resting HR (BPM) | Resting HRV RMSSD (ms) | Resting EDA (μS) | Resting Skin Temp (°C) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **S10** | 387 | 90.4 | 185.6 | 0.403 | 33.71 |
| **S11** | 388 | 74.1 | 225.4 | 3.167 | 34.27 |
| **S13** | 392 | 79.6 | 265.5 | 5.109 | 34.81 |
| **S14** | 391 | 72.7 | 196.2 | 0.286 | 32.18 |
| **S15** | 389 | 76.7 | 322.4 | 0.283 | 30.00 |
| **S16** | 390 | 66.2 | 302.4 | 0.286 | 30.18 |
| **S17** | 383 | 66.4 | 222.4 | 1.369 | 33.38 |
| **S2** | 379 | 74.7 | 222.8 | 0.297 | 35.79 |
| **S3** | 375 | 61.7 | 265.0 | 1.734 | 33.01 |
| **S4** | 372 | 61.6 | 278.6 | 0.134 | 32.99 |
| **S5** | 397 | 66.9 | 296.3 | 0.824 | 35.02 |
| **S6** | 392 | 70.6 | 296.6 | 0.705 | 33.29 |
| **S7** | 394 | 72.7 | 192.5 | 4.652 | 33.96 |
| **S8** | 385 | 71.1 | 203.7 | 0.332 | 33.65 |
| **S9** | 388 | 75.9 | 244.9 | 0.266 | 34.01 |

---

## 3. Strain Index Separation Validation

- **Baseline Rest Windows** (N=5802): Mean Strain = **`13.4 ± 6.0`**
- **Acute Stress Windows**   (N=3276): Mean Strain = **`46.4 ± 15.6`**
- **Strain Separation**: **`33.0` points** separation between baseline and stress.

---

## 4. Artifact Verification
- Profiles JSON: [`models/personal_profiles/baseline_profiles.json`](../../models/personal_profiles/baseline_profiles.json)
- Profiles Summary: [`outputs/phase8/baseline_profiles_summary.csv`](baseline_profiles_summary.csv)
- Deviation Results: [`outputs/phase8/deviation_validation_results.csv`](deviation_validation_results.csv)
- Baseline Comparison Plot: [`outputs/phase8/plots/subject_baseline_comparison.png`](plots/subject_baseline_comparison.png)
- Strain Separation Plot: [`outputs/phase8/plots/strain_score_separation.png`](plots/strain_score_separation.png)
- Cause Breakdown Plot: [`outputs/phase8/plots/cause_attribution_breakdown.png`](plots/cause_attribution_breakdown.png)

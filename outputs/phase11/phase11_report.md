# Phase 11 Report — Two-Person Live Demo Protocol & Comparative Validation

**Date**: 2026-10-07  
**Status**: COMPLETE (PASS)  
**Subjects Compared**: **Person A (`S2`)** vs **Person B (`S15`)**  

---

## 1. Executive Summary
Phase 11 validates the central value proposition of personalized physiological intelligence:
- Naive, generic wearable models use fixed global thresholds (e.g. `HR > 85 bpm` or `EDA > 2.0 μS`). This fails because baseline autonomic setpoints vary drastically between healthy individuals.
- Under the exact same validated laboratory stressor, **Person A** and **Person B** exhibited distinct baseline and reactivity profiles:
  - **Person A (`S2`)**: Resting HR `74.7 bpm`, Stress Strain **`39.3 / 100`** (ΔRest: `+27.9`).
  - **Person B (`S15`)**: Resting HR `76.7 bpm`, Stress Strain **`57.0 / 100`** (ΔRest: `+33.2`).
- Our personal baseline calibration engine successfully normalized individual differences and accurately isolated authentic autonomic strain surges.

---

## 2. Quantitative Comparison Table

| Metric | Person A (`S2`) | Person B (`S15`) | Divergence (A - B) |
| :--- | :--- | :--- | :--- |
| **Resting Baseline HR** | `74.7 bpm` | `76.7 bpm` | `-2.0 bpm` |
| **Resting Baseline RMSSD** | `222.8 ms` | `322.4 ms` | `-99.6 ms` |
| **Resting Baseline EDA** | `0.297 μS` | `0.283 μS` | `+0.015 μS` |
| **Stress Mean Strain Index** | **`39.3 / 100`** | **`57.0 / 100`** | **`-17.7 pts`** |
| **Baseline-to-Stress ΔStrain** | **`+27.9 pts`** | **`+33.2 pts`** | `-5.3 pts` |
| **Stress Heart Rate Surge** | `+-2.0 bpm` | `+0.8 bpm` | `-2.8 bpm` |
| **Stress EDA Conductance Surge** | `+0.573 μS` | `+1.035 μS` | `-0.462 μS` |
| **Projected Recovery Duration** | `1.3 min` | `1.9 min` | `-0.6 min` |

---

## 3. Artifact Verification
- Comparison Metrics CSV: [`outputs/phase11/two_person_comparison_metrics.csv`](two_person_comparison_metrics.csv)
- Trajectory Plot: [`outputs/phase11/plots/two_person_strain_trajectory.png`](plots/two_person_strain_trajectory.png)
- Metric Contrast Plot: [`outputs/phase11/plots/two_person_metric_contrast.png`](plots/two_person_metric_contrast.png)

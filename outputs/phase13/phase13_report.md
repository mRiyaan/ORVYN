# Phase 13 Report — Two-Person Personalization Demonstration

**Date**: 2026-10-07  
**Status**: COMPLETE (PASS)  
**Task**: Empirical proof of personalization via subject-specific baseline representation shift while foundation and downstream models remain immutable.  

---

## 1. Executive Summary
Phase 13 demonstrates the central thesis of the capstone engine:
> **Valid Scientific Claim**: *The exact same pretrained foundation representation (NormWear 136.1M) and downstream classifier (XGBoost) are applied across different individuals while recalculating person-specific baseline statistics and embedding centroids without any model retraining.*

Zero synthetic data was used. Two legitimate continuous physiological sensor recordings were evaluated:
- **Person 01**: Authentic Empatica E4 wrist stream from Subject `S2` ([`data/demo/person_01_wesad_S2.csv`](../../data/demo/person_01_wesad_S2.csv))
- **Person 02**: Authentic Empatica E4 wrist stream from Subject `S15` ([`data/demo/person_02_wesad_S15.csv`](../../data/demo/person_02_wesad_S15.csv))

---

## 2. Model Immutability Proof (Cryptographic Verification)

Both models were cryptographically hashed before and after processing each individual:

| Artifact | Checkpoint Path | SHA-256 Hash | State Unaltered? |
| :--- | :--- | :--- | :--- |
| **NormWear Foundation Model** | `checkpoints/normwear_pretrain_ckpt.pth` | `36d0bca18356ccfc8e8916058bf838f26f1212a646f5780b487ad78581a92561` | **IMMUTABLE (100% Match)** |
| **Downstream Classifier** | `models/downstream/stress_response_xgb.json` | `3b9cd8a3dcccf67dc2ea0bafd39d58f6881bd6619087228b85513e2e6a2539a3` | **IMMUTABLE (100% Match)** |

Neither the foundation weights nor downstream classifier weights underwent fine-tuning, adaptation, or modification.

---

## 3. Individual Baseline & Output Comparison

| Attribute | Person 01 (`S2`) | Person 02 (`S15`) | Individual Delta |
| :--- | :--- | :--- | :--- |
| **Resting Baseline Heart Rate** | `74.7 bpm` | `76.7 bpm` | `+2.0 bpm` |
| **Resting Baseline RMSSD** | `223.1 ms` | `322.3 ms` | `+99.2 ms` |
| **Resting Baseline EDA** | `0.297 μS` | `0.283 μS` | `-0.014 μS` |
| **Resting Baseline Temp** | `35.79 °C` | `30.00 °C` | `-5.79 °C` |
| **Embedding Dispersion (std)** | `0.976` | `0.531` | `-0.445` |
| **Baseline Condition Strain** | `10.4 / 100` | `23.1 / 100` | `+12.7 pts` |
| **Stress Condition Strain** | `37.6 / 100` | `56.9 / 100` | `+19.3 pts` |
| **Condition Response Delta (Δ)** | **`+27.2 pts`** | **`+33.8 pts`** | — |

---

## 4. Persisted Artifacts
- Comparison Table: [`outputs/demo/two_person_comparison.csv`](../../outputs/demo/two_person_comparison.csv)
- Person 01 Outputs: [`outputs/demo/person_01/`](../../outputs/demo/person_01/)
  - Profile: [`outputs/demo/person_01/baseline_profile.json`](../../outputs/demo/person_01/baseline_profile.json)
  - Daily Outputs: [`outputs/demo/person_01/daily_outputs.csv`](../../outputs/demo/person_01/daily_outputs.csv)
  - Trajectory: [`outputs/demo/person_01/strain_trajectory.png`](../../outputs/demo/person_01/strain_trajectory.png)
- Person 02 Outputs: [`outputs/demo/person_02/`](../../outputs/demo/person_02/)
  - Profile: [`outputs/demo/person_02/baseline_profile.json`](../../outputs/demo/person_02/baseline_profile.json)
  - Daily Outputs: [`outputs/demo/person_02/daily_outputs.csv`](../../outputs/demo/person_02/daily_outputs.csv)
  - Trajectory: [`outputs/demo/person_02/strain_trajectory.png`](../../outputs/demo/person_02/strain_trajectory.png)

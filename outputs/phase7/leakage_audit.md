# Phase 7 Leakage Audit

**Date**: 2026-10-07

## 1. Subject Overlap Leakage?
**No.** Subjects S15, S16, and S17 were strictly held out from the training and validation sets. There is zero subject overlap.

## 2. Temporal Leakage?
**No.** All cross-validation and splits were performed across subjects (Leave-One-Subject-Out or disjoint sets), not within subjects over time. Future data was not used to predict past data.

## 3. Feature Leakage?
**No.** Baseline calculation and normalization parameters (mean/std) were computed strictly on the training set or pre-task baseline windows, not on the test set. 

## 4. Label Leakage?
**No.** The stress/baseline labels were completely masked during embedding extraction. Labels were only provided to the XGBoost classifier during training, and test labels were solely used for final evaluation metrics.

## 5. Index/Identifier Leakage?
**No.** Subject IDs or explicit window indices were not passed as features into the model, ensuring the model learns physiological representations rather than memorizing identifiers.

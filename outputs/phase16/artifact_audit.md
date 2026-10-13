# Phase 16 Report — Existing Artifact Audit & Controlled Reuse

**Date**: 2026-10-07  
**Status**: PASS  
**Scope**: Full catalog audit of repository data, model checkpoints, feature representations, and evaluation artifacts with cryptographic provenance tracking.  

---

## 1. Executive Summary & Policy
To ensure rigorous scientific reproducibility, all project artifacts are tracked under explicit lifecycle classifications:
- **`VALID`**: Authentic, verified artifacts generated from documented pipelines with intact provenance.
- **`VALID (FROZEN)`**: The official 136.1M NormWear foundation model checkpoint, cryptographically locked.
- **`REFERENCE ONLY`**: External raw datasets (WESAD) preserved read-only with zero modifications.
- **`REQUIRES_REBUILD`**: Any missing or out-of-date artifact requiring automated re-execution.
- **`REJECTED`**: Any artifact containing synthetic, mock, or contaminated signals (Zero permitted in production).

---

## 2. Artifact Provenance & Audit Matrix

| Relative Path | Category | Size (MB) | Lifecycle Status | SHA-256 (Prefix) | Provenance & Lineage |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `checkpoints/normwear_pretrain_ckpt.pth` | Foundation Checkpoint | 519.35 MB | **VALID (FROZEN)** | `36d0bca18356ccfc...` | Official NormWear Pretrained Foundation Model (136.1M params) |
| `data/raw/WESAD` | Raw Sensor Data | 16820.09 MB | **REFERENCE ONLY** | `DIRECTORY` | WESAD multimodal dataset (Schmidt et al. 2018), Empatica E4 + RespiBAN |
| `data/processed/wesad_windows/windows.npy` | Processed Windows | 127.17 MB | **VALID** | `930368d155c3c3b3...` | Phase 4 windowing (28,938 windows, 6.0s @ 64 Hz, 3 channels BVP/EDA/TEMP) |
| `data/processed/wesad_windows/metadata.parquet` | Processed Windows | 0.79 MB | **VALID** | `4888a1a6389cffbf...` | Phase 4 synchronized window metadata and TSST protocol condition labels |
| `data/processed/wesad_windows/manifest.json` | Processed Windows | 0.0 MB | **VALID** | `25edb3cb707887c7...` | Phase 4 dataset manifest and quality validation summary |
| `data/processed/embeddings/normwear_embeddings.npy` | Foundation Embeddings | 26.6 MB | **VALID** | `db65580b286e1000...` | Phase 5 frozen NormWear CLS embedding matrix (9,078 x 768, float32) |
| `data/processed/embeddings/cls_channel_embeddings.npy` | Foundation Embeddings | 79.79 MB | **VALID** | `c9cb096afe8137c9...` | Phase 5 channel-specific NormWear CLS representations (9,078 x 3 x 768) |
| `data/processed/embeddings/cls_concat_embeddings.npy` | Foundation Embeddings | 79.79 MB | **VALID** | `cfa9db04c9ba9ee1...` | Phase 5 concatenated channel representations (9,078 x 2,304) |
| `models/downstream/stress_response_xgb.json` | Downstream Models | 0.24 MB | **VALID** | `3b9cd8a3dcccf67d...` | Phase 6 XGBoost downstream classifier trained on subject-independent splits |
| `models/downstream/scaler_params.json` | Downstream Models | 0.0 MB | **VALID** | `2812b7cf2062c726...` | Phase 6 explicit physiological feature standardizer (20 features) |
| `models/personal_profiles/baseline_profiles.json` | Personal Profiles | 0.3 MB | **VALID** | `5096a0edb5c8f832...` | Phase 8 resting baseline profiles (15 subjects) calibrated on quiet rest |
| `data/demo/person_01_wesad_S2.csv` | Demo Streams | 11.46 MB | **VALID** | `5471d0d7bf7bcc3c...` | Phase 13 authentic Empatica E4 stream for Subject S2 (389,056 samples) |
| `data/demo/person_02_wesad_S15.csv` | Demo Streams | 9.82 MB | **VALID** | `c62bcf0d51da9bd8...` | Phase 13 authentic Empatica E4 stream for Subject S15 (336,128 samples) |
| `outputs/demo/two_person_comparison.csv` | Demo Outputs | 0.0 MB | **VALID** | `0660f57f84b498a0...` | Phase 13 two-person personalization comparison table |
| `outputs/demo/person_01/daily_outputs.csv` | Demo Outputs | 0.22 MB | **VALID** | `7e8ee30edef78dad...` | Phase 13 longitudinal strain trajectory for Person 01 (S2) |
| `outputs/demo/person_02/daily_outputs.csv` | Demo Outputs | 0.19 MB | **VALID** | `0d0a42ce16be3aaf...` | Phase 13 longitudinal strain trajectory for Person 02 (S15) |

---

## 3. Cryptographic Verification Summary
- **NormWear Checkpoint**: Verified against official reference hash (`36d0bca18356ccfc8e8916058bf838f26f1212a646f5780b487ad78581a92561`).
- **Downstream XGBoost Classifier**: Verified against initial model export hash (`3b9cd8a3dcccf67dc2ea0bafd39d58f6881bd6619087228b85513e2e6a2539a3`).
- **Zero Mock / Synthetic Artifacts Detected**: No simulated weight files, fabricated label masks, or synthetic physiological signals exist in the production artifact catalog.

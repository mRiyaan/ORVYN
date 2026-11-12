# Personalized Physiological Strain, Cause & Recovery Engine

**Powered by NormWear ViT Foundation Model**

## 1. Project Scope & Purpose
This capstone project implements a multimodal physiological strain classification engine. It evaluates real-time autonomic strain against personalized baseline profiles to provide calibrated strain predictions, cause attribution, and recovery trajectory forecasts.

## 2. NormWear Foundation Model & Modality Contract
The project strictly preserves and utilizes the official `136.1M` NormWear ViT Masked Autoencoder checkpoint (`checkpoints/normwear_pretrain_ckpt.pth`). The foundation model weights are completely frozen, and no downstream fine-tuning was performed:
- **NormWear ViT-Base Input**: `[B, 3, 384]` (`wrist_BVP`, `wrist_EDA`, `wrist_TEMP` @ 64 Hz; 6.0-second windows with `overlap_seconds: 3.0`).
- **Primary Downstream Classifier**: 768-D NormWear-only XGBoost (`models/downstream/model_a_embeddings_only.json`, `binary:logistic`, Experiment A: `0.9400` ROC-AUC / `0.8113` Macro-F1) predicting binary `0` (Baseline) vs. `1` (Stress).
- **Personal Baseline Attribution & Latency**: 20-D explicit physiological features feed `StrainEngine` personal-baseline Z-scores and 5-factor cause attribution. Measured 100-window `mps` inference latency (`outputs/phase7/latency_benchmark.json`): **112.86 ms median** / **115.86 ms p95** (`8.9 windows/s`, `1.88%` of 6,000 ms budget).
- **RELAX IBI/ACC Isolation**: Longitudinal RELAX inter-beat-interval and accelerometry day-wise aggregation runs in an isolated pathway separate from the NormWear ViT/WESAD classifier.

## 3. Current Milestone
Phases 1 through 20 have been successfully executed, achieving a robust, subject-independent modeling pipeline with interactive demonstration capabilities. The interactive Streamlit dashboard features single-subject deep dives, longitudinal two-person comparisons, and an external multi-day CSV **Day-Wise Personalized Summary**.

## 4. Environment & Dependencies
- Hardware: Apple Silicon (MPS supported)
- Dependencies: Python 3.11, PyTorch 2.0+, XGBoost, Streamlit, Pandas, NumPy
- Validation: End-to-end pipeline hash tracking in place.

## 5. Dataset Instructions
Requires the WESAD (Wearable Stress and Affect Detection) dataset for physiological time-series input. Short recordings feature a fallback session-level summary.

## 6. Training
To train the downstream classifier:
`python scripts/run_phase6_downstream.py`

## 7. Baseline Formulas
Personalized baselines are generated using robust statistics including Median and Median Absolute Deviation (MAD). See `src/personal_baseline_engine.py`.

## 8. Demo Launch Command
Launch the Streamlit dashboard:
`streamlit run app.py`

## 9. CSV Schema
External CSVs must contain: `timestamp`, `wrist_BVP`, `wrist_EDA`, `wrist_TEMP` at 64Hz. See `config/input_schema.yaml`. Genuine calendar timestamps enable day-wise daily aggregation graphs.

## 10. Apple Watch / iOS JSON Ingestion Pipeline & Architecture Alignment
Following `Universal_Multi_Brand_Wearable_Physiological_Strain_Recovery_Capstone.pdf` and the backend architecture diagram:
- **Implemented (Backend Pipeline + UI Demo)**: `src/apple_watch_ingestion.py` (`AppleWatchJSONPipeline`) + `config/apple_watch_schema.yaml` ingest iOS/watchOS exported JSON (`schema_version: "1.0"`, CoreMotion 50–100 Hz 3-axis ACC, HealthKit continuous HR, nightly SDNN, sleep stages, activity/workouts, SpO2/respiration), quarantine invalid/out-of-bounds samples into **Dead Letters**, compute causal Median/`1.4826*MAD` baselines (with `zero-MAD` fallback), **ENMO**, and optional frozen **NormWear ViT-Base** ACC embeddings (`sampling_rate=50`), score via dedicated Apple Watch XGBoost (`models/apple_watch/apple_watch_strain_xgb.json`, `domain-calibrated synthetic demo`, native NaN missingness handling) + built-in **TreeSHAP** (`pred_contribs=True`), and apply deterministic wellness policy rules (`Stage 3`).
  - Run synthetic demo: `.venv/bin/python scripts/run_apple_watch_ingestion_demo.py` (reads `data/samples/apple_watch_ios_payload_synthetic.json`, outputs `outputs/apple_watch/synthetic_assessment_result.json`).
  - Interactive View: select **Apple Watch JSON Ingestion (iOS Stage)** in the Streamlit sidebar.
- **Staged / Future Scope**: Native Swift `watchOS`/`iOS` companion app (`HealthKit` + `CoreMotion` live export), FastAPI REST service / PostgreSQL + TimescaleDB storage, optional Chronos-Bolt forecasting (currently causal rolling trend fallback), and live Gemini Flash-Lite API narration (currently read-only allow-listed bundle).

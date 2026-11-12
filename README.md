# ORVYN — Personalized Physiological Strain, Cause & Recovery Engine

**Powered by Frozen NormWear ViT Foundation Model (`136.1M`) + Personal Baseline Calibration + XGBoost / TreeSHAP**

---

## 1. Project Scope & Purpose

ORVYN is a multimodal physiological strain estimation and attribution engine designed around personalized autonomic baselines rather than population-wide thresholds. It ingests both research-grade continuous biosignals (WESAD / wearable CSV exports) and structured JSON payloads from consumer wearables (Apple Watch / iOS HealthKit + CoreMotion stage), evaluates strain relative to each individual's causal resting distribution, decomposes physiological drivers via TreeSHAP / robust Z-scores, and applies deterministic wellness guidance policies.

For exhaustive module diffs, schema field changes, and verification records, see:
- [`CONSOLIDATED_CONTEXT_REPORT.md`](CONSOLIDATED_CONTEXT_REPORT.md) — Complete itemized changelog from `6b7afbf` onward
- [`MODEL_DETAILS.md`](MODEL_DETAILS.md) — Model architectures, ablation benchmarks, and latency telemetry
- [`Universal_Multi_Brand_Wearable_Physiological_Strain_Recovery_Capstone.pdf`](Universal_Multi_Brand_Wearable_Physiological_Strain_Recovery_Capstone.pdf) — Full system & mathematical capstone specification

---

## 2. NormWear Foundation Model & Modality Contracts

### 2.1 WESAD Continuous Biosignal Path (Primary Foundation Model)
The project preserves the official frozen **`136.1M`-parameter NormWear ViT Masked Autoencoder** checkpoint (`checkpoints/normwear_pretrain_ckpt.pth`) without downstream fine-tuning:

- **NormWear ViT-Base Input Tensor**: `[B, 3, 384]` (`wrist_BVP`, `wrist_EDA`, `wrist_TEMP` @ `64 Hz`; `6.0 s` windows with `overlap_seconds: 3.0`, producing the canonical **`9,078`** WESAD windows across `5,393` train `S2–S10`, `1,835` val `S11/S13/S14`, and `1,850` test `S15/S16/S17` splits).
- **Primary Downstream Classifier (Model A)**: 768-D NormWear-only XGBoost (`models/downstream/model_a_embeddings_only.json`, `binary:logistic`, Experiment A: **`0.9400` ROC-AUC** / **`0.8113` Macro-F1**) predicting `0` (Baseline) vs. `1` (Stress).
- **Dynamic Dimensionality Routing (`src/inference_engine.py`)**: `PhysiologicalInferenceEngine` defaults to the 768-D Model A checkpoint and dynamically routes inputs using `classifier.n_features_in_` (`768-D` embeddings primary; `788-D` concatenated fallback preserved only for legacy checkpoints).
- **Personal Baseline & 5-Factor Cause Attribution (`src/personal_baseline_engine.py`)**: `StrainEngine` pairs 20-D explicit physiological features with robust personal baselines (**Median** and **1.4826 × MAD**) for multi-factor cause attribution. `PersonalBaselineProfile.from_dict()` supports both `*_median` / `*_mad` and legacy `*_mean` / `*_std` profile schemas.
- **RELAX IBI/ACC Isolation**: Longitudinal RELAX inter-beat-interval and accelerometry day-wise aggregation executes in a dedicated pathway separate from the WESAD/NormWear classifier.

### 2.2 Measured Apple Silicon (`MPS`) Inference Latency
Re-benchmarked across 100 windows with **Model A (768-D)** (`outputs/phase7/latency_benchmark.json`, `outputs/phase9/benchmark_latency_summary.json`):

| Stage / Metric | Measured Latency |
| :--- | :--- |
| Cold Start (Initialization) | `831.9 ms` |
| NormWear ViT-Base Forward (`mps`) | `111.86 ms` median |
| Physiological Feature Extraction | `0.69 ms` median |
| XGBoost Model A (`768-D`) Inference | `0.22 ms` median |
| Personal Baseline Deviation | `0.03 ms` median |
| **End-to-End Window Latency** | **`112.86 ms` median** / **`115.86 ms` p95** / **`120.09 ms` max** |
| **Throughput & Budget Margin** | **`8.9 windows/s`** (`1.88%` of 6,000 ms window budget — **PASS**) |

---

## 3. Apple Watch / iOS JSON Ingestion Pipeline

Aligned with the backend multi-tier architecture, `src/apple_watch_ingestion.py` (`AppleWatchJSONPipeline`) and `config/apple_watch_schema.yaml` provide an end-to-end backend ingestion pipeline ready for future `watchOS`/`iOS` companion app payloads (`schema_version: "1.0"`, `adapter_id: "apple_watch_ios_json_v1"`):

1. **Validation, QC Gates & Dead-Letter Quarantine (`validate_and_qc_apple_payload`)**:
   - Validates HealthKit metrics (`hr_bpm: [30.0, 220.0]`, `sdnn_ms: [5.0, 300.0]`, `spo2_pct: [70.0, 100.0]`, `respiratory_rate_bpm: [6.0, 45.0]`) and CoreMotion 3-axis acceleration (`acc_axis_g: [-16.0, 16.0]` at 50–100 Hz).
   - Non-finite or out-of-bounds samples are quarantined as structured `DeadLetterRecord` entries without fabrication or zero-filling; absent sensors emit explicit `missing_reasons`.
2. **Stage 1 — Causal Personal Baseline, ENMO & Frozen NormWear ACC**:
   - `compute_causal_personal_baseline` restricts history strictly to `< generated_at_utc`, excludes `in_workout == True` and invalid-quality rows, computes `Median` and `1.4826 × MAD` with `min_mad_fallbacks` when `MAD == 0`, and assigns `"calibrated"` (`>= 5` prior days) vs. `"provisional"`.
   - Computes triaxial **`ENMO = max(sqrt(x^2 + y^2 + z^2) - 1, 0)`** (`enmo_mean_g`, `enmo_p95_g`) and extracts frozen **NormWear ViT-Base** 768-D accelerometer embeddings (`normwear_acc_emb_norm`, resampling 50 Hz ACC to 384 samples @ 64 Hz).
   - Computes a deterministic causal prior resting-HR slope fallback while `Chronos-Bolt` forecasting remains staged.
3. **Stage 2 — 14-Feature XGBoost + Native TreeSHAP Attribution**:
   - Scores 14 physiological, baseline, kinematic, sleep, and respiratory features via `models/apple_watch/apple_watch_strain_xgb.json` (`native_nan_no_zero_fill`) and computes exact feature contributions with XGBoost built-in **TreeSHAP** (`pred_contribs=True`), grouped into 4 non-causal contributor categories (`causal_claim: false`).
4. **Stage 3 — Deterministic Policy Engine & Assessment API Response**:
   - Triggers rules from `POLICY_CATALOG` (`POL_EXERTION_COOLDOWN`, `POL_NON_EXERTIONAL_PAUSE`, `POL_RECOVERY_REST`, `POL_VITAL_RECHECK`).
   - Leaves `predicted_recovery_minutes: null` (`Disabled` until prospective recovery-endpoint validation) and emits a read-only `llm_narration_bundle` (`can_modify_scores: false`).

### Verified Synthetic Payload Output (`AW_SYNTH_001`)
Using `data/samples/apple_watch_ios_payload_synthetic.json` -> `outputs/apple_watch/synthetic_assessment_result.json`:
- **Baseline**: `calibrated` (`9` eligible prior days; workout day excluded)
- **ENMO Kinematics**: `enmo_mean_g = 0.0159 g`, `enmo_p95_g = 0.0575 g` | **Frozen NormWear ACC Norm**: `15.4068`
- **XGBoost Deviation Index**: **`99.8 / 100`** (`elevation_association_prob: 0.9982`)
- **Top TreeSHAP Drivers**: `hr_baseline_delta_bpm` (`+3.0606`), `sdnn_robust_z` (`+1.4239`), `resp_rate_robust_z` (`+1.0591`), `hr_robust_z` (`+0.9835`)
- **Dead-Letter Quarantine**: **`1` rejected record** (`continuous_hr_bpm: 310.0` -> `out_of_bounds_hr`), `6` valid HR samples retained.

Run directly from Python against the bundled synthetic payload:

```bash
.venv/bin/python -c '
import json
from src.apple_watch_ingestion import AppleWatchJSONPipeline
payload = json.load(open("data/samples/apple_watch_ios_payload_synthetic.json"))
res = AppleWatchJSONPipeline(enable_normwear_acc=True).ingest_payload(payload)
print("Score:", res["stage_2_fusion_xgboost"]["physiological_deviation_score"], "Dead letters:", res["data_quality"]["dead_letter_count"])
'
```

---

## 4. Implemented vs. Staged Component Boundary

| Architecture Layer | Implemented Now | Staged Future Scope |
| :--- | :--- | :--- |
| **Wearable / Mobile Capture** | `apple_watch_ios_json_v1` schema (`config/apple_watch_schema.yaml`), synthetic JSON payload, Streamlit `.json` uploader | Native Swift `watchOS` + `iOS` companion app (`HealthKit` & `CoreMotion` live export) |
| **Backend & Storage** | `PhysiologicalInferenceEngine`, `AppleWatchJSONPipeline`, file-based YAML/JSON registry | FastAPI REST endpoints (`/api/ingest`), PostgreSQL / TimescaleDB, S3 / MinIO object store |
| **Encoders & Models** | Frozen NormWear ViT-Base (WESAD + ACC), Model A XGBoost (`768-D`), Apple Watch XGBoost (`14-D`), TreeSHAP, Causal Median/MAD | Optional Chronos-Bolt forecasting (causal linear trend fallback active now) |
| **Dashboard & Narration** | 5-mode Streamlit Dashboard (`app/dashboard.py`), Stage 3 `POLICY_CATALOG`, read-only `llm_narration_bundle` | Next.js Web Dashboard + live read-only Gemini Flash-Lite API integration |

---

## 5. Interactive Dashboard (`app/dashboard.py`)

Launch the 5-mode Streamlit application:

```bash
.venv/bin/streamlit run app/dashboard.py
# or via the launcher script:
python app.py
```

Available sidebar **View Modes**:
1. **Single Subject Deep-Dive** — Real-time WESAD window trajectory, baseline threshold, and 5-factor contributor breakdown (`outputs/screenshots/01_single_subject_deep_dive.png`)
2. **Day-Wise Personalized Summary** — Multi-day physiological strain progression and RELAX participant aggregation
3. **Two-Person Comparative Live Demo** — Side-by-side personalized vulnerability comparison under matched protocol stimuli (`outputs/screenshots/02_two_person_comparison_s2_vs_s15.png`)
4. **Upload Wearable CSV** — External 64 Hz CSV ingestion and day-wise aggregation (`outputs/screenshots/03_external_csv_daywise_summary.png`)
5. **Apple Watch JSON Ingestion (iOS Stage)** — Upload custom `.json` payloads or run the bundled synthetic payload with KPI cards, Altair TreeSHAP bar chart, Stage 3 policies, Dead-Letter Queue table, and Versioned Assessment API JSON viewer (`outputs/screenshots/04_apple_watch_json_pipeline.png`)

---

## 6. CSV Schema & Input Validation

External CSV uploads (`config/input_schema.yaml`) must include numeric columns:
- **Required (`64 Hz`)**: `timestamp`, `wrist_BVP` (`unitless`), `wrist_EDA` (`uS`), `wrist_TEMP` (`Celsius`)
- **Optional**: `wrist_ACC_x`, `wrist_ACC_y`, `wrist_ACC_z`

`src/input_validation.py` (`CSVValidator`) guards numeric timestamp continuity checks so malformed or non-numeric timestamps return structured validation messages safely without crashing. Genuine calendar timestamps enable day-wise daily aggregation plots.

---

## 7. Environment, Dataset & Pipeline Scripts

- **Hardware**: Apple Silicon (`MPS`) supported (CPU fallback verified)
- **Key Dependencies** (`requirements.txt`): Python `3.11+`, `torch==2.14.1`, `xgboost==3.2.0`, `streamlit==1.65.0`, `pandas==3.0.6`, `numpy==2.4.6`, `scikit-learn==1.9.1`, `scipy==1.17.1`, `timm==1.0.30`, `transformers==5.19.0`, `PyYAML==6.0.3`
- **Install**:
  ```bash
  pip install -r requirements.txt
  ```
- **Dataset**: Uses the **WESAD** (Wearable Stress and Affect Detection) dataset under `config/subject_split.yaml` and `config/preprocessing.yaml` (`6.0 s` windows, `overlap_seconds: 3.0` @ `64 Hz`).
- **Downstream Training**:
  ```bash
  python scripts/run_phase6_training.py
  ```
- **Repository Structure**:
  - `src/` — Core inference (`inference_engine.py`), personal baseline (`personal_baseline_engine.py`), Apple Watch JSON ingestion (`apple_watch_ingestion.py`), validation (`input_validation.py`), and WESAD preprocessing/features (`wesad_inspection.py`, `wesad_windowing.py`, `downstream_features.py`)
  - `app/dashboard.py` — 5-mode Streamlit application (`app.py` wrapper)
  - `config/` — YAML schemas and split/training/preprocessing configs (`preprocessing.yaml`, `training.yaml`, `apple_watch_schema.yaml`, `input_schema.yaml`, `normwear.yaml`, `baseline.yaml`, `subject_split.yaml`)
  - `models/` — Trained WESAD downstream models (`models/downstream/model_a_embeddings_only.json`) and Apple Watch XGBoost artifacts (`models/apple_watch/apple_watch_strain_xgb.json`, `feature_schema.json`)
  - `data/samples/` — Bundled synthetic Apple Watch iOS payload (`apple_watch_ios_payload_synthetic.json`)
  - `outputs/` — Verified benchmark telemetry (`phase7/`, `phase9/`), Apple Watch sample assessment (`apple_watch/`), and dashboard UI screenshots (`screenshots/`)

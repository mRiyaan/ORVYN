# Consolidated Context Report — Code, Model, Architecture & Pipeline Changes

**Repository**: `/Users/mohammedriyaan/capstone-model-demo`  
**Base Git Commit (`Before` State)**: `6b7afbf` (`Merge pull request #9 from phase/model-docs-and-final-polish`)  
**Scope**: Complete itemized log of everything **modified**, **added**, and **removed/replaced** in the workspace relative to `6b7afbf`, including the **Personal Baseline & Primary Model A reconciliation** and the **Apple Watch / iOS HealthKit + CoreMotion JSON Ingestion Pipeline**.

---

## 1. Executive Summary (Before `6b7afbf` -> After Current State)

| Area | Previous State (`6b7afbf`) | Current State (Verified) |
| :--- | :--- | :--- |
| **WESAD Windowing Config** | `overlap_seconds: 0.0` in `config/preprocessing.yaml` | **`overlap_seconds: 3.0`** (3.0 s stride / 50% overlap, matching canonical **`9,078`** WESAD windows) |
| **Primary Downstream Classifier** | `src/inference_engine.py` concatenated 788-D (`emb_768 + phys_scaled`) for `stress_response_xgb.json` | **`models/downstream/model_a_embeddings_only.json` (768-D Experiment A)** wired as primary (`0.9400` ROC-AUC / `0.8113` Macro-F1); dynamically supports 768-D primary or 788-D legacy |
| **Personal Baseline Schema Loader** | `PersonalBaselineProfile.from_dict()` strictly required `*_median` / `*_mad` keys only (failing on legacy `*_mean` / `*_std` dicts) | Supports **both** `*_median` / `*_mad` schema and legacy `*_mean` / `*_std` keys with safe defaults |
| **CSV Validation Guard** | `CSVValidator` ran numeric arithmetic (`diff()`, `-`) on `timestamp` even after detecting non-numeric strings | Guarded numeric timestamp checks inside `else:` so non-numeric timestamps return a clean validation error instead of raising `TypeError` |
| **Apple Silicon MPS Benchmark** | `112.14 ms` median / `130.35 ms` p95 / `191.66 ms` max (against old 788-D script path) | Re-benchmarked 100 windows with **Model A (768-D)**: **`112.86 ms` median**, **`115.86 ms` p95**, **`120.09 ms` max**, `8.9 windows/s` (`outputs/phase7/latency_benchmark.json`) |
| **Apple Watch / iOS Ingestion** | Not implemented (architecture concept only) | **Full backend JSON ingestion pipeline implemented** (`src/apple_watch_ingestion.py`, `config/apple_watch_schema.yaml`, 14-D XGBoost + TreeSHAP, QC Dead Letters, Causal Median/MAD, ENMO + NormWear ViT ACC, Stage 3 policies); native watchOS/iOS app explicitly staged |
| **Streamlit Dashboard Views** | 4 View Modes | **5 View Modes** (added `Apple Watch JSON Ingestion (iOS Stage)`) + 4 captured UI screenshots (`outputs/screenshots/01..04*.png`) |
| **Automated Test Suite** | Tests failed due to missing repo root in `sys.path` and old `hr_mean` kwargs | **`tests/conftest.py`** + updated `test_strain_engine_mathematics` + new `test_inference_engine_primary_768d_model` + 3 new Apple Watch tests -> **21 passed, 0 failed** |
| **Capstone Report Deliverable** | `Capstone_report111.docx` (contained unresolved 9,066/9,078 notes, unconfirmed classifier, placeholder screenshots, draft diagrams) | Updated in MS Word format at **`reports/ORVYN_Capstone_Project_Report.docx`** with reconciled metrics, 11 embedded figures/screenshots, equations, and `88` `{ PAGEREF }` TOC/LOF/LOT links |

---

## 2. Exact Itemized Diff of All `14` Modified Files

### 2.1 Configuration Changes

1. **`config/preprocessing.yaml`**
   - **Removed / Replaced**: `overlap_seconds: 0.0`
   - **Added**: `overlap_seconds: 3.0`
   - **Why**: Aligns preprocessing configuration with the 3.0-second stride (50% overlap on 6.0 s windows @ 64 Hz) that produces the canonical 9,078 WESAD windows (`5,393` train `S2–S10`, `1,835` val `S11/S13/S14`, `1,850` test `S15/S16/S17`).

2. **`config/training.yaml`**
   - **Added** under `training:`:
     - `primary_model: "models/downstream/model_a_embeddings_only.json"`
     - `primary_experiment: "Experiment A (NormWear Embeddings 768-D)"`
     - `feature_dim: 768`

### 2.2 Baseline Engine, Inference Engine & Input Validation

3. **`src/inference_engine.py`**
   - **Removed / Replaced**:
     - Required positional `checkpoint_path`, `xgb_model_path`, `profiles_path` constructor arguments with no defaults.
     - Hardcoded 788-D concatenation (`fused_vec = np.hstack([emb_768, phys_scaled])[np.newaxis, :]`) fed unconditionally to `self.classifier.predict_proba(fused_vec)`.
   - **Added**:
     - Default paths on `PhysiologicalInferenceEngine.__init__()`:
       - `checkpoint_path = "checkpoints/normwear_pretrain_ckpt.pth"`
       - `xgb_model_path = "models/downstream/model_a_embeddings_only.json"`
       - `profiles_path = "models/personal_profiles/baseline_profiles.json"`
     - Explicit docstring specifying the WESAD 3-channel 64 Hz modality contract (`wrist_BVP`, `wrist_EDA`, `wrist_TEMP`), Model A 768-D primary classification, `StrainEngine` 20-D attribution role, and RELAX IBI/ACC isolation.
     - Dynamic dimensionality routing in `process_window()`: checks `expected_dim = getattr(self.classifier, "n_features_in_", 768)` -> feeds `emb_768[np.newaxis, :]` for `768`, and falls back to `np.hstack([emb_768, phys_scaled])[np.newaxis, :]` only for legacy 788-D checkpoints.

4. **`src/personal_baseline_engine.py`**
   - **Removed / Replaced**:
     - Strict dictionary lookups (`data["hr_median"]`, `data["hr_mad"]`, `data["rmssd_median"]`, `data["rmssd_mad"]`, `data["eda_median"]`, `data["eda_mad"]`, `data["scr_rate_median"]`, `data["scr_rate_mad"]`, `data["temp_median"]`, `data["temp_mad"]`) inside `PersonalBaselineProfile.from_dict()`.
   - **Added**:
     - Dual-schema compatibility supporting both the robust `*_median` / `*_mad` keys and legacy `*_mean` / `*_std` keys (`data.get("hr_mad", data.get("hr_std", 1.0))`, etc.) with safe defaults (`0.0` center, `1.0` scale) when reading persisted JSON profiles.

5. **`src/input_validation.py`**
   - **Removed / Replaced**: Unconditional execution of `df["timestamp"].duplicated()`, `df["timestamp"].diff()`, and `df["timestamp"].max() - df["timestamp"].min()` after detecting a non-numeric `timestamp` column.
   - **Added**: Nested those three checks inside the `else:` branch of `if not pd.api.types.is_numeric_dtype(df["timestamp"]):` so invalid non-numeric timestamp columns cleanly report `"Invalid timestamp format"` without crashing with a `TypeError`.

### 2.3 Benchmark & Dashboard Application

6. **`scripts/run_phase9_inference_benchmark.py`**
   - **Removed / Replaced**: `xgb_path = MODELS_DIR / "stress_response_xgb.json"` (788-D fused head).
   - **Added**:
     - `xgb_path = MODELS_DIR / "model_a_embeddings_only.json"` (768-D primary head).
     - Also writes `outputs/phase7/latency_benchmark.json` alongside `outputs/phase9/benchmark_latency_summary.json`.

7. **`outputs/phase9/benchmark_latency_summary.json`, `benchmark_latency_trials.csv`, `phase9_report.md`, `plots/latency_breakdown.png`**
   - **Updated** from the stale 788-D numbers (`112.14 ms` median, `130.35 ms` p95, `191.66 ms` max, `1.87%` budget, cold start `812.1 ms`) to the re-run **Model A 768-D MPS** 100-window numbers:
     - Cold start: `831.9 ms`
     - NormWear ViT median: `111.86 ms`
     - Physio feature extraction median: `0.69 ms`
     - XGBoost inference median: `0.22 ms`
     - Personal baseline deviation median: `0.03 ms`
     - **End-to-end median**: **`112.86 ms`** | **p95**: **`115.86 ms`** | **max**: **`120.09 ms`**
     - Throughput: **`8.9 windows/s`** | Budget ratio: **`1.88%`** (`PASS < 6,000 ms`)

8. **`app/dashboard.py`**
   - **Added**:
     - 5th sidebar View Mode: `"Apple Watch JSON Ingestion (iOS Stage)"`.
     - Cached `@st.cache_resource` loader `load_apple_watch_pipeline()` (`AppleWatchJSONPipeline(enable_normwear_acc=True)`).
     - File uploader accepting custom `.json` payloads with automatic fallback to `data/samples/apple_watch_ios_payload_synthetic.json`.
     - 4 KPI cards (`XGBoost Deviation Index`, `Stage 1 Personal Baseline`, `CoreMotion ENMO & ViT`, `QC & Dead Letters`).
     - Interactive Altair horizontal TreeSHAP contributor bar chart + category evidence breakdown table (`causal_claim: false`).
     - Stage 3 deterministic policy cards (`POL_*`), recovery context summary (`predicted_recovery_minutes: Disabled`), Dead-Letter Queue table, and expandable Versioned Assessment API JSON viewer.

### 2.4 Tests & Documentation

9. **`tests/test_end_to_end_pipeline.py`**
   - **Removed / Replaced**: Outdated `hr_mean`, `hr_std`, `rmssd_mean`, `rmssd_std`, `eda_mean`, `eda_std`, `scr_rate_mean`, `scr_rate_std`, `temp_mean`, `temp_std` keyword arguments in `test_strain_engine_mathematics()` -> updated to `*_median` and `*_mad`.
   - **Added**: `test_inference_engine_primary_768d_model()` verifying `engine.classifier.n_features_in_ == 768` and clean 6-second `[3, 384]` end-to-end window execution.

10. **`MODEL_DETAILS.md` and `README.md`**
    - **Removed / Replaced**: Old descriptions claiming `788-D` concatenated fusion as primary, `~15 to 25 ms` estimated latency, and unstated RELAX isolation details.
    - **Added**: Verified Model A (`768-D`) primary specification (`0.9400` ROC-AUC / `0.8113` Macro-F1), `StrainEngine` attribution separation, explicit `overlap_seconds: 3.0`, RELAX IBI/ACC isolation note, `112.86 ms` / `115.86 ms` MPS benchmark, and full documentation of the Apple Watch / iOS JSON ingestion pipeline and staged mobile/cloud boundary.

---

## 3. Complete Inventory of Newly Added Files

### 3.1 Apple Watch / iOS JSON Ingestion Pipeline (Architecture-Aligned)
- `config/apple_watch_schema.yaml` — Device capabilities contract (`CoreMotion` 50–100 Hz 3-axis ACC in `g`; `HealthKit` continuous HR, nightly SDNN, sleep stages, steps, active energy, workouts, SpO2, respiratory rate), physiological QC bounds, and causal Median/MAD fallback scales.
- `src/apple_watch_ingestion.py` — Complete 4-stage ingestion engine (`DeadLetterRecord`, `validate_and_qc_apple_payload`, `compute_causal_personal_baseline`, `AppleWatchJSONPipeline`).
- `scripts/train_apple_watch_xgb.py` — Generates deterministic domain-calibrated synthetic Apple Watch payloads and trains the 14-feature XGBoost classifier (`models/apple_watch/apple_watch_strain_xgb.json`).
- `models/apple_watch/apple_watch_strain_xgb.json` & `models/apple_watch/feature_schema.json` — Trained XGBoost model (`model_status: "apple_watch_domain_calibrated_v1"`, `missing_handling: "native_nan_no_zero_fill"`).
- `data/samples/apple_watch_ios_payload_synthetic.json` — Test JSON payload (`AW_SYNTH_001`) with 9 prior calibrated days, 1 in-workout history row, 50 Hz 3-axis accelerometer waveform, sleep/vital records, 6 valid HR samples, and 1 corrupted `310.0 bpm` sample.
- `scripts/run_apple_watch_ingestion_demo.py` — CLI runner executing the sample payload and saving `outputs/apple_watch/synthetic_assessment_result.json`.
- `tests/test_apple_watch_pipeline.py` — 3 unit/integration tests covering Dead-Letter quarantine + non-fabricated missingness, causal workout/future exclusion + zero-MAD fallback, and full NormWear-ACC + XGBoost + TreeSHAP + Stage 3 policy output.

### 3.2 Tests, Benchmarks, Screenshots & MS Word Capstone Report
- `tests/conftest.py` — Adds repository root to `sys.path` so `pytest` imports `src.*` cleanly.
- `outputs/phase7/latency_benchmark.json` — Verified 100-window MPS benchmark output for 768-D Model A.
- `outputs/screenshots/01_single_subject_deep_dive.png` — Live Streamlit UI screenshot (WESAD S10).
- `outputs/screenshots/02_two_person_comparison_s2_vs_s15.png` — Live Streamlit UI screenshot (WESAD S2 vs. S15).
- `outputs/screenshots/03_external_csv_daywise_summary.png` — Live Streamlit UI screenshot (RELAX Participant 57 CSV day-wise summary).
- `outputs/screenshots/04_apple_watch_json_pipeline.png` — Live Streamlit UI screenshot (Apple Watch JSON `AW_SYNTH_001`).
- `reports/figures/` — Staged diagram assets (`fig3_1_system_architecture.png` through `fig3_5_uml_sequence_diagram.png`, plus cropped Alliance logos).
- `scripts/update_capstone_docx.py` -> Generated `reports/ORVYN_Capstone_Project_Report.docx` directly in MS Word OOXML format preserving the user's Alliance University template while reconciling all codebase changes, diagrams, screenshots, equations, tables, and live `{ PAGEREF }` TOC/LOF/LOT links.
- *(Abandoned PDF helper scripts `scripts/build_capstone_report.py` and `scripts/pdf_tools.swift` remain unused on disk after switching to direct `.docx` editing per user instruction.)*

---

## 4. Deep-Dive: Apple Watch / watchOS / iOS JSON Ingestion Pipeline

Built directly from the backend system architecture diagram and the Capstone specification PDF (`Universal_Multi_Brand_Wearable_Physiological_Strain_Recovery_Capstone.pdf`):

### 4.1 Stage-by-Stage Execution Flow (`AppleWatchJSONPipeline.ingest_payload`)

1. **Input Contract (`schema_version: "1.0"`, `adapter_id: "apple_watch_ios_json_v1"`)**
   - Accepts an iOS HealthKit + CoreMotion exported JSON object containing:
     - `device` metadata (`manufacturer`, `model`, `watchos_version`, `ios_companion_version`, `developer_access`)
     - `history_days`: Array of historical daily summaries (`date_utc`, `resting_hr_bpm`, `sdnn_ms`, `sleep_duration_h`, `enmo_mean_g`, `steps`, `active_energy_kcal`, `spo2_pct`, `respiratory_rate_bpm`, `in_workout`, `quality`)
     - `signals`: `continuous_hr_bpm` (timestamped samples), `nightly_sdnn_ms`, `sleep`, `activity`, `optional_vitals`, and `raw_acc` (`cadence_hz`, `x`, `y`, `z` in `g`)

2. **Validation, QC Masks & Dead-Letter Quarantine (`validate_and_qc_apple_payload`)**
   - Enforces physiological bounds from `config/apple_watch_schema.yaml`:
     - `hr_bpm`: `[30.0, 220.0]`
     - `sdnn_ms`: `[5.0, 300.0]`
     - `spo2_pct`: `[70.0, 100.0]`
     - `respiratory_rate_bpm`: `[6.0, 45.0]`
     - `acc_axis_g`: `[-16.0, 16.0]`
   - Non-finite or out-of-bounds samples are **rejected without fabrication** and recorded as structured `DeadLetterRecord` entries (`signal_name`, `reason`, `raw_value`, `timestamp_utc`, `record_index`). Missing blocks (e.g., omitted `raw_acc`) record explicit `missing_reasons` rather than zero-filling.

3. **Stage 1: Causal Personal Baseline + ENMO + Frozen NormWear ACC**
   - **`compute_causal_personal_baseline`**: Filters `history_days` strictly to timestamps earlier than `generated_at_utc`, excluding `in_workout == True` and invalid-quality rows. Computes per-metric `median` and scaled dispersion (`1.4826 * MAD`) with schema `min_mad_fallbacks` when `MAD == 0` (preventing division by zero). Reports `"calibrated"` (`>= 5` prior days) vs. `"provisional"`.
   - **ENMO Kinematics**: From 50 Hz CoreMotion triaxial acceleration `(x, y, z)`, computes `ENMO = max(sqrt(x^2 + y^2 + z^2) - 1, 0)` in `g` (`enmo_mean_g`, `enmo_p95_g`).
   - **Frozen NormWear ViT-Base ACC Encoder**: Resamples 3-axis ACC to NormWear's 384-sample (`64 Hz`, 6-second) contract and computes the L2 norm of the frozen 768-D accelerometer embedding (`normwear_acc_emb_norm`).
   - **Temporal Trend Hook**: Computes causal prior resting-HR slope (`prior_resting_hr_slope_bpm_per_day`) as the active deterministic fallback while `Chronos-Bolt` remains staged (`optional_staged_not_required_for_v1`).

4. **Stage 2: Domain-Calibrated XGBoost (14 Features) + Native TreeSHAP Attribution**
   - Features fed to `models/apple_watch/apple_watch_strain_xgb.json` (preserving native `NaN` for missing optional signals):
     - Autonomic / non-exertional: `hr_mean_bpm`, `hr_baseline_delta_bpm`, `hr_robust_z`
     - Recovery / HRV / Sleep: `sdnn_ms`, `sdnn_robust_z`, `sleep_deficit_h`, `sleep_efficiency_pct`
     - Activity / Kinematics: `enmo_mean_g`, `enmo_p95_g`, `steps_ratio_to_baseline`, `active_energy_ratio_to_baseline`, `normwear_acc_emb_norm`
     - Respiratory / Oxygen: `spo2_delta_pct`, `resp_rate_robust_z`
   - Calls `booster.predict(dmat, pred_contribs=True)` (XGBoost built-in TreeSHAP) to produce ranked feature evidence and category totals across four interpretable categories: `non_exertional_physiological_elevation`, `recovery_context_deviation`, `activity_associated_elevation`, and `respiratory_or_oxygen_deviation` — all explicitly tagged `causal_claim: false`.

5. **Stage 3: Deterministic Policy Engine & Assessment API JSON**
   - Maps active co-occurring states deterministically against `POLICY_CATALOG` (`POL_EXERTION_COOLDOWN`, `POL_NON_EXERTIONAL_PAUSE`, `POL_RECOVERY_REST`, `POL_VITAL_RECHECK`).
   - Keeps `predicted_recovery_minutes: null` (`Disabled until prospective recovery-endpoint validation (PDF Section 11.2)`).
   - Emits a read-only `llm_narration_bundle` for `Gemini Flash-Lite` with `can_modify_scores: false`.

### 4.2 Observed Verification Run (`AW_SYNTH_001` via `scripts/run_apple_watch_ingestion_demo.py`)
- **Output File**: `outputs/apple_watch/synthetic_assessment_result.json`
- **Baseline**: `calibrated` (`9` eligible prior days, workout row excluded)
- **ENMO Mean / P95**: `0.0159 g` / `0.0575 g` (stationary / non-exertional context)
- **Frozen NormWear ACC**: Active (`normwear_acc_emb_norm: 15.4068`)
- **XGBoost Deviation Score**: **`99.8 / 100`** (`elevation_association_prob: 0.9982`)
- **Co-occurring States**: `["non_exertional_physiological_elevation", "recovery_context_deviation", "respiratory_or_oxygen_deviation"]`
- **Top TreeSHAP Contributors**: `hr_baseline_delta_bpm` (`+3.0606` log-odds, `38.8%`), `sdnn_robust_z` (`+1.4239`, `18.1%`), `resp_rate_robust_z` (`+1.0591`, `13.4%`), `hr_robust_z` (`+0.9835`, `12.5%`)
- **Dead-Letter Quarantine**: **`1` rejected record** (`continuous_hr_bpm: 310.0` flagged `out_of_bounds_hr`), `6` valid HR samples retained.

### 4.3 Implemented vs. Staged Component Boundary

| Layer | Implemented Now | Staged Future Scope |
| :--- | :--- | :--- |
| **Wearable / Mobile Capture** | JSON Schema `v1.0` + Synthetic Exporter + Streamlit JSON Uploader | Native `watchOS` + `iOS` Swift HealthKit / CoreMotion companion app |
| **API & Persistence** | `AppleWatchJSONPipeline` Python API + JSON/YAML artifact registry | FastAPI REST service, PostgreSQL / TimescaleDB, S3 / MinIO storage |
| **Models & Encoders** | Frozen NormWear ViT-Base (WESAD + ACC), Model A XGBoost (768-D), Apple Watch XGBoost (14-D), TreeSHAP, Causal Median/MAD | Optional Chronos-Bolt forecasting (currently causal linear trend fallback) |
| **UI & Narration** | 5-mode Streamlit App (`app/dashboard.py`), Stage 3 `POLICY_CATALOG` + read-only `llm_narration_bundle` | Next.js Web Dashboard + live read-only Gemini Flash-Lite API integration |

---

## 5. Verification Summary

- **Automated Test Suite**: `.venv/bin/pytest tests/ -v` -> **`21 passed, 0 failed`** (`3` Apple Watch pipeline tests, `4` baseline deviation tests, `7` CSV validation tests, `7` end-to-end WESAD/Model A pipeline tests).
- **MPS Real-Time Benchmark**: `outputs/phase7/latency_benchmark.json` -> **`112.86 ms` median**, **`115.86 ms` p95**, `8.9 windows/s` (`PASS`).
- **MS Word Capstone Report**: `reports/ORVYN_Capstone_Project_Report.docx` (`2,694,276` bytes) validated across all OOXML parts (`14` embedded images, `88` `{ PAGEREF }` bookmark links, `0` errors).

# Personalized Physiological Strain, Cause & Recovery Engine

**Powered by NormWear ViT Foundation Model**

## 1. Project Scope & Purpose
This capstone project implements a multimodal physiological strain classification engine. It evaluates real-time autonomic strain against personalized baseline profiles to provide calibrated strain predictions, cause attribution, and recovery trajectory forecasts.

## 2. NormWear Foundation Model
The project strictly preserves and utilizes the official `136.1M` NormWear ViT Masked Autoencoder checkpoint. The foundation model weights are completely frozen, and no downstream fine-tuning was performed.

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

## 10. Future Work & Limitations
This system relies on retrospective data. Future research involves real-time edge deployment and broader population validation outside the controlled WESAD environment.

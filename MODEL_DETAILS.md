# Model Architecture, Background Technology & Input/Output Specification

This document provides a comprehensive, end-to-end explanation of the machine learning model architecture, background technologies, input data schemas, and output formatting used in the **Personalized Physiological Strain, Cause & Recovery Engine**.

---

## 1. Executive Summary

| Parameter | Specification |
| :--- | :--- |
| **Model Class** | Frozen NormWear Foundation Classifier + Personal Baseline Strain Engine |
| **Foundation Backbone** | **Vision Transformer (ViT-Base)** Masked Autoencoder (**NormWear**) |
| **Pretrained Parameters** | **136,112,328 (~136.1 Million)** parameters |
| **Backbone Training State** | **Frozen** (zero-shot feature extractor; no weight modification) |
| **Downstream Classifier** | **XGBoost Classifier** (`model_a_embeddings_only.json`, 768-D NormWear-only Experiment A, `binary:logistic`, 100 estimators, max depth 6; outputs `0` Baseline vs. `1` Stress; `0.9400` ROC-AUC / `0.8113` Macro-F1) |
| **Input Signals & Modality Contract** | 3-Channel Wrist Physiological Time-Series (`wrist_BVP`, `wrist_EDA`, `wrist_TEMP`) @ 64 Hz (`overlap_seconds: 3.0`, stride 3.0 s); **RELAX IBI/ACC remains strictly isolated** in its separate daily-aggregation pathway |
| **Input Window** | 6.0-second sliding windows @ 64 Hz $\rightarrow$ **Shape: `[B, 3, 384]`** |
| **Core Outputs** | Strain Index (0–100), Strain State, Stress Probability, Cause Attribution (%), Recovery ETA (mins), and Recovery Curve |

> [!NOTE]
> **Does it use Vision Transformers?**  
> **Yes.** The core foundation model is a **Vision Transformer (ViT)**. Instead of taking raw images, it transforms 1D wearable physiological time-series signals into 2D time-frequency scalograms via Continuous Wavelet Transform (CWT). The 2D spectrograms are then split into patches and processed by a 12-layer Vision Transformer backbone.

---

## 2. Background Technologies & Architecture Deep-Dive

The system employs a four-stage hierarchical architecture:

```
[Raw Time-Series: wrist_BVP, wrist_EDA, wrist_TEMP] ([B, 3, 384] @ 64 Hz, overlap=3.0s)
            │
            ├───► [1. Continuous Wavelet Transform (CWT)]
            │            │ (Converts 1D signals to 2D time-frequency scalograms)
            │            ▼
            │     [2. NormWear ViT-Base MAE (136.1M Params, Frozen)]
            │            │ (Extracts 768-dim deep foundation representation)
            │            ▼
            │     [Deep Representation: 768-dim vector]
            │            │
            │            ├───► [3a. Primary XGBoost Classifier (768-D Experiment A)]
            │            │            └──► Binary Stress Probability (0=Baseline, 1=Stress)
            │            │
            ├───► [3b. Domain Physiological Feature Extraction (20-dim)]
            │            │ (Computes explicit cardiac, EDA & thermal biomarkers; RELAX IBI/ACC isolated)
            │            ▼
            └──────────► [4. Personal Baseline & Recovery Engine (StrainEngine)]
                         │ (Computes Robust z-scores + 5-factor Cause Attribution)
                         ▼
        [Calibrated Strain Index (0–100) + Cause Attribution + Recovery Trajectory]
```

### 2.1. Signal-to-Spectrogram Conversion: Continuous Wavelet Transform (CWT)
Vision Transformers require multi-dimensional grid structures. Rather than projecting 1D time-series directly, NormWear transforms physiological signals into **2D Time-Frequency Scalograms**:
1. For each sensor channel, the original time series $x(t)$, its first discrete difference $\Delta x(t)$ (velocity), and its second discrete difference $\Delta^2 x(t)$ (acceleration) are computed.
2. A Continuous Wavelet Transform (using PyTorch-accelerated Ricker/Mexican-hat wavelets across frequencies from 0.1 Hz to 64 Hz) is applied:
   $$\text{Scalogram}(a, b) = \frac{1}{\sqrt{|a|}} \int_{-\infty}^{\infty} x(t) \psi^*\left(\frac{t - b}{a}\right) dt$$
3. This creates a 3-channel 2D image-like tensor per sensor variable: shape `[batch_size, n_vars, 3, sequence_length, n_frequencies]`.

### 2.2. NormWear Foundation Model (ViT-Base Masked Autoencoder)
The foundation model is based on **NormWear**, pre-trained on massive wearable datasets via Masked Autoencoding (MAE):
- **Model Architecture**: Vision Transformer (ViT-Base) with Cross-Variate CLS token fusion.
- **Total Parameters**: 136,112,328 (~136.1 Million).
- **Patch Embedding**: Patches of size $(9, 5)$ (9 time steps $\times$ 5 frequency scale bins).
- **Transformer Encoder**: 12 Transformer layers, 12 multi-head self-attention heads, hidden dimension $E = 768$, MLP expansion ratio of 4.0.
- **Cross-Variate Fusion**: A cross-channel `[CLS]` token fusion layer is triggered every 2 layers (`fuse_frequency = 2`) to model inter-dependencies between BVP, EDA, and temperature.
- **Positional Encoding**: 2D sinusoidal position embeddings with bicubic spatial interpolation to adapt dynamically to variable window lengths.
- **Extracted Representation**: The `[CLS]` token from each channel is extracted (`[1, 3, 768]`) and mean-pooled across channels to yield a single **768-dimensional latent vector** (`emb_768`).

### 2.3. Domain-Specific Explicit Physiological Features
Deep transformer representations are complemented with 20 engineered physiological features (`src/downstream_features.py`):
1. **Cardiac / BVP (8 features)**: Mean, standard deviation, peak-to-peak amplitude, skewness, kurtosis, estimated heart rate (HR bpm), SDNN (standard deviation of inter-beat intervals), and RMSSD (root mean square of successive differences).
2. **Electrodermal Activity / EDA (6 features)**: Tonic mean, standard deviation, min, max, dynamic range, and linear slope (sympathetic arousal marker).
3. **Skin Temperature / TEMP (3 features)**: Mean, standard deviation, and linear slope (peripheral vasoconstriction marker).
4. **Baseline Deviation (3 features)**: Instantaneous delta from personal resting baseline ($\Delta\text{EDA}$, $\Delta\text{TEMP}$, $\Delta\text{HR}$).

These 20 features are scaled using a fitted `StandardScaler` to yield a standardized 20-dimensional vector.

### 2.4. Downstream Classifier (Primary 768-D XGBoost — Experiment A)
The primary production classifier (`models/downstream/model_a_embeddings_only.json`, **Experiment A**, **0.9400 ROC-AUC / 0.8113 Macro-F1**) takes the frozen **768-dimensional** NormWear ViT representation directly:
$$p_{\text{stress}} = \text{XGBoost}(\mathbf{e}_{\text{ViT}}^{(768)}) \in [0.0, 1.0]$$
predicting binary `0` (Baseline) vs. `1` (Stress). Meanwhile, the 20-D physiological features feed the `StrainEngine` for personal-baseline Z-scores and 5-factor cause attribution (while the RELAX IBI/ACC longitudinal pipeline remains strictly isolated). *(Legacy 788-D fused artifact `stress_response_xgb.json` is retained untouched on disk for audit compatibility.)*

#### What does XGBoost do?
**XGBoost (eXtreme Gradient Boosting)** is an ensemble machine learning algorithm based on Gradient Boosted Decision Trees (GBDT). In this architecture, it acts as the **supervised decision-making head**:
1. **Ensemble of Decision Trees**: Instead of relying on a single model, XGBoost builds a sequence of shallow decision trees ($M = 100$ to $150$ estimators, `max_depth = 4` to $6$).
2. **Iterative Error Correction**: Each successive tree is trained to predict the residual gradients (errors) made by the previous trees.
3. **Thresholding & Non-Linear Mapping**: It evaluates decision rules on individual features (e.g., *"Is skin conductance slope $> 0.15$ AND is embedding dimension 42 $< -0.21$?"*).
4. **Probability Output**: The scores across all trees are summed and mapped through a logistic sigmoid function:
   $$p_{\text{stress}} = \frac{1}{1 + e^{-\sum_{m=1}^{M} f_m(\mathbf{z}_{\text{fused}})}}$$
   yielding a calibrated probability between $0.0$ (relaxed/baseline) and $1.0$ (acute stress).

#### Why is XGBoost used here?
1. **Clean Latent Separability & Zero-Shot Generalization**:
   - In subject-independent validation (`0.9400` ROC-AUC / `0.8113` Macro-F1), the **768-D NormWear-only Experiment A** head outperforms both physiological-only features (`0.9163` ROC-AUC / `0.7859` Macro-F1) and concatenated 788-D fusion (`0.9355` ROC-AUC / `0.8104` Macro-F1). Keeping the 20-D biomarkers dedicated to `StrainEngine` personal-baseline Z-scores and 5-factor cause attribution preserves clean architectural separation without feature collinearity.
2. **Prevents Overfitting on Small Cohorts**:
   - Physiological benchmark datasets (such as WESAD, with 15 subjects) have limited subject diversity. Fine-tuning the 136.1M parameter Vision Transformer directly would lead to severe overfitting and catastrophic forgetting.
   - Freezing the ViT backbone and training an XGBoost downstream classifier guarantees strong generalization across unseen subjects.
3. **Captures Non-Linear Physiological Interactions**:
   - Stress responses are non-linear (e.g., an elevated heart rate alone might just be physical movement, but an elevated heart rate paired with an EDA spike and skin temperature drop indicates sympathetic fight-or-flight activation). Decision trees naturally identify these multi-condition threshold combinations without requiring manual feature cross-products.
4. **Sub-Millisecond Classifier Inference**:
   - Tree traversal is computationally trivial compared to neural network forward passes. XGBoost evaluates the 768-dimensional vector in **0.22 ms median**, enabling real-time edge telemetry with zero GPU/Metal thread contention (`n_jobs=1`).
5. **Effective Class Imbalance Handling**:
   - Physiological baseline periods far exceed stress episodes. XGBoost natively integrates training-only class weighting ($w_0 = 0.7773, w_1 = 1.4015$), preventing majority-class bias and maximizing balanced accuracy and ROC-AUC.


### 2.5. Personal Baseline & Autonomic Recovery Engine
Generic models fail when evaluating physiological stress because resting heart rate or skin conductance varies significantly between individuals. The system calibrates predictions using the individual's resting profile:
- **Profile Parameters**: Baseline Median and Median Absolute Deviation (MAD) for HR, RMSSD, EDA, and Temperature, plus the 768-dim baseline embedding centroid.
- **Robust Z-Score**:
  $$z = 0.6745 \times \frac{x - \text{Median}}{\text{MAD}}$$
  *(Vagal withdrawal is computed by reversing RMSSD deviation; vasoconstriction by reversing skin temperature deviation).*
- **Composite Physiological Deviation**:
  $$z_{\text{composite}} = 0.45 \cdot z_{\text{sympathetic}} + 0.35 \cdot z_{\text{cardio}} + 0.10 \cdot z_{\text{thermal}} + 0.10 \cdot z_{\text{embedding\_distance}}$$
- **Personalized Strain Score (0–100)**: Blends the machine learning probability ($p_{\text{stress}}$) with the sigmoidal deviation probability:
  $$\text{Strain Index} = \text{Clip}\left(\left(0.60 \cdot p_{\text{stress}} + 0.40 \cdot \sigma(1.5 \cdot (z_{\text{composite}} - 1.0))\right) \times 100, \, 0, \, 100\right)$$
- **Cause Attribution**: Quantifies percentage contribution across:
  - Sympathetic Arousal (EDA / Skin Conductance)
  - Cardiovascular Strain (Heart Rate Surge)
  - Parasympathetic Withdrawal (Low RMSSD / Vagal tone)
  - Peripheral Vasomotor Reflex (Skin Temperature Drop)
  - NormWear ViT Morphological Drift
- **Recovery Trajectory Forecasting**:
  Projects recovery ETA using individual baseline vagal reserve ($\text{RMSSD}_{\text{baseline}} / 40.0$) and models exponential decay back to homeostatic baseline over a 15-minute horizon.

---

## 3. Input Specification & Parameters

### 3.1. Required Input Channels

| Channel Name | Description | Source Sensor | Unit | Sampling Rate |
| :--- | :--- | :--- | :--- | :--- |
| `wrist_BVP` | Blood Volume Pulse | Photoplethysmography (PPG) | Unitless / ADC | 64 Hz |
| `wrist_EDA` | Electrodermal Activity | Galvanic Skin Conductance | $\mu\text{S}$ (microSiemens) | 64 Hz |
| `wrist_TEMP`| Skin Temperature | Skin thermistor | $^\circ\text{C}$ (Celsius) | 64 Hz |

*(Optional channels supported by the schema: `wrist_ACC_x`, `wrist_ACC_y`, `wrist_ACC_z` accelerometry).*

### 3.2. Tensor Input Format (For Real-Time / Batch Inference)
- **Data Type**: `torch.float32` or `numpy.ndarray` (`float32`).
- **Input Shape**: `[batch_size, 3, 384]` (or unbatched `[3, 384]`).
  - Dimension 0: Batch size (e.g., 1 for single window).
  - Dimension 1: Channel index (`0: wrist_BVP`, `1: wrist_EDA`, `2: wrist_TEMP`).
  - Dimension 2: Time samples ($6.0\text{ seconds} \times 64\text{ Hz} = 384\text{ samples}$).

### 3.3. CSV File Input Format (For Multi-Day / External Uploads)
When uploading external data or reading time-series files, the CSV must adhere to the following schema:
- **Columns**:
  - `timestamp`: Epoch timestamp (in seconds or milliseconds) or ISO 8601 string.
  - `wrist_BVP`: Float values.
  - `wrist_EDA`: Float values ($\ge 0.0$).
  - `wrist_TEMP`: Float values (typically $25.0^\circ\text{C}$ to $38.0^\circ\text{C}$).
- **Sampling Frequency**: Must be sampled or resampled to 64 Hz.

---

## 4. Output Specification & Data Structure

When calling `engine.process_window(subject_id, window)`, the engine produces a rich dictionary containing comprehensive strain telemetry:

### 4.1. Real-Time Telemetry JSON Schema

```json
{
  "subject_id": "S2",
  "strain_score": 78.4,
  "strain_state": "Acute Physiological Surge",
  "status_color": "#dc3545",
  "ml_stress_prob": 0.8421,
  "composite_deviation_z": 2.31,
  "primary_cause": "Sympathetic Arousal (EDA/Skin Conductance)",
  "cause_attribution": {
    "Sympathetic Arousal (EDA/Skin Conductance)": 44.2,
    "Cardiovascular Strain (Heart Rate Surge)": 28.1,
    "Parasympathetic Withdrawal (Low HRV)": 15.3,
    "Peripheral Vasomotor Reflex (Skin Temp)": 7.8,
    "NormWear ViT Morphological Drift": 4.6
  },
  "projected_recovery_time_mins": 8.5,
  "projected_recovery_curve": {
    "minutes": [0.0, 2.0, 5.0, 8.0, 12.0, 15.0],
    "projected_strain": [78.4, 43.5, 18.2, 11.4, 10.0, 10.0]
  },
  "metrics": {
    "hr_bpm": 86.4,
    "hr_delta_bpm": 18.2,
    "rmssd_ms": 19.4,
    "eda_scl": 4.821,
    "eda_delta_scl": 3.105,
    "scr_rate_per_min": 7.5,
    "temp_c": 31.42,
    "normwear_emb_distance": 1.482
  },
  "timing_ms": {
    "foundation_vit_forward": 14.8,
    "physio_feature_extraction": 1.2,
    "downstream_xgb_inference": 0.4,
    "personal_baseline_deviation": 0.3,
    "total_end_to_end": 16.7
  }
}
```

### 4.2. Field Descriptions

| Output Field | Type | Description |
| :--- | :--- | :--- |
| `strain_score` | `float` | Continuous calibrated strain metric from 0.0 to 100.0. |
| `strain_state` | `string` | Categorical state: `"Homeostatic Rest"`, `"Mild Activation"`, `"Moderate Autonomic Strain"`, or `"Acute Physiological Surge"`. |
| `status_color` | `string` | UI color representation (Green `#28a745`, Teal `#17a2b8`, Amber `#ffc107`, Red `#dc3545`). |
| `ml_stress_prob` | `float` | Raw probability output from XGBoost ($p_{\text{stress}} \in [0, 1]$). |
| `composite_deviation_z` | `float` | Multi-signal deviation z-score relative to personal resting baseline. |
| `primary_cause` | `string` | The dominant physiological driver responsible for the current strain spike. |
| `cause_attribution` | `dict` | Percentage share breakdown across 5 physiological causal drivers (sums to 100%). |
| `projected_recovery_time_mins` | `float` | Estimated time (in minutes) required to return to homeostatic baseline. |
| `projected_recovery_curve` | `dict` | Time points (minutes) and predicted decay values tracking recovery. |
| `metrics` | `dict` | Explicit measured vitals, baseline deltas ($\Delta\text{HR}$, $\Delta\text{EDA}$), and embedding distance. |
| `timing_ms` | `dict` | Latency profiling per pipeline stage (foundation ViT forward pass, feature extraction, XGBoost, end-to-end). |

---

## 5. Hardware Acceleration & Operational Requirements

- **Supported Acceleration**: 
  - **Apple Silicon MPS (`torch.device("mps")`)**: Fully supported with native Metal acceleration.
  - **NVIDIA CUDA (`torch.device("cuda")`)**: Supported for high-throughput batch inference.
  - **CPU fallback (`torch.device("cpu")`)**: Supported with single-thread OpenMP isolation (`OMP_NUM_THREADS=1`).
- **Memory Footprint**:
  - Model weights on disk: ~545 MB.
  - Active runtime memory: ~1.2 GB RAM / Unified Memory.
- **Inference Latency (100-window benchmark, Apple Silicon `mps`, `outputs/phase7/latency_benchmark.json`)**:
  - End-to-end median per 6.0-second window: **112.86 ms** (`111.86 ms` NormWear ViT forward, `0.69 ms` 20-D physiological features, `0.22 ms` 768-D XGBoost inference, `0.03 ms` StrainEngine deviation).
  - End-to-end p95 latency: **115.86 ms** (`8.9 windows/second`, `1.88%` of the 6,000 ms real-time window budget).

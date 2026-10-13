# Phase 3 Report — Real WESAD Dataset Inspection & Validation

**Date**: 2026-10-07  
**Status**: COMPLETE (PASS)  
**Execution**: Executed natively on local environment with zero synthetic/fabricated data.

---

## 1. Dataset Source & Provenance
- **Dataset Name**: WESAD (Wearable Stress and Affect Detection)
- **Primary Repository Page**: [UCI Machine Learning Repository #465](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)
- **Author Institutional Page**: [University of Siegen Ubiquitous Computing](https://www.eti.uni-siegen.de/ubicomp/home/datasets/icmi18/index.html.en?lang=en)
- **Author Portal Redirect**: [ubi29.informatik.uni-siegen.de/usi/data_wesad.html](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)
- **Raw Download URL Actually Used**: `https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gx/download`
- **Total Raw Size**: `17,637,138,437` bytes (~16.43 GiB)
- **Reference Readme SHA-256**: `167f2f6388c029f4cf669e5dd54089854ce355c55b149520da9266fea65d809b`
- **Extraction Path**: `data/raw/WESAD/`
- **Citation**: Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger, and Kristof Van Laerhoven. "Introducing WESAD, a multimodal dataset for Wearable Stress and Affect Detection", ICMI 2018.

---

## 2. Dataset Structure & Subject Inventory
- **Discovered Subjects**: 15 subjects (S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S13, S14, S15, S16, S17)
- **Successfully Loaded**: 15 / 15 (100% load success rate)
- **File Format**: Synchronized Python dictionaries stored in `.pkl` format (deserialized via `pickle.load(f, encoding="latin1")`).
- **Device Groups**: Two physical devices per subject:
  1. `chest`: RespiBAN sensor unit (700 Hz)
  2. `wrist`: Empatica E4 smartwatch (multimodal: 4, 32, 64 Hz)
- **Subject Inventory File**: [`outputs/phase3/subject_inventory.csv`](subject_inventory.csv)

---

## 3. Signal Modalities & Sampling Rates

Verified directly against local array lengths and official device specifications:

| Device | Modality | Dimension | Official Nominal Rate | Verified Array Length Ratio | Status |
|:---|:---|:---:|:---:|:---:|:---:|
| **Wrist (Empatica E4)** | **BVP** (Blood Volume Pulse) | 1D | **64 Hz** | 1.0000x | VERIFIED |
| Wrist (Empatica E4) | **EDA** (Electrodermal Activity) | 1D | **4 Hz** | 0.0625x (1/16th of BVP) | VERIFIED |
| Wrist (Empatica E4) | **TEMP** (Skin Temperature) | 1D | **4 Hz** | 0.0625x (1/16th of BVP) | VERIFIED |
| Wrist (Empatica E4) | **ACC** (Triaxial Accelerometer) | 3D (`[N, 3]`) | **32 Hz** | 0.5000x (1/2 of BVP) | VERIFIED |
| **Chest (RespiBAN)** | **ECG** (Electrocardiogram) | 1D | **700 Hz** | 10.9375x (700/64) | VERIFIED |
| Chest (RespiBAN) | **EDA** (Electrodermal Activity) | 1D | **700 Hz** | 10.9375x | VERIFIED |
| Chest (RespiBAN) | **EMG** (Electromyogram) | 1D | **700 Hz** | 10.9375x | VERIFIED |
| Chest (RespiBAN) | **RESP** (Respiration) | 1D | **700 Hz** | 10.9375x | VERIFIED |
| Chest (RespiBAN) | **TEMP** (Temperature) | 1D | **700 Hz** | 10.9375x | VERIFIED |
| Chest (RespiBAN) | **ACC** (Triaxial Accelerometer) | 3D (`[N, 3]`) | **700 Hz** | 10.9375x | VERIFIED |
| **Label Stream** | Ground Truth Condition | 1D | **700 Hz** | Aligned with Chest Stream | VERIFIED |

- **Channel Inventory File**: [`outputs/phase3/channel_inventory.csv`](channel_inventory.csv)
- **Sampling Audit File**: [`outputs/phase3/sampling_audit.csv`](sampling_audit.csv)
- **Length Consistency File**: [`outputs/phase3/channel_length_consistency.csv`](channel_length_consistency.csv)

---

## 4. Ground Truth Labels & Session Distribution

All label streams are synchronously sampled at 700 Hz alongside the chest recording.

### Aggregated Distribution:
|   label_code | label_name            |   sample_count |   overall_pct |   total_hours |
|-------------:|:----------------------|---------------:|--------------:|--------------:|
|            0 | undefined / transient |       27654897 |     45.4793   |     10.9742   |
|            1 | baseline              |       12327702 |     20.2733   |      4.89195  |
|            2 | stress                |        6976201 |     11.4726   |      2.76833  |
|            3 | amusement             |        3902501 |      6.41778  |      1.54861  |
|            4 | meditation            |        8264199 |     13.5907   |      3.27944  |
|            5 | ignore / rest         |         552300 |      0.908275 |      0.219167 |
|            6 | ignore / rest         |         552998 |      0.909423 |      0.219444 |
|            7 | ignore / rest         |         576802 |      0.948569 |      0.22889  |

- **Baseline (`Code 1`)**: Neutral resting baseline (~20 min per subject reading magazines).
- **Stress (`Code 2`)**: Trier Social Stress Test (TSST: public speaking preparation, speech, mental arithmetic).
- **Amusement (`Code 3`)**: Funny video sequences.
- **Meditation (`Code 4`)**: Guided breathing / meditation exercise.
- **Undefined (`Code 0`) & Rest (`Codes 5-7`)**: Inter-task rest, equipment transitions, questionnaire filling.

- **Label Inventory File**: [`outputs/phase3/label_inventory.csv`](label_inventory.csv)
- **Label Mapping Document**: [`outputs/phase3/label_mapping.md`](label_mapping.md)

---

## 5. Raw Signal Quality Audit
Audited across all 15 subjects and all channels:
- **Total NaN Values**: `0`
- **Total Inf Values**: `0`
- **Finite Fraction**: `1.000000`
- **Corrupt Subjects / Files**: `0`
- **Quality Audit File**: [`outputs/phase3/raw_signal_quality.csv`](raw_signal_quality.csv)

---

## 6. NormWear Foundation Model Compatibility Matrix & Proposed Input Contract

Verified against the frozen Phase 2 NormWear interface:
- **NormWear Input Tensor**: `[batch_size, n_channels, sequence_length]`
- **Data Type**: `torch.float32`
- **Native Frequency**: 64 Hz

### Recommended Primary Candidate Configuration for Phase 4:
To build a realistic consumer wearable system, the downstream architecture focuses on **wrist-worn Empatica E4 signals**:
1. **Channel 0**: `wrist_BVP` (PPG) — Native 64 Hz. Matches NormWear pretraining domain.
2. **Channel 1**: `wrist_EDA` (Electrodermal Activity) — 4 Hz $ightarrow$ resampled to 64 Hz via polyphase interpolation.
3. **Channel 2**: `wrist_TEMP` (Skin Temperature) — 4 Hz $ightarrow$ resampled to 64 Hz via linear interpolation.

Optional Motion-Aware 6-Channel Variant:
- **Channels 3-5**: `wrist_ACC_x`, `wrist_ACC_y`, `wrist_ACC_z` (32 Hz $ightarrow$ resampled to 64 Hz).

### Proposed Fixed Channel Order:
`["wrist_BVP", "wrist_EDA", "wrist_TEMP"]`  
*(Preserved strictly across all subjects and train/val/test splits to maintain token embedding consistency).*

- **Compatibility Matrix File**: [`outputs/phase3/normwear_wesad_compatibility.csv`](normwear_wesad_compatibility.csv)

---

## 7. Diagnostic Visual Inspection
Diagnostic plots generated from real physiological recordings of subject `S2`:
- **BVP / PPG**: [`outputs/phase3/plots/S2_bvp_diagnostic.png`](plots/S2_bvp_diagnostic.png)
- **EDA**: [`outputs/phase3/plots/S2_eda_diagnostic.png`](plots/S2_eda_diagnostic.png)
- **Skin Temperature**: [`outputs/phase3/plots/S2_temp_diagnostic.png`](plots/S2_temp_diagnostic.png)
- **3-Axis Acceleration**: [`outputs/phase3/plots/S2_acc_diagnostic.png`](plots/S2_acc_diagnostic.png)
- **Condition Timeline**: [`outputs/phase3/plots/S2_label_timeline.png`](plots/S2_label_timeline.png)

---

## 8. Non-Negotiable Engineering Rules Compliance Audit
- [x] Real official NormWear checkpoint preserved & frozen.
- [x] Real WESAD dataset downloaded & validated.
- [x] Zero synthetic or fabricated physiological values.
- [x] Zero synthetic or fabricated labels.
- [x] Zero downstream training performed (Phase 3 stop condition obeyed).
- [x] Latin1 pickle decoding verified.
- [x] Provenance recorded in `data/raw/DATA_SOURCE.md`.

**Phase 3 Acceptance Gate**: **PASS**

# Phase 17 Artifact Contracts Report

| Artifact | Format | Creator Script | Input | Output | Version |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NormWear Checkpoint** | `.pth` | Pre-trained (Official) | N/A | `[1, 768]` embeddings | v1.0 |
| **Windows Array** | `.npy` | `src/data_processing.py` | Raw CSV | `[N, 3, 384]` array | v1.1 |
| **Windows Metadata** | `.parquet` | `src/data_processing.py` | Raw CSV | Tabular metadata | v1.1 |
| **Embeddings** | `.npy` | `src/embedding_extraction.py` | Windows Array | `[N, 768]` embeddings | v1.0 |
| **XGBoost Classifier** | `.json` | `src/train_downstream.py` | Embeddings | Probabilities | v1.0 |
| **Baseline Profiles** | `.json` | `src/personal_baseline_engine.py`| Metadata/Embeddings | Baseline Stats | v1.2 |

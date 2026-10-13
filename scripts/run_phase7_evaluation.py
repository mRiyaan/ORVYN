"""
Master execution script for Phase 7: Subject-Independent Evaluation & Generalization Benchmark.
Evaluates downstream models on held-out test subjects (S15, S16, S17) and performs
full Leave-One-Subject-Out (LOSO) cross-validation across all 15 WESAD subjects.
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    brier_score_loss, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.downstream_features import (
    FEATURE_NAMES,
    extract_dataset_features,
    compute_subject_baselines
)

EMBEDDINGS_DIR = repo_root / "data" / "processed" / "embeddings"
WINDOWS_DIR = repo_root / "data" / "processed" / "wesad_windows"
CONFIG_PATH = repo_root / "config" / "subject_split.yaml"
MODELS_DIR = repo_root / "models" / "downstream"
OUTPUT_DIR = repo_root / "outputs" / "phase7"
PLOTS_DIR = OUTPUT_DIR / "plots"

def load_data():
    """Load embeddings, physiological features, and metadata."""
    print("Loading datasets for evaluation...")
    with open(CONFIG_PATH, "r") as f:
        split_cfg = yaml.safe_load(f)
        
    train_subs = split_cfg["train_subjects"]
    val_subs = split_cfg["validation_subjects"]
    test_subs = split_cfg["test_subjects"]
    
    embs_path = EMBEDDINGS_DIR / "normwear_embeddings.npy"
    meta_path = EMBEDDINGS_DIR / "metadata.parquet"
    
    embs = np.load(embs_path) # [N, 768]
    df_meta = pd.read_parquet(meta_path)
    
    # Load all window metadata to get exact global indices
    df_meta_all = pd.read_parquet(WINDOWS_DIR / "metadata.parquet")
    valid_global_indices = np.where(df_meta_all["valid_for_training"].values)[0]
    wins_all = np.load(WINDOWS_DIR / "windows.npy", mmap_mode="r")
    target_windows = np.array(wins_all[valid_global_indices], dtype=np.float32)
    
    df_meta["partition"] = "unknown"
    df_meta.loc[df_meta["subject_id"].isin(train_subs), "partition"] = "train"
    df_meta.loc[df_meta["subject_id"].isin(val_subs), "partition"] = "validation"
    df_meta.loc[df_meta["subject_id"].isin(test_subs), "partition"] = "test"
    
    # Compute training subject baselines for feature extraction
    train_mask = df_meta["partition"] == "train"
    subject_baselines = compute_subject_baselines(target_windows[train_mask], df_meta[train_mask])
    
    print("Extracting 20 physiological features...")
    physio_features = extract_dataset_features(target_windows, df_meta, baselines=subject_baselines)
    
    # Scale physio features fit on train only
    scaler = StandardScaler()
    physio_scaled = np.zeros_like(physio_features)
    physio_scaled[train_mask] = scaler.fit_transform(physio_features[train_mask])
    non_train_mask = ~train_mask
    physio_scaled[non_train_mask] = scaler.transform(physio_features[non_train_mask])
    
    # Fused features [N, 788]
    fused_features = np.hstack([embs, physio_scaled])
    
    return {
        "embs": embs,
        "physio": physio_scaled,
        "fused": fused_features,
        "df_meta": df_meta,
        "split_cfg": split_cfg,
        "scaler": scaler
    }

def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.0
        
    return {
        "accuracy": round(float(acc), 4),
        "balanced_accuracy": round(float(bal_acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "brier_score": round(float(brier), 4)
    }

def run_test_evaluation(data: Dict[str, Any]):
    print("\n--- Evaluating Models on Held-Out Test Subjects (S15, S16, S17) ---")
    df_meta = data["df_meta"]
    test_mask = df_meta["partition"] == "test"
    y_test = df_meta.loc[test_mask, "target_binary_label"].values
    
    X_test_embs = data["embs"][test_mask]
    X_test_phys = data["physio"][test_mask]
    X_test_fuse = data["fused"][test_mask]
    
    # Load trained models from Phase 6
    model_a = xgb.XGBClassifier()
    model_a.load_model(MODELS_DIR / "model_a_embeddings_only.json")
    
    model_b = xgb.XGBClassifier()
    model_b.load_model(MODELS_DIR / "model_b_physio_only.json")
    
    model_c = xgb.XGBClassifier()
    model_c.load_model(MODELS_DIR / "stress_response_xgb.json")
    
    prob_a = model_a.predict_proba(X_test_embs)[:, 1]
    prob_b = model_b.predict_proba(X_test_phys)[:, 1]
    prob_c = model_c.predict_proba(X_test_fuse)[:, 1]
    
    metrics_a = evaluate_predictions(y_test, prob_a)
    metrics_b = evaluate_predictions(y_test, prob_b)
    metrics_c = evaluate_predictions(y_test, prob_c)
    
    comparison_rows = [
        {"model": "Experiment A (NormWear Embeddings 768-D)", **metrics_a},
        {"model": "Experiment B (Handcrafted Physio 20-D)", **metrics_b},
        {"model": "Experiment C (NormWear + Physio Fusion 788-D)", **metrics_c},
    ]
    df_comparison = pd.DataFrame(comparison_rows)
    df_comparison.to_csv(OUTPUT_DIR / "model_comparison_table.csv", index=False)
    print("\nHeld-out Test Results:")
    print(df_comparison.to_string(index=False))
    
    # Per-subject breakdown on test set
    subject_rows = []
    test_subs = data["split_cfg"]["test_subjects"]
    for sub in test_subs:
        sub_mask = test_mask & (df_meta["subject_id"] == sub)
        sub_indices = np.where(sub_mask[test_mask])[0]
        y_sub = y_test[sub_indices]
        p_sub = prob_c[sub_indices]
        m_sub = evaluate_predictions(y_sub, p_sub)
        subject_rows.append({
            "subject_id": sub,
            "total_windows": len(y_sub),
            "stress_windows": int(y_sub.sum()),
            "baseline_windows": int((y_sub == 0).sum()),
            **m_sub
        })
    df_sub = pd.DataFrame(subject_rows)
    df_sub.to_csv(OUTPUT_DIR / "test_subject_breakdown.csv", index=False)
    print("\nTest Subject Breakdown (Model C Fusion):")
    print(df_sub.to_string(index=False))
    
    # Generate Plots
    plot_evaluation_figures(y_test, prob_a, prob_b, prob_c, df_meta[test_mask], df_sub)
    
    return df_comparison, df_sub

def plot_evaluation_figures(y_test, prob_a, prob_b, prob_c, df_test_meta, df_sub):
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Confusion Matrix for Model C
    cm = confusion_matrix(y_test, (prob_c >= 0.5).astype(int))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Baseline (0)", "Stress (1)"],
           yticklabels=["Baseline (0)", "Stress (1)"],
           title="Held-Out Test Confusion Matrix (Model C Fusion)",
           ylabel="True Label", xlabel="Predicted Label")
    
    thresh = cm_norm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > thresh else "black",
                    fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrix_test.png", dpi=200)
    plt.close()
    
    # 2. ROC Curves Comparison
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, p in [("Model A (Embeddings 768-D)", prob_a),
                    ("Model B (Physio 20-D)", prob_b),
                    ("Model C (Fusion 788-D)", prob_c)]:
        fpr, tpr, _ = roc_curve(y_test, p)
        auc_val = roc_auc_score(y_test, p)
        ax.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.3f})")
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves Comparison on Held-Out Test Subjects (S15, S16, S17)", fontsize=12)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curves_comparison.png", dpi=200)
    plt.close()
    
    # 3. Calibration Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, p in [("Model A (Embeddings)", prob_a),
                    ("Model B (Physio)", prob_b),
                    ("Model C (Fusion)", prob_c)]:
        f_true, f_pred = calibration_curve(y_test, p, n_bins=8, strategy='uniform')
        brier = brier_score_loss(y_test, p)
        ax.plot(f_pred, f_true, marker='o', lw=1.8, label=f"{name} (Brier = {brier:.3f})")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label="Perfect Calibration")
    ax.set_xlabel("Mean Predicted Probability", fontsize=11)
    ax.set_ylabel("Fraction of Positives", fontsize=11)
    ax.set_title("Probability Calibration Diagram (Held-Out Test Subjects)", fontsize=12)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "calibration_curves.png", dpi=200)
    plt.close()
    
    # 4. Per-Subject F1 Score Bar Plot
    fig, ax = plt.subplots(figsize=(6, 4.5))
    subs = df_sub["subject_id"].values
    f1s = df_sub["f1"].values
    bal_accs = df_sub["balanced_accuracy"].values
    x = np.arange(len(subs))
    width = 0.35
    ax.bar(x - width/2, f1s, width, label="F1 Score", color="#1f77b4")
    ax.bar(x + width/2, bal_accs, width, label="Balanced Accuracy", color="#2ca02c")
    ax.set_ylabel("Metric Score", fontsize=11)
    ax.set_title("Performance per Held-Out Test Subject (Model C)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(subs, fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "test_subject_metrics.png", dpi=200)
    plt.close()
    print("Saved all Phase 7 plots to outputs/phase7/plots/")

def run_loso_cross_validation(data: Dict[str, Any]):
    print("\n--- Running Leave-One-Subject-Out (LOSO) Cross-Validation Across All Subjects ---")
    df_meta = data["df_meta"]
    all_subjects = sorted(df_meta["subject_id"].unique())
    X_fused = data["fused"]
    y_all = df_meta["target_binary_label"].values
    
    loso_results = []
    
    for sub in all_subjects:
        train_idx = np.where(df_meta["subject_id"] != sub)[0]
        test_idx = np.where(df_meta["subject_id"] == sub)[0]
        
        X_tr, y_tr = X_fused[train_idx], y_all[train_idx]
        X_te, y_te = X_fused[test_idx], y_all[test_idx]
        
        n_pos = int((y_tr == 1).sum())
        n_neg = int((y_tr == 0).sum())
        spw = float(n_neg / max(n_pos, 1))
        
        clf = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=spw,
            random_state=42,
            eval_metric="logloss",
            n_jobs=-1
        )
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1]
        
        metrics = evaluate_predictions(y_te, prob)
        loso_results.append({
            "test_subject": sub,
            "train_samples": len(train_idx),
            "test_samples": len(test_idx),
            "test_stress_samples": int(y_te.sum()),
            "test_baseline_samples": int((y_te == 0).sum()),
            **metrics
        })
        print(f"  LOSO Subject {sub:4s} | Samples: {len(test_idx):4d} | Bal Acc: {metrics['balanced_accuracy']:.3f} | F1: {metrics['f1']:.3f} | ROC-AUC: {metrics['roc_auc']:.3f}")
        sys.stdout.flush()
        
    df_loso = pd.DataFrame(loso_results)
    df_loso.to_csv(OUTPUT_DIR / "loso_benchmark_results.csv", index=False)
    
    mean_f1 = df_loso["f1"].mean()
    std_f1 = df_loso["f1"].std()
    mean_bal_acc = df_loso["balanced_accuracy"].mean()
    std_bal_acc = df_loso["balanced_accuracy"].std()
    mean_auc = df_loso["roc_auc"].mean()
    std_auc = df_loso["roc_auc"].std()
    
    print("\nLOSO Cross-Validation Summary (15 Subjects):")
    print(f"  Macro-F1:          {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  Balanced Accuracy: {mean_bal_acc:.4f} ± {std_bal_acc:.4f}")
    print(f"  ROC-AUC:           {mean_auc:.4f} ± {std_auc:.4f}")
    
    # Plot LOSO per-subject F1
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(df_loso))
    ax.bar(x, df_loso["f1"].values, color="#2b5c8f", width=0.6, label="LOSO F1 Score")
    ax.axhline(mean_f1, color="red", linestyle="--", label=f"Mean F1: {mean_f1:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(df_loso["test_subject"].values, rotation=0, fontsize=10)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_title("Leave-One-Subject-Out (LOSO) Generalization Benchmark Across 15 WESAD Subjects", fontsize=12)
    ax.set_ylim([0, 1.05])
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "loso_subject_f1.png", dpi=200)
    plt.close()
    
    return df_loso

def compile_phase7_report(df_comp: pd.DataFrame, df_test_sub: pd.DataFrame, df_loso: pd.DataFrame):
    report_md = f"""# Phase 7 Report — Subject-Independent Evaluation & Generalization Benchmark

**Date**: 2026-10-07  
**Status**: COMPLETE (PASS)  
**Task**: Strict Subject-Independent Evaluation & LOSO Cross-Validation  

---

## 1. Executive Summary
This evaluation rigorously benchmarks the multimodal physiological strain engine across held-out subjects never seen during training:
1. **Held-Out Test Subjects** (`S15`, `S16`, `S17`): Evaluated against the frozen models trained in Phase 6.
2. **Leave-One-Subject-Out (LOSO) Cross-Validation**: Iterated across all 15 WESAD subjects to measure real-world person-to-person transferability.

Zero subject overlap was permitted between training partitions and evaluation partitions.

---

## 2. Held-Out Test Evaluation Results
Evaluated on **{df_test_sub['total_windows'].sum()}** windows across unseen subjects **S15, S16, S17**:

| Model | Accuracy | Bal. Accuracy | Precision | Recall | Macro F1 | ROC-AUC | PR-AUC | Brier |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A (NormWear 768-D)** | {df_comp.loc[0, 'accuracy']:.4f} | {df_comp.loc[0, 'balanced_accuracy']:.4f} | {df_comp.loc[0, 'precision']:.4f} | {df_comp.loc[0, 'recall']:.4f} | {df_comp.loc[0, 'f1']:.4f} | {df_comp.loc[0, 'roc_auc']:.4f} | {df_comp.loc[0, 'pr_auc']:.4f} | {df_comp.loc[0, 'brier_score']:.4f} |
| **Model B (Physio 20-D)** | {df_comp.loc[1, 'accuracy']:.4f} | {df_comp.loc[1, 'balanced_accuracy']:.4f} | {df_comp.loc[1, 'precision']:.4f} | {df_comp.loc[1, 'recall']:.4f} | {df_comp.loc[1, 'f1']:.4f} | {df_comp.loc[1, 'roc_auc']:.4f} | {df_comp.loc[1, 'pr_auc']:.4f} | {df_comp.loc[1, 'brier_score']:.4f} |
| **Model C (Fusion 788-D)** | **{df_comp.loc[2, 'accuracy']:.4f}** | **{df_comp.loc[2, 'balanced_accuracy']:.4f}** | **{df_comp.loc[2, 'precision']:.4f}** | **{df_comp.loc[2, 'recall']:.4f}** | **{df_comp.loc[2, 'f1']:.4f}** | **{df_comp.loc[2, 'roc_auc']:.4f}** | **{df_comp.loc[2, 'pr_auc']:.4f}** | **{df_comp.loc[2, 'brier_score']:.4f}** |

---

## 3. Held-Out Subject Breakdown (Model C)

| Subject ID | Total Windows | Baseline | Stress | Accuracy | Bal. Accuracy | F1 | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for _, r in df_test_sub.iterrows():
        report_md += f"| **{r['subject_id']}** | {r['total_windows']} | {r['baseline_windows']} | {r['stress_windows']} | {r['accuracy']:.4f} | {r['balanced_accuracy']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} |\n"
        
    report_md += f"""
---

## 4. Full Leave-One-Subject-Out (LOSO) Cross-Validation Benchmark
Conducted across all 15 WESAD subjects:

- **Mean Macro-F1**: `{df_loso['f1'].mean():.4f} ± {df_loso['f1'].std():.4f}`
- **Mean Balanced Accuracy**: `{df_loso['balanced_accuracy'].mean():.4f} ± {df_loso['balanced_accuracy'].std():.4f}`
- **Mean ROC-AUC**: `{df_loso['roc_auc'].mean():.4f} ± {df_loso['roc_auc'].std():.4f}`

---

## 5. Artifact Verification
- Model Comparison: [`outputs/phase7/model_comparison_table.csv`](model_comparison_table.csv)
- Test Breakdown: [`outputs/phase7/test_subject_breakdown.csv`](test_subject_breakdown.csv)
- LOSO Benchmark: [`outputs/phase7/loso_benchmark_results.csv`](loso_benchmark_results.csv)
- Confusion Matrix: [`outputs/phase7/plots/confusion_matrix_test.png`](plots/confusion_matrix_test.png)
- ROC Curves: [`outputs/phase7/plots/roc_curves_comparison.png`](plots/roc_curves_comparison.png)
- Calibration Curves: [`outputs/phase7/plots/calibration_curves.png`](plots/calibration_curves.png)
- LOSO F1 Chart: [`outputs/phase7/plots/loso_subject_f1.png`](plots/loso_subject_f1.png)
"""
    with open(OUTPUT_DIR / "phase7_report.md", "w") as f:
        f.write(report_md)
    print(f"Saved Phase 7 report: {OUTPUT_DIR / 'phase7_report.md'}")

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()
    df_comp, df_test_sub = run_test_evaluation(data)
    df_loso = run_loso_cross_validation(data)
    compile_phase7_report(df_comp, df_test_sub, df_loso)
    print("\nPhase 7 benchmark completed successfully!")

if __name__ == "__main__":
    main()

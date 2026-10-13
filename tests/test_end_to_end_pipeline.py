"""
End-to-End Automated Regression & Integrity Test Suite.
Validates model provenance, data authenticity, feature extraction,
baseline engine mathematics, and subject partition independence.
"""

import os
import sys
import json
import hashlib
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.downstream_features import extract_window_features, extract_window_features_dict, FEATURE_NAMES
from src.personal_baseline_engine import PersonalBaselineProfile, StrainEngine

EXPECTED_CKPT_SHA256 = "36d0bca18356ccfc8e8916058bf838f26f1212a646f5780b487ad78581a92561"

def test_checkpoint_integrity():
    ckpt_path = repo_root / "checkpoints" / "normwear_pretrain_ckpt.pth"
    assert ckpt_path.exists(), "Checkpoint file missing"
    
    h = hashlib.sha256()
    with open(ckpt_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    actual_sha = h.hexdigest()
    assert actual_sha == EXPECTED_CKPT_SHA256, f"Checkpoint SHA mismatch: {actual_sha}"

def test_wesad_raw_dataset_authenticity():
    raw_dir = repo_root / "data" / "raw" / "WESAD"
    expected_subs = [f"S{i}" for i in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]]
    for sub in expected_subs:
        pkl_path = raw_dir / sub / f"{sub}.pkl"
        assert pkl_path.exists(), f"Missing raw file for {sub}"
        assert pkl_path.stat().st_size > 10 * 1024 * 1024, f"File too small for {sub}"

def test_processed_windows_integrity():
    wins_path = repo_root / "data" / "processed" / "wesad_windows" / "windows.npy"
    meta_path = repo_root / "data" / "processed" / "wesad_windows" / "metadata.parquet"
    
    assert wins_path.exists(), "windows.npy missing"
    assert meta_path.exists(), "metadata.parquet missing"
    
    wins = np.load(wins_path, mmap_mode="r")
    df_meta = pd.read_parquet(meta_path)
    
    assert wins.shape == (28938, 3, 384), f"Unexpected shape {wins.shape}"
    assert len(df_meta) == 28938, f"Unexpected metadata row count {len(df_meta)}"
    
    valid_df = df_meta[df_meta["valid_for_training"]]
    assert len(valid_df) == 9078, f"Expected 9078 valid training windows, got {len(valid_df)}"
    assert valid_df["label_purity"].min() >= 0.80, "Purity violated"
    assert set(valid_df["target_binary_label"].unique()) == {0, 1}, "Labels invalid"

def test_physiological_feature_extraction():
    wins_path = repo_root / "data" / "processed" / "wesad_windows" / "windows.npy"
    wins = np.load(wins_path, mmap_mode="r")
    
    # Test on first window
    sample = np.array(wins[0], dtype=np.float32)
    feats_vec = extract_window_features(sample)
    feats_dict = extract_window_features_dict(sample)
    
    assert len(feats_vec) == len(FEATURE_NAMES), f"Expected {len(FEATURE_NAMES)} features, got {len(feats_vec)}"
    for val in feats_vec:
        assert not np.isnan(val), "NaN in feature"
        assert not np.isinf(val), "Inf in feature"
        
    # Check physiological plausibility
    assert 40.0 <= feats_dict["hr_mean_bpm"] <= 180.0, f"Implausible HR: {feats_dict['hr_mean_bpm']}"
    assert feats_dict["hrv_rmssd_ms"] >= 0.0, "Negative RMSSD"
    assert 20.0 <= feats_dict["temp_mean_c"] <= 42.0, f"Implausible Temp: {feats_dict['temp_mean_c']}"

def test_subject_split_leakage():
    config_path = repo_root / "config" / "subject_split.yaml"
    with open(config_path, "r") as f:
        split_cfg = yaml.safe_load(f)
        
    train = set(split_cfg["train_subjects"])
    val = set(split_cfg["validation_subjects"])
    test = set(split_cfg["test_subjects"])
    
    assert len(train.intersection(val)) == 0, "Train-Val subject leakage!"
    assert len(train.intersection(test)) == 0, "Train-Test subject leakage!"
    assert len(val.intersection(test)) == 0, "Val-Test subject leakage!"
    assert len(train.union(val).union(test)) == 15, "Subject count mismatch"

def test_strain_engine_mathematics():
    # Construct synthetic baseline profile
    dummy_profile = PersonalBaselineProfile(
        subject_id="TEST_S1",
        embedding_centroid=np.zeros(768, dtype=np.float32),
        embedding_radius_std=1.0,
        hr_mean=70.0,
        hr_std=5.0,
        rmssd_mean=35.0,
        rmssd_std=5.0,
        eda_mean=1.0,
        eda_std=0.2,
        scr_rate_mean=2.0,
        scr_rate_std=0.5,
        temp_mean=32.0,
        temp_std=0.5,
        n_calibration_windows=50
    )
    
    engine = StrainEngine({"TEST_S1": dummy_profile})
    
    # 1. Homeostatic input (exact resting state)
    res_homeo = engine.evaluate_window(
        subject_id="TEST_S1",
        embedding=np.zeros(768, dtype=np.float32),
        physio_features={
            "hr_mean_bpm": 70.0,
            "hrv_rmssd_ms": 35.0,
            "eda_mean_scl": 1.0,
            "eda_scr_rate_per_min": 2.0,
            "temp_mean_c": 32.0
        },
        p_stress=0.05
    )
    assert res_homeo["strain_score"] < 30.0, f"Homeostatic strain too high: {res_homeo['strain_score']}"
    assert res_homeo["strain_state"] in ["Homeostatic Rest", "Mild Activation"]
    
    # 2. Severe stress input (high HR, low HRV, huge EDA surge)
    res_stress = engine.evaluate_window(
        subject_id="TEST_S1",
        embedding=np.ones(768, dtype=np.float32) * 2.0,
        physio_features={
            "hr_mean_bpm": 95.0,      # +25 bpm tachycardia
            "hrv_rmssd_ms": 15.0,      # vagal suppression
            "eda_mean_scl": 4.5,       # +3.5 uS sympathetic surge
            "eda_scr_rate_per_min": 10.0,
            "temp_mean_c": 30.5
        },
        p_stress=0.95
    )
    assert res_stress["strain_score"] > 70.0, f"Stress strain too low: {res_stress['strain_score']}"
    assert res_stress["strain_score"] > res_homeo["strain_score"], "Stress strain must exceed baseline"
    assert res_stress["primary_cause"] != "", "Missing primary cause attribution"

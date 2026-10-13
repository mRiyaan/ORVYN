"""
WESAD Dataset Inspection and Validation Pipeline.

Provides reusable modular functions to load, audit, validate, and inventory
the raw WESAD multimodal dataset without altering raw files or fabricating any values.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Official WESAD nominal sampling rates based on paper and device specifications
# Chest (RespiBAN): 700 Hz for all modalities
# Wrist (Empatica E4): BVP=64 Hz, EDA=4 Hz, TEMP=4 Hz, ACC=32 Hz
NOMINAL_SAMPLING_RATES = {
    "chest": {
        "ACC": 700,
        "ECG": 700,
        "EMG": 700,
        "EDA": 700,
        "Temp": 700,
        "Resp": 700,
    },
    "wrist": {
        "BVP": 64,
        "EDA": 4,
        "TEMP": 4,
        "ACC": 32,
    }
}

# Official WESAD label mapping from dataset documentation (readme.pdf / ICMI 2018)
# 0 = not defined / transient
# 1 = baseline
# 2 = stress (TSST)
# 3 = amusement
# 4 = meditation
# 5 = ignore / rest
# 6 = ignore / rest
# 7 = ignore / rest
LABEL_NAMES = {
    0: "undefined / transient",
    1: "baseline",
    2: "stress",
    3: "amusement",
    4: "meditation",
    5: "ignore / rest",
    6: "ignore / rest",
    7: "ignore / rest"
}

def load_wesad_subject(subject_dir: Union[str, Path], subject_id: str) -> Dict[str, Any]:
    """
    Load a raw WESAD subject pickle file using latin1 encoding.
    Does NOT modify the file.
    Robustly handles directory variations (e.g. subject_dir/S2/S2.pkl or globbing).
    """
    subject_dir = Path(subject_dir)
    primary_path = subject_dir / subject_id / f"{subject_id}.pkl"
    if primary_path.is_file():
        pkl_path = primary_path
    else:
        # Search recursively in case of nested directory extraction
        candidates = list(subject_dir.glob(f"**/{subject_id}/{subject_id}.pkl")) + \
                     list(subject_dir.glob(f"**/{subject_id}.pkl"))
        if candidates and candidates[0].is_file():
            pkl_path = candidates[0]
        else:
            raise FileNotFoundError(f"Subject pickle file not found for {subject_id} in {subject_dir}")

    with pkl_path.open("rb") as f:
        data = pickle.load(f, encoding="latin1")
    return data

def inspect_subject(subject_dir: Union[str, Path], subject_id: str) -> Dict[str, Any]:
    """
    Inspect a single subject's top-level structure, devices, signals, and label array.
    """
    subject_dir = Path(subject_dir)
    primary_path = subject_dir / subject_id / f"{subject_id}.pkl"
    
    if primary_path.is_file():
        pkl_path = primary_path
    else:
        candidates = list(subject_dir.glob(f"**/{subject_id}/{subject_id}.pkl")) + \
                     list(subject_dir.glob(f"**/{subject_id}.pkl"))
        pkl_path = candidates[0] if (candidates and candidates[0].is_file()) else primary_path

    exists = pkl_path.is_file()
    
    info: Dict[str, Any] = {
        "subject_id": subject_id,
        "source_file": str(pkl_path),
        "file_exists": exists,
        "load_success": False,
        "signal_groups": [],
        "label_exists": False,
        "label_length": 0,
        "status": "MISSING_FILE",
        "notes": ""
    }
    
    if not exists:
        return info
        
    try:
        data = load_wesad_subject(subject_dir, subject_id)
        info["load_success"] = True
        
        # Verify signal groups
        signal_dict = data.get("signal", {})
        info["signal_groups"] = list(signal_dict.keys())
        
        # Verify labels
        labels = data.get("label")
        if labels is not None:
            info["label_exists"] = True
            info["label_length"] = len(np.array(labels).flatten())
            info["status"] = "PASS"
            info["notes"] = f"Devices: {list(signal_dict.keys())}"
        else:
            info["status"] = "MISSING_LABELS"
            info["notes"] = "No label key in pickle"
            
    except Exception as e:
        info["status"] = "LOAD_ERROR"
        info["notes"] = str(e)
        
    return info

def inspect_channels(data: Dict[str, Any], subject_id: str) -> List[Dict[str, Any]]:
    """
    Inspect all sensor channels across devices for a loaded subject dictionary.
    """
    records = []
    signal_dict = data.get("signal", {})
    
    for device_name, channels in signal_dict.items():
        if not isinstance(channels, dict):
            continue
        for channel_name, arr in channels.items():
            shape = getattr(arr, "shape", None)
            dtype = str(getattr(arr, "dtype", None))
            sample_count = len(arr) if hasattr(arr, "__len__") else 0
            
            records.append({
                "subject_id": subject_id,
                "device": device_name,
                "channel": channel_name,
                "shape": str(list(shape)) if shape is not None else "None",
                "sample_count": sample_count,
                "dtype": dtype,
                "available": True,
                "notes": f"Dim count: {len(shape) if shape is not None else 0}"
            })
    return records

def inspect_sampling(data: Dict[str, Any], subject_id: str) -> List[Dict[str, Any]]:
    """
    Inspect sampling rates and inferred durations across modalities.
    """
    records = []
    signal_dict = data.get("signal", {})
    
    for device_name, channels in signal_dict.items():
        if not isinstance(channels, dict):
            continue
        device_nominals = NOMINAL_SAMPLING_RATES.get(device_name.lower(), {})
        
        for channel_name, arr in channels.items():
            sample_count = len(arr) if hasattr(arr, "__len__") else 0
            # Check nominal rate
            fs = device_nominals.get(channel_name)
            if fs is None:
                # Try case insensitive lookup
                for k, v in device_nominals.items():
                    if k.lower() == channel_name.lower():
                        fs = v
                        break
            
            duration_s = (sample_count / fs) if fs else None
            
            records.append({
                "subject_id": subject_id,
                "device": device_name,
                "channel": channel_name,
                "source_sampling_rate_hz": fs if fs is not None else "UNKNOWN",
                "sample_count": sample_count,
                "estimated_duration_seconds": round(duration_s, 3) if duration_s is not None else "UNKNOWN",
                "sampling_rate_source": "WESAD Readme / Empatica & RespiBAN Specifications",
                "status": "VERIFIED" if fs is not None else "UNVERIFIED"
            })
            
    # Also include the label array (sampled synchronously with RespiBAN chest device at 700 Hz)
    labels = data.get("label")
    if labels is not None:
        labels_len = len(np.array(labels).flatten())
        fs_lbl = 700
        dur_lbl = labels_len / fs_lbl
        records.append({
            "subject_id": subject_id,
            "device": "label_stream",
            "channel": "label",
            "source_sampling_rate_hz": fs_lbl,
            "sample_count": labels_len,
            "estimated_duration_seconds": round(dur_lbl, 3),
            "sampling_rate_source": "WESAD Readme (Synchronized to 700 Hz Chest stream)",
            "status": "VERIFIED"
        })
        
    return records

def inspect_labels(data: Dict[str, Any], subject_id: str) -> List[Dict[str, Any]]:
    """
    Inspect the distribution of labels for a subject.
    """
    labels = data.get("label")
    if labels is None:
        return []
    
    labels_flat = np.array(labels).flatten()
    total_samples = len(labels_flat)
    unique_codes, counts = np.unique(labels_flat, return_counts=True)
    
    records = []
    for code, count in zip(unique_codes, counts):
        code_int = int(code)
        pct = (count / total_samples) * 100.0
        dur_s = count / 700.0  # Synchronized to 700 Hz
        
        records.append({
            "subject_id": subject_id,
            "label_code": code_int,
            "label_name": LABEL_NAMES.get(code_int, f"Unknown ({code_int})"),
            "sample_count": int(count),
            "percentage": round(pct, 4),
            "estimated_duration_seconds": round(dur_s, 2)
        })
    return records

def audit_raw_signal_quality(data: Dict[str, Any], subject_id: str) -> List[Dict[str, Any]]:
    """
    Compute objective numeric quality statistics for every signal channel.
    """
    records = []
    signal_dict = data.get("signal", {})
    
    for device_name, channels in signal_dict.items():
        if not isinstance(channels, dict):
            continue
        for channel_name, arr in channels.items():
            arr_np = np.asarray(arr)
            # If multi-column (e.g. 3-axis ACC), evaluate both per-axis and overall
            if arr_np.ndim == 2 and arr_np.shape[1] > 1:
                axes = ["x", "y", "z"] if arr_np.shape[1] == 3 else [f"col_{i}" for i in range(arr_np.shape[1])]
                for col_idx, axis_label in enumerate(axes):
                    col_data = arr_np[:, col_idx]
                    records.append(_compute_quality_stats(
                        subject_id, device_name, f"{channel_name}_{axis_label}", col_data
                    ))
            else:
                col_data = arr_np.flatten()
                records.append(_compute_quality_stats(
                    subject_id, device_name, channel_name, col_data
                ))
    return records

def _compute_quality_stats(subject_id: str, device: str, channel: str, arr: np.ndarray) -> Dict[str, Any]:
    sample_count = len(arr)
    nan_count = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.number) else 0
    inf_count = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.floating) else 0
    finite_count = sample_count - nan_count - inf_count
    finite_frac = (finite_count / sample_count) if sample_count > 0 else 0.0
    
    if finite_count > 0 and np.issubdtype(arr.dtype, np.number):
        valid_vals = arr[np.isfinite(arr)]
        if len(valid_vals) > 0:
            val_min = float(np.min(valid_vals))
            val_max = float(np.max(valid_vals))
            val_mean = float(np.mean(valid_vals))
            val_std = float(np.std(valid_vals))
        else:
            val_min, val_max, val_mean, val_std = float('nan'), float('nan'), float('nan'), float('nan')
    else:
        val_min, val_max, val_mean, val_std = float('nan'), float('nan'), float('nan'), float('nan')
        
    return {
        "subject_id": subject_id,
        "device": device,
        "channel": channel,
        "sample_count": sample_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "finite_fraction": round(finite_frac, 6),
        "min": round(val_min, 4),
        "max": round(val_max, 4),
        "mean": round(val_mean, 4),
        "std": round(val_std, 4),
    }

def audit_channel_length_consistency(data: Dict[str, Any], subject_id: str) -> List[Dict[str, Any]]:
    """
    Check duration consistency across modalities within a subject.
    """
    sampling_records = inspect_sampling(data, subject_id)
    durations = [r["estimated_duration_seconds"] for r in sampling_records if isinstance(r["estimated_duration_seconds"], (int, float))]
    
    if not durations:
        return []
    
    reference_duration = max(durations)
    
    consistency_records = []
    for r in sampling_records:
        dur = r["estimated_duration_seconds"]
        if isinstance(dur, (int, float)):
            diff_s = round(dur - reference_duration, 3)
            diff_pct = round(abs(diff_s) / reference_duration * 100.0, 4)
            # Devices turned on at slightly different moments during experiment setup:
            # Difference within a few seconds (e.g. < 1%) is expected hardware operational offset.
            status = "CONSISTENT" if diff_pct < 1.0 else ("MINOR_OFFSET" if diff_pct < 5.0 else "LARGE_DISCREPANCY")
        else:
            diff_s, diff_pct = "UNKNOWN", "UNKNOWN"
            status = "UNKNOWN"
            
        consistency_records.append({
            "subject_id": subject_id,
            "device": r["device"],
            "channel": r["channel"],
            "sample_count": r["sample_count"],
            "fs_hz": r["source_sampling_rate_hz"],
            "duration_seconds": dur,
            "ref_duration_seconds": reference_duration,
            "delta_seconds": diff_s,
            "delta_pct": diff_pct,
            "status": status
        })
    return consistency_records

"""
Physiological Feature Extraction and Fusion for Downstream Stress Detection.

Derives explicit statistical, cardiac (BVP/HRV proxy), electrodermal (EDA),
and thermal (TEMP) biomarkers directly from verified WESAD wrist windows.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import scipy.signal
import scipy.stats

FEATURE_NAMES = [
    # Cardiac features (from wrist BVP @ 64 Hz)
    "bvp_mean",
    "bvp_std",
    "bvp_ptp",
    "bvp_skew",
    "bvp_kurtosis",
    "hr_bpm_est",
    "hrv_sdnn_ms",
    "hrv_rmssd_ms",
    # Electrodermal features (from wrist EDA @ 64 Hz)
    "eda_mean",
    "eda_std",
    "eda_min",
    "eda_max",
    "eda_range",
    "eda_slope",
    # Temperature features (from wrist TEMP @ 64 Hz)
    "temp_mean",
    "temp_std",
    "temp_slope",
    # Personal baseline deviation features (when baseline profile is available)
    "eda_dev_from_base",
    "temp_dev_from_base",
    "hr_dev_from_base",
]

def extract_window_features(
    window: np.ndarray,
    subject_baseline: Optional[Dict[str, float]] = None
) -> np.ndarray:
    """
    Extract 20 explicit physiological features from a [3, sequence_length] window.
    Channels: 0: BVP, 1: EDA, 2: TEMP.
    """
    bvp = window[0]
    eda = window[1]
    temp = window[2]
    
    n_samples = len(bvp)
    duration_s = n_samples / 64.0
    t_sec = np.arange(n_samples) / 64.0
    
    # 1. BVP Features
    b_mean = float(np.mean(bvp))
    b_std = float(np.std(bvp))
    b_ptp = float(np.ptp(bvp))
    b_skew = float(scipy.stats.skew(bvp))
    b_kurt = float(scipy.stats.kurtosis(bvp))
    
    # Peak detection for heart rate & HRV estimation
    # At 64 Hz, physiological distance between beats is >= 22 samples (~170 bpm max)
    prominence = max(b_std * 0.5, 1.0)
    peaks, _ = scipy.signal.find_peaks(bvp, distance=22, prominence=prominence)
    
    if len(peaks) >= 2:
        ibis = np.diff(peaks) / 64.0 * 1000.0 # Inter-beat intervals in ms
        mean_ibi = np.mean(ibis)
        hr_bpm = float(60000.0 / mean_ibi) if mean_ibi > 0 else 75.0
        hr_bpm = np.clip(hr_bpm, 40.0, 190.0)
        hrv_sdnn = float(np.std(ibis))
        if len(ibis) >= 2:
            hrv_rmssd = float(np.sqrt(np.mean(np.diff(ibis)**2)))
        else:
            hrv_rmssd = hrv_sdnn
    else:
        hr_bpm = 75.0
        hrv_sdnn = 0.0
        hrv_rmssd = 0.0
        
    # 2. EDA Features
    e_mean = float(np.mean(eda))
    e_std = float(np.std(eda))
    e_min = float(np.min(eda))
    e_max = float(np.max(eda))
    e_range = float(e_max - e_min)
    if n_samples >= 2:
        e_slope = float(np.polyfit(t_sec, eda, 1)[0])
    else:
        e_slope = 0.0
        
    # 3. TEMP Features
    t_mean = float(np.mean(temp))
    t_std = float(np.std(temp))
    if n_samples >= 2:
        t_slope = float(np.polyfit(t_sec, temp, 1)[0])
    else:
        t_slope = 0.0
        
    # 4. Personal Baseline Deviation Features
    if subject_baseline is not None:
        eda_dev = e_mean - subject_baseline.get("eda_mean", e_mean)
        temp_dev = t_mean - subject_baseline.get("temp_mean", t_mean)
        hr_dev = hr_bpm - subject_baseline.get("hr_bpm", hr_bpm)
    else:
        eda_dev = 0.0
        temp_dev = 0.0
        hr_dev = 0.0
        
    feat_vec = np.array([
        b_mean, b_std, b_ptp, b_skew, b_kurt,
        hr_bpm, hrv_sdnn, hrv_rmssd,
        e_mean, e_std, e_min, e_max, e_range, e_slope,
        t_mean, t_std, t_slope,
        eda_dev, temp_dev, hr_dev
    ], dtype=np.float32)
    
    return feat_vec

def extract_window_features_dict(
    window: np.ndarray,
    subject_baseline: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Extracts explicit physiological features and returns a dictionary with standard keys and aliases.
    """
    vec = extract_window_features(window, subject_baseline)
    d = {name: float(val) for name, val in zip(FEATURE_NAMES, vec)}
    # Aliases for downstream components
    d["hr_mean_bpm"] = d["hr_bpm_est"]
    d["eda_mean_scl"] = d["eda_mean"]
    d["eda_scr_rate_per_min"] = max(d["eda_std"] * 5.0, 0.5)
    d["temp_mean_c"] = d["temp_mean"]
    return d

def compute_subject_baselines(
    windows: np.ndarray,
    metadata_df: Any
) -> Dict[str, Dict[str, float]]:
    """
    Compute subject-level baseline norms using ONLY neutral baseline windows (label == 0).
    Never uses stress or test windows.
    """
    baselines = {}
    subjects = metadata_df["subject_id"].unique()
    
    for sid in subjects:
        # Select baseline windows for this subject
        base_mask = (metadata_df["subject_id"] == sid) & (metadata_df["target_binary_label"] == 0)
        base_indices = np.where(base_mask.values)[0]
        
        if len(base_indices) > 0:
            sub_wins = windows[base_indices]
            # Average across baseline windows
            sub_eda_mean = float(np.mean(sub_wins[:, 1, :]))
            sub_temp_mean = float(np.mean(sub_wins[:, 2, :]))
            
            # Compute average HR during baseline
            hrs = []
            for w in sub_wins[:min(50, len(sub_wins))]:
                feats = extract_window_features(w, subject_baseline=None)
                hrs.append(feats[5]) # hr_bpm_est
            sub_hr_mean = float(np.mean(hrs)) if hrs else 75.0
            
            baselines[sid] = {
                "eda_mean": sub_eda_mean,
                "temp_mean": sub_temp_mean,
                "hr_bpm": sub_hr_mean,
                "baseline_window_count": len(base_indices)
            }
        else:
            baselines[sid] = {
                "eda_mean": 1.0,
                "temp_mean": 32.0,
                "hr_bpm": 75.0,
                "baseline_window_count": 0
            }
            
    return baselines

def extract_dataset_features(
    windows: np.ndarray,
    metadata_df: Any,
    subject_baselines: Optional[Dict[str, Dict[str, float]]] = None,
    baselines: Optional[Dict[str, Dict[str, float]]] = None
) -> np.ndarray:
    """
    Extract explicit features for all windows in the dataset.
    Returns: [N, 20] float32 matrix.
    """
    if subject_baselines is None and baselines is not None:
        subject_baselines = baselines
    n_windows = len(windows)
    feats_list = []
    
    for i in range(n_windows):
        sid = metadata_df.iloc[i]["subject_id"]
        base_dict = subject_baselines.get(sid) if subject_baselines else None
        feat_vec = extract_window_features(windows[i], subject_baseline=base_dict)
        feats_list.append(feat_vec)
        
    return np.stack(feats_list, axis=0).astype(np.float32)

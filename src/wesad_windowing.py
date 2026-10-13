"""
WESAD Signal Cleaning, Synchronization, Windowing & Label Alignment.

Performs deterministic multi-rate synchronization (4 Hz -> 64 Hz for EDA/TEMP),
strict window segmentation (6.0s windows, 3.0s stride), label alignment with
label-purity enforcement, and signal quality auditing.
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import scipy.signal
import scipy.interpolate

DEFAULT_CONFIG = {
    "window_seconds": 6.0,
    "stride_seconds": 3.0,
    "target_sampling_rate_hz": 64,
    "label_purity_threshold": 0.80,
    "selected_channels": ["wrist_BVP", "wrist_EDA", "wrist_TEMP"],
    "channel_order": ["wrist_BVP", "wrist_EDA", "wrist_TEMP"],
    "baseline_code": 1,
    "stress_code": 2,
}

def resample_4hz_to_64hz(signal_4hz: np.ndarray, target_length: int, method: str = "cubic") -> np.ndarray:
    """
    Resample a 4 Hz signal (EDA or TEMP) to 64 Hz (16x upsampling).
    Uses monotonic/cubic interpolation with exact time-grid alignment.
    """
    sig = signal_4hz.flatten().astype(np.float32)
    n_orig = len(sig)
    if n_orig < 2:
        return np.repeat(sig, 16)[:target_length].astype(np.float32)
        
    t_orig = np.arange(n_orig) / 4.0
    t_target = np.arange(target_length) / 64.0
    
    # Clip t_target to t_orig bounds to prevent extrapolation edge issues
    t_target_clipped = np.clip(t_target, t_orig[0], t_orig[-1])
    
    if method == "cubic" and n_orig >= 4:
        interp = scipy.interpolate.interp1d(t_orig, sig, kind="cubic", assume_sorted=True)
    else:
        interp = scipy.interpolate.interp1d(t_orig, sig, kind="linear", assume_sorted=True)
        
    return interp(t_target_clipped).astype(np.float32)

def extract_subject_windows(
    subject_data: Dict[str, Any],
    subject_id: str,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Extract synchronized, quality-audited windows for a single WESAD subject.
    Returns:
        windows: np.ndarray of shape [N_windows, 3, sequence_length] in float32.
        metadata: List of dicts with window metadata, label purity, and quality metrics.
    """
    cfg = config or DEFAULT_CONFIG
    win_sec = cfg["window_seconds"]
    stride_sec = cfg["stride_seconds"]
    target_fs = cfg["target_sampling_rate_hz"]
    purity_thresh = cfg["label_purity_threshold"]
    
    win_samples = int(round(win_sec * target_fs))      # 6.0s * 64 = 384
    stride_samples = int(round(stride_sec * target_fs)) # 3.0s * 64 = 192
    
    wrist = subject_data["signal"]["wrist"]
    bvp = np.array(wrist["BVP"]).flatten().astype(np.float32)
    eda = np.array(wrist["EDA"]).flatten().astype(np.float32)
    temp = np.array(wrist["TEMP"]).flatten().astype(np.float32)
    labels_700 = np.array(subject_data["label"]).flatten().astype(np.int32)
    
    # Common duration is determined by BVP length at 64 Hz
    n_samples_64 = len(bvp)
    total_duration_s = n_samples_64 / target_fs
    
    # Resample EDA and TEMP to match exact BVP length at 64 Hz
    eda_64 = resample_4hz_to_64hz(eda, n_samples_64, method="cubic")
    temp_64 = resample_4hz_to_64hz(temp, n_samples_64, method="linear")
    
    # Pre-allocate stacked synchronized matrix: [3, n_samples_64]
    # Channel order: 0: BVP, 1: EDA, 2: TEMP
    sync_matrix = np.stack([bvp, eda_64, temp_64], axis=0) # [3, N]
    
    # Compute window intervals
    num_windows = (n_samples_64 - win_samples) // stride_samples + 1
    if num_windows <= 0:
        return np.empty((0, 3, win_samples), dtype=np.float32), []
        
    windows_list = []
    metadata_list = []
    
    for w_idx in range(num_windows):
        start_sample_64 = w_idx * stride_samples
        end_sample_64 = start_sample_64 + win_samples
        
        t_start_s = start_sample_64 / target_fs
        t_end_s = end_sample_64 / target_fs
        
        # Window signal slice: [3, win_samples]
        win_sig = sync_matrix[:, start_sample_64:end_sample_64]
        
        # Label slice from 700 Hz label stream
        lbl_start_700 = int(round(t_start_s * 700.0))
        lbl_end_700 = int(round(t_end_s * 700.0))
        win_labels = labels_700[lbl_start_700:lbl_end_700]
        
        # Evaluate label distribution
        if len(win_labels) > 0:
            codes, counts = np.unique(win_labels, return_counts=True)
            lbl_counts = dict(zip(codes.tolist(), counts.tolist()))
            n_lbl = len(win_labels)
            
            baseline_frac = lbl_counts.get(1, 0) / n_lbl
            stress_frac = lbl_counts.get(2, 0) / n_lbl
            amusement_frac = lbl_counts.get(3, 0) / n_lbl
            meditation_frac = lbl_counts.get(4, 0) / n_lbl
            transient_frac = lbl_counts.get(0, 0) / n_lbl
            rest_frac = sum(lbl_counts.get(c, 0) for c in [5, 6, 7]) / n_lbl
            
            dominant_code = int(codes[np.argmax(counts)])
            dominant_purity = float(np.max(counts) / n_lbl)
        else:
            baseline_frac = stress_frac = amusement_frac = meditation_frac = transient_frac = rest_frac = 0.0
            dominant_code = 0
            dominant_purity = 0.0

        # Quality auditing on the window
        finite_count = int(np.isfinite(win_sig).sum())
        total_vals = win_sig.size
        finite_frac = finite_count / total_vals
        
        # Flatline detection: check standard deviation across time for each channel
        stds = np.std(win_sig, axis=1) # [3]
        flatline = bool(np.any(stds < 1e-7))
        
        # Signal amplitude bounds
        bvp_std = float(stds[0])
        eda_val = float(np.mean(win_sig[1]))
        temp_val = float(np.mean(win_sig[2]))
        
        # Label alignment decision
        valid_for_training = False
        rejection_reason = "OK"
        target_label = -1 # -1 = non-target / excluded
        
        if finite_frac < 1.0:
            rejection_reason = "non_finite_values"
        elif flatline:
            rejection_reason = "flatline_signal"
        elif baseline_frac >= purity_thresh:
            target_label = 0
            valid_for_training = True
        elif stress_frac >= purity_thresh:
            target_label = 1
            valid_for_training = True
        elif dominant_purity < purity_thresh:
            rejection_reason = "ambiguous_labels"
        else:
            rejection_reason = f"unsupported_condition_{dominant_code}"
            
        window_id = f"{subject_id}_w{w_idx:05d}"
        
        metadata_list.append({
            "window_id": window_id,
            "subject_id": subject_id,
            "start_time_s": round(t_start_s, 3),
            "end_time_s": round(t_end_s, 3),
            "source_start_idx": start_sample_64,
            "source_end_idx": end_sample_64,
            "dominant_raw_label": dominant_code,
            "label_purity": round(dominant_purity, 4),
            "baseline_frac": round(baseline_frac, 4),
            "stress_frac": round(stress_frac, 4),
            "amusement_frac": round(amusement_frac, 4),
            "meditation_frac": round(meditation_frac, 4),
            "transient_frac": round(transient_frac, 4),
            "target_binary_label": target_label,
            "valid_for_training": valid_for_training,
            "rejection_reason": rejection_reason,
            "finite_fraction": round(finite_frac, 6),
            "bvp_std": round(bvp_std, 4),
            "eda_mean": round(eda_val, 4),
            "temp_mean": round(temp_val, 4),
            "sequence_length": win_samples,
            "sampling_rate_hz": target_fs,
            "channel_order": "wrist_BVP,wrist_EDA,wrist_TEMP",
        })
        windows_list.append(win_sig)
        
    windows_array = np.stack(windows_list, axis=0).astype(np.float32)
    return windows_array, metadata_list

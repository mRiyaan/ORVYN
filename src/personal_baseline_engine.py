"""
Personalized Physiological Strain, Cause Attribution & Recovery Engine.
Computes subject-specific resting baseline profiles, evaluates real-time deviations,
calculates calibrated Strain Index (0-100), decomposes causal drivers, and forecasts recovery trajectories.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd

class BaselineFormulas:
    """Mathematical formulas for baseline deviation."""
    
    @staticmethod
    def absolute_deviation(x: float, mu: float) -> float:
        """Delta = x - mu"""
        return x - mu
        
    @staticmethod
    def relative_deviation(x: float, mu: float) -> float:
        """Delta_rel = (x - mu) / |mu|"""
        if abs(mu) < 1e-6:
            return 0.0
        return (x - mu) / abs(mu)
        
    @staticmethod
    def z_score(x: float, mu: float, sigma: float, low_variance_flag: bool = False) -> float:
        """Z-score = (x - mu) / sigma with optional low variance handling"""
        if sigma < 1e-4 or low_variance_flag:
            sigma = max(sigma, 1.0)
        return (x - mu) / sigma
        
    @staticmethod
    def robust_z_score(x: float, median: float, mad: float) -> float:
        """Robust Z-score = 0.6745 * (x - median) / MAD"""
        if mad < 1e-4:
            mad = max(mad, 1.0)
        return 0.6745 * (x - median) / mad

class PersonalBaselineProfile:
    """Stores resting physiological baseline statistics and NormWear embedding centroid for an individual."""
    def __init__(
        self,
        subject_id: str,
        embedding_centroid: np.ndarray,      # [768]
        embedding_radius_std: float,
        hr_median: float,
        hr_mad: float,
        rmssd_median: float,
        rmssd_mad: float,
        eda_median: float,
        eda_mad: float,
        scr_rate_median: float,
        scr_rate_mad: float,
        temp_median: float,
        temp_mad: float,
        n_calibration_windows: int
    ):
        self.subject_id = subject_id
        self.embedding_centroid = np.asarray(embedding_centroid, dtype=np.float32)
        self.embedding_radius_std = float(max(embedding_radius_std, 1e-4))
        self.hr_median = float(hr_median)
        self.hr_mad = float(max(hr_mad, 1.0))
        self.rmssd_median = float(rmssd_median)
        self.rmssd_mad = float(max(rmssd_mad, 1.0))
        self.eda_median = float(eda_median)
        self.eda_mad = float(max(eda_mad, 0.05))
        self.scr_rate_median = float(scr_rate_median)
        self.scr_rate_mad = float(max(scr_rate_mad, 0.1))
        self.temp_median = float(temp_median)
        self.temp_mad = float(max(temp_mad, 0.1))
        self.n_calibration_windows = int(n_calibration_windows)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "embedding_centroid": self.embedding_centroid.tolist(),
            "embedding_radius_std": self.embedding_radius_std,
            "hr_median": self.hr_median,
            "hr_mad": self.hr_mad,
            "rmssd_median": self.rmssd_median,
            "rmssd_mad": self.rmssd_mad,
            "eda_median": self.eda_median,
            "eda_mad": self.eda_mad,
            "scr_rate_median": self.scr_rate_median,
            "scr_rate_mad": self.scr_rate_mad,
            "temp_median": self.temp_median,
            "temp_mad": self.temp_mad,
            "n_calibration_windows": self.n_calibration_windows
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PersonalBaselineProfile":
        return cls(
            subject_id=data["subject_id"],
            embedding_centroid=np.array(data["embedding_centroid"], dtype=np.float32),
            embedding_radius_std=data["embedding_radius_std"],
            hr_median=data["hr_median"],
            hr_mad=data["hr_mad"],
            rmssd_median=data["rmssd_median"],
            rmssd_mad=data["rmssd_mad"],
            eda_median=data["eda_median"],
            eda_mad=data["eda_mad"],
            scr_rate_median=data["scr_rate_median"],
            scr_rate_mad=data["scr_rate_mad"],
            temp_median=data["temp_median"],
            temp_mad=data["temp_mad"],
            n_calibration_windows=data["n_calibration_windows"]
        )


class StrainEngine:
    """Evaluates multi-modal physiological strain, cause attribution, and recovery trajectory."""
    def __init__(self, baseline_profiles: Dict[str, PersonalBaselineProfile]):
        self.profiles = baseline_profiles

    def evaluate_window(
        self,
        subject_id: str,
        embedding: np.ndarray,          # [768]
        physio_features: Dict[str, float], # hr, rmssd, eda_mean, eda_scr_rate, temp_mean
        p_stress: float                 # Model output probability [0, 1]
    ) -> Dict[str, Any]:
        """
        Evaluate real-time strain against the subject's personal baseline profile.
        """
        profile = self.profiles.get(subject_id)
        if profile is None:
            raise KeyError(f"No baseline profile registered for subject '{subject_id}'")

        # 1. Embedding Deviation
        emb_dist = float(np.linalg.norm(embedding - profile.embedding_centroid))
        z_emb = max(0.0, (emb_dist / profile.embedding_radius_std) - 1.0)

        # 2. Physiological Z-Scores (Using Robust Z-Score formula)
        hr = physio_features.get("hr_mean_bpm", 70.0)
        rmssd = physio_features.get("hrv_rmssd_ms", 30.0)
        eda = physio_features.get("eda_mean_scl", 1.0)
        scr = physio_features.get("eda_scr_rate_per_min", 2.0)
        temp = physio_features.get("temp_mean_c", 32.0)

        z_hr = BaselineFormulas.robust_z_score(hr, profile.hr_median, profile.hr_mad)
        # Lower RMSSD signifies vagal withdrawal (higher stress), so we reverse the difference
        z_vagal_withdrawal = BaselineFormulas.robust_z_score(profile.rmssd_median, rmssd, profile.rmssd_mad)
        z_eda = BaselineFormulas.robust_z_score(eda, profile.eda_median, profile.eda_mad)
        z_scr = BaselineFormulas.robust_z_score(scr, profile.scr_rate_median, profile.scr_rate_mad)
        # Peripheral vasoconstriction causes temperature drop
        z_vasoconstriction = BaselineFormulas.robust_z_score(profile.temp_median, temp, profile.temp_mad)

        # Combined autonomic deviation
        sympathetic_z = max(0.0, 0.6 * z_eda + 0.4 * z_scr)
        cardio_z = max(0.0, 0.5 * z_hr + 0.5 * z_vagal_withdrawal)
        thermal_z = max(0.0, z_vasoconstriction)

        composite_physio_z = 0.45 * sympathetic_z + 0.35 * cardio_z + 0.10 * thermal_z + 0.10 * z_emb

        # 3. Personalized Strain Index (0 to 100)
        dev_prob = 1.0 / (1.0 + np.exp(-1.5 * (composite_physio_z - 1.0)))
        blended_prob = 0.60 * p_stress + 0.40 * dev_prob
        strain_score = float(np.clip(blended_prob * 100.0, 0.0, 100.0))

        # Strain Category
        if strain_score < 25.0:
            strain_state = "Homeostatic Rest"
            status_color = "#28a745" # Green
        elif strain_score < 50.0:
            strain_state = "Mild Activation"
            status_color = "#17a2b8" # Teal
        elif strain_score < 75.0:
            strain_state = "Moderate Autonomic Strain"
            status_color = "#ffc107" # Amber
        else:
            strain_state = "Acute Physiological Surge"
            status_color = "#dc3545" # Red

        # 4. Cause Attribution Decomposition (% share)
        weights = {
            "Sympathetic Arousal (EDA/Skin Conductance)": max(sympathetic_z, 0.05),
            "Cardiovascular Strain (Heart Rate Surge)": max(z_hr, 0.05),
            "Parasympathetic Withdrawal (Low HRV)": max(z_vagal_withdrawal, 0.05),
            "Peripheral Vasomotor Reflex (Skin Temp)": max(thermal_z, 0.05),
            "NormWear ViT Morphological Drift": max(z_emb, 0.05),
        }
        total_w = sum(weights.values())
        cause_attribution = {k: round(float(v / total_w * 100.0), 1) for k, v in weights.items()}
        primary_cause = max(cause_attribution.items(), key=lambda x: x[1])[0]

        # 5. Recovery Trajectory Forecasting
        vagal_factor = np.clip(profile.rmssd_median / 40.0, 0.5, 2.0)
        base_recovery_time_mins = float(np.clip((strain_score / 15.0) / vagal_factor, 0.5, 30.0))

        # Exponential decay trajectory (projected strain over next 15 minutes)
        t_minutes = np.array([0, 2, 5, 8, 12, 15], dtype=float)
        decay_constant = base_recovery_time_mins / 2.5
        projected_strain = strain_score * np.exp(-t_minutes / max(decay_constant, 0.5))
        projected_strain = np.clip(projected_strain, 10.0, 100.0).tolist()

        return {
            "subject_id": subject_id,
            "strain_score": round(strain_score, 1),
            "strain_state": strain_state,
            "status_color": status_color,
            "ml_stress_prob": round(float(p_stress), 4),
            "composite_deviation_z": round(float(composite_physio_z), 2),
            "primary_cause": primary_cause,
            "cause_attribution": cause_attribution,
            "projected_recovery_time_mins": round(base_recovery_time_mins, 1),
            "projected_recovery_curve": {
                "minutes": t_minutes.tolist(),
                "projected_strain": [round(s, 1) for s in projected_strain]
            },
            "metrics": {
                "hr_bpm": round(float(hr), 1),
                "hr_delta_bpm": round(float(hr - profile.hr_median), 1),
                "rmssd_ms": round(float(rmssd), 1),
                "eda_scl": round(float(eda), 3),
                "eda_delta_scl": round(float(eda - profile.eda_median), 3),
                "scr_rate_per_min": round(float(scr), 2),
                "temp_c": round(float(temp), 2),
                "normwear_emb_distance": round(float(emb_dist), 3)
            }
        }

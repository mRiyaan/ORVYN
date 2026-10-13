import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import xgboost as xgb
import torch
from sklearn.preprocessing import StandardScaler

from NormWear.main_model import NormWearModel
from src.downstream_features import (
    FEATURE_NAMES,
    extract_window_features,
    extract_window_features_dict
)
from src.personal_baseline_engine import (
    PersonalBaselineProfile,
    StrainEngine
)

class PhysiologicalInferenceEngine:
    """
    Production-ready real-time inference engine.
    Processes 6-second multimodal wrist physiological windows and produces full strain telemetry.
    """
    def __init__(
        self,
        checkpoint_path: Union[str, Path],
        xgb_model_path: Union[str, Path],
        profiles_path: Union[str, Path],
        scaler_mean: Optional[np.ndarray] = None,
        scaler_scale: Optional[np.ndarray] = None,
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 1. Load Downstream Classifier with n_jobs=1 (avoids OpenMP/Metal thread conflict)
        xgb_p = Path(xgb_model_path)
        if not xgb_p.exists():
            raise FileNotFoundError(f"XGBoost model not found at: {xgb_p}")
        self.classifier = xgb.XGBClassifier(n_jobs=1)
        self.classifier.load_model(str(xgb_p))
        self.classifier.set_params(n_jobs=1)

        # 2. Load Frozen Foundation Model
        ckpt_p = Path(checkpoint_path)
        if not ckpt_p.exists():
            raise FileNotFoundError(f"NormWear checkpoint not found at: {ckpt_p}")
        self.foundation_model = NormWearModel(weight_path=str(ckpt_p), optimized_cwt=True).to(self.device)
        self.foundation_model.eval()
        for p in self.foundation_model.parameters():
            p.requires_grad = False

        # 3. Load Personal Baseline Profiles
        prof_p = Path(profiles_path)
        if not prof_p.exists():
            raise FileNotFoundError(f"Profiles JSON not found at: {prof_p}")
        with open(prof_p, "r") as f:
            profiles_raw = json.load(f)
        self.profiles: Dict[str, PersonalBaselineProfile] = {
            k: PersonalBaselineProfile.from_dict(v) for k, v in profiles_raw.items()
        }
        self.strain_engine = StrainEngine(self.profiles)

        # 4. Feature Scaler
        self.scaler_mean = np.asarray(scaler_mean, dtype=np.float32) if scaler_mean is not None else np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        self.scaler_scale = np.asarray(scaler_scale, dtype=np.float32) if scaler_scale is not None else np.ones(len(FEATURE_NAMES), dtype=np.float32)

    def process_window(
        self,
        subject_id: str,
        window: np.ndarray # Shape [3, 384] (BVP, EDA, TEMP @ 64 Hz)
    ) -> Dict[str, Any]:
        """
        Processes a single 6.0s window [3, 384] and outputs real-time strain analysis.
        """
        t0 = time.perf_counter()
        
        # Verify shape
        if window.ndim == 2:
            assert window.shape == (3, 384), f"Expected shape (3, 384), got {window.shape}"
            win_batch = window[np.newaxis, :, :] # [1, 3, 384]
        else:
            win_batch = window
            window = win_batch[0]

        # 1. NormWear Foundation Forward Pass
        t_fwd_start = time.perf_counter()
        tensor_in = torch.from_numpy(win_batch.astype(np.float32))
        with torch.inference_mode():
            raw_out = self.foundation_model.get_embedding(tensor_in, sampling_rate=64, device=self.device)
            # Token 0 is CLS token across each channel
            cls_out = raw_out[:, :, 0, :].clone().detach().cpu().numpy() # [1, 3, 768]
            del raw_out
        emb_768 = np.mean(cls_out[0], axis=0).astype(np.float32) # [768]
        t_foundation_ms = (time.perf_counter() - t_fwd_start) * 1000.0

        # 2. Explicit Physiological Feature Extraction
        t_phys_start = time.perf_counter()
        phys_dict = extract_window_features_dict(window)
        phys_vec = np.array([phys_dict[f] for f in FEATURE_NAMES], dtype=np.float32)
        # Standardize using fitted parameters
        phys_scaled = (phys_vec - self.scaler_mean) / np.maximum(self.scaler_scale, 1e-6)
        t_physio_ms = (time.perf_counter() - t_phys_start) * 1000.0

        # 3. Downstream XGBoost Classification
        t_xgb_start = time.perf_counter()
        fused_vec = np.hstack([emb_768, phys_scaled])[np.newaxis, :] # [1, 788]
        p_stress = float(self.classifier.predict_proba(fused_vec)[0, 1])
        t_xgb_ms = (time.perf_counter() - t_xgb_start) * 1000.0

        # 4. Personal Baseline Deviation & Recovery Engine
        t_eng_start = time.perf_counter()
        strain_analysis = self.strain_engine.evaluate_window(
            subject_id=subject_id,
            embedding=emb_768,
            physio_features=phys_dict,
            p_stress=p_stress
        )
        t_engine_ms = (time.perf_counter() - t_eng_start) * 1000.0
        
        total_latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            **strain_analysis,
            "timing_ms": {
                "foundation_vit_forward": round(t_foundation_ms, 2),
                "physio_feature_extraction": round(t_physio_ms, 2),
                "downstream_xgb_inference": round(t_xgb_ms, 2),
                "personal_baseline_deviation": round(t_engine_ms, 2),
                "total_end_to_end": round(total_latency_ms, 2)
            }
        }

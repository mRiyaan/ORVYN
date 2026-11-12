"""
Apple Watch / iOS HealthKit + CoreMotion JSON Ingestion Pipeline.

Implements the staged wearable-to-backend flow:
  iOS/watchOS JSON Payload
    -> Data Validation & QC Masks (+ Dead-Letter Quarantine)
    -> Stage 1: Causal Personal Baseline (Median/MAD), ENMO & Frozen NormWear ViT-Base ACC
    -> Stage 2: Downstream XGBoost Strain Estimation + TreeSHAP Contributor Attribution
    -> Stage 3: Deterministic Recovery Context, Wellness Policy & Grounded Assessment API JSON
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import xgboost as xgb

from src.inference_engine import NormWearModel

SCHEMA_VERSION = "1.0"
ADAPTER_ID = "apple_watch_ios_json_v1"

APPLE_WATCH_FEATURES: List[str] = [
    "hr_mean_bpm",
    "hr_baseline_delta_bpm",
    "hr_robust_z",
    "sdnn_ms",
    "sdnn_robust_z",
    "enmo_mean_g",
    "enmo_p95_g",
    "steps_ratio_to_baseline",
    "active_energy_ratio_to_baseline",
    "sleep_deficit_h",
    "sleep_efficiency_pct",
    "spo2_delta_pct",
    "resp_rate_robust_z",
    "normwear_acc_emb_norm",
]

DEFAULT_MAD_FALLBACKS: Dict[str, float] = {
    "resting_hr_bpm": 2.5,
    "sdnn_ms": 8.0,
    "sleep_duration_h": 0.5,
    "enmo_mean_g": 0.015,
    "steps": 1000.0,
    "active_energy_kcal": 80.0,
    "spo2_pct": 0.5,
    "respiratory_rate_bpm": 1.0,
}

DEFAULT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "hr_bpm": (30.0, 220.0),
    "sdnn_ms": (5.0, 300.0),
    "spo2_pct": (70.0, 100.0),
    "respiratory_rate_bpm": (6.0, 45.0),
    "acc_axis_g": (-16.0, 16.0),
}

FEATURE_TO_CATEGORY: Dict[str, str] = {
    "hr_mean_bpm": "non_exertional_physiological_elevation",
    "hr_baseline_delta_bpm": "non_exertional_physiological_elevation",
    "hr_robust_z": "non_exertional_physiological_elevation",
    "sdnn_ms": "recovery_context_deviation",
    "sdnn_robust_z": "recovery_context_deviation",
    "sleep_deficit_h": "recovery_context_deviation",
    "sleep_efficiency_pct": "recovery_context_deviation",
    "enmo_mean_g": "activity_associated_elevation",
    "enmo_p95_g": "activity_associated_elevation",
    "steps_ratio_to_baseline": "activity_associated_elevation",
    "active_energy_ratio_to_baseline": "activity_associated_elevation",
    "normwear_acc_emb_norm": "activity_associated_elevation",
    "spo2_delta_pct": "respiratory_or_oxygen_deviation",
    "resp_rate_robust_z": "respiratory_or_oxygen_deviation",
}

POLICY_CATALOG: Dict[str, Dict[str, str]] = {
    "POL_EXERTION_COOLDOWN": {
        "pattern": "activity_associated_elevation",
        "guidance": "Cool down gradually, hydrate according to personal needs, and account for recent workout intensity.",
        "guardrail": "Wellness guidance only; activity overlap indicates exertion context rather than diagnosis.",
    },
    "POL_NON_EXERTIONAL_PAUSE": {
        "pattern": "non_exertional_physiological_elevation",
        "guidance": "Take a brief rest or paced-breathing break, and re-check wearable fit and measurement quality.",
        "guardrail": "Observed autonomic pattern during low movement; not psychological stress diagnosis or medical triage.",
    },
    "POL_RECOVERY_REST": {
        "pattern": "recovery_context_deviation",
        "guidance": "Prioritize sleep opportunity, light manageable activity, and monitor next-day HRV/resting-HR trends.",
        "guardrail": "Descriptive recovery deviation vs. personal baseline; not a medical overtraining diagnosis.",
    },
    "POL_VITAL_RECHECK": {
        "pattern": "respiratory_or_oxygen_deviation",
        "guidance": "Verify band tightness/positioning and repeat measurement under rest; consult a professional if persistent.",
        "guardrail": "Technical trend flag only; consumer watch SpO2/respiration is non-diagnostic.",
    },
    "POL_INSUFFICIENT_DATA": {
        "pattern": "insufficient_data",
        "guidance": "Wearable coverage or baseline calibration is incomplete; ensure snug sensor contact and sync recent days.",
        "guardrail": "Missing modalities are kept explicitly unavailable rather than zero-filled.",
    },
}


@dataclass
class DeadLetterRecord:
    """Quarantined record failing schema, bounds, or temporal QC gates."""

    signal_name: str
    reason: str
    raw_value: Any
    timestamp_utc: Optional[str] = None
    record_index: Optional[int] = None


def _parse_utc_iso(ts: Any) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp string into timezone-aware UTC datetime."""
    if not isinstance(ts, str) or not ts.strip():
        return None
    try:
        parsed = datetime.fromisoformat(ts.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _in_bounds(val: Any, bounds: Tuple[float, float]) -> Optional[float]:
    """Convert value to finite float if within [low, high], else return None."""
    if val is None or isinstance(val, bool):
        return None
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(fval) or fval < bounds[0] or fval > bounds[1]:
        return None
    return fval


def validate_and_qc_apple_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate iOS/watchOS JSON payload, separate QC-passed streams from dead letters."""
    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")

    schema_ver = payload.get("schema_version")
    if schema_ver != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version '{schema_ver}'; expected '{SCHEMA_VERSION}'.")

    participant_id = payload.get("participant_id")
    if not isinstance(participant_id, str) or not participant_id.strip():
        raise ValueError("Missing or invalid 'participant_id'.")

    captured_at = _parse_utc_iso(payload.get("captured_at_utc"))
    if captured_at is None:
        raise ValueError("Missing or invalid ISO-8601 UTC 'captured_at_utc'.")

    signals = payload.get("signals")
    if not isinstance(signals, dict):
        raise ValueError("Missing or invalid 'signals' object.")

    dead_letters: List[DeadLetterRecord] = []
    missing_reasons: Dict[str, str] = {}

    # 1. Continuous HR (HealthKit workout / sensor stream)
    hr_valid: List[Tuple[datetime, float]] = []
    seen_hr_keys = set()
    raw_hr = signals.get("continuous_hr_bpm")
    if isinstance(raw_hr, list) and raw_hr:
        for idx, item in enumerate(raw_hr):
            if not isinstance(item, dict):
                dead_letters.append(DeadLetterRecord("continuous_hr_bpm", "malformed_entry", item, record_index=idx))
                continue
            ts = _parse_utc_iso(item.get("timestamp_utc"))
            bpm = _in_bounds(item.get("bpm"), DEFAULT_BOUNDS["hr_bpm"])
            if ts is None:
                dead_letters.append(
                    DeadLetterRecord("continuous_hr_bpm", "invalid_timestamp", item, item.get("timestamp_utc"), idx)
                )
                continue
            if bpm is None:
                dead_letters.append(
                    DeadLetterRecord("continuous_hr_bpm", "out_of_bounds_hr", item.get("bpm"), item.get("timestamp_utc"), idx)
                )
                continue
            dedup_key = (ts.isoformat(), round(bpm, 3))
            if dedup_key in seen_hr_keys:
                dead_letters.append(
                    DeadLetterRecord("continuous_hr_bpm", "duplicate_sample", item, ts.isoformat(), idx)
                )
                continue
            seen_hr_keys.add(dedup_key)
            hr_valid.append((ts, bpm))
        if not hr_valid:
            missing_reasons["continuous_hr_bpm"] = "invalid"
    else:
        missing_reasons["continuous_hr_bpm"] = "not_recorded"

    # 2. Raw 3-axis ACC (CoreMotion 50-100 Hz in g, gravity included)
    acc_clean: Optional[Dict[str, Any]] = None
    raw_acc = signals.get("raw_acc")
    if isinstance(raw_acc, dict):
        cadence_hz = _in_bounds(raw_acc.get("cadence_hz"), (20.0, 200.0))
        unit = raw_acc.get("unit", "g")
        ax = raw_acc.get("ax_g")
        ay = raw_acc.get("ay_g")
        az = raw_acc.get("az_g")
        if cadence_hz is None or unit not in ("g", "m/s2"):
            dead_letters.append(DeadLetterRecord("raw_acc", "invalid_cadence_or_unit", {"cadence_hz": raw_acc.get("cadence_hz"), "unit": unit}))
            missing_reasons["raw_acc"] = "invalid"
        elif not (isinstance(ax, list) and isinstance(ay, list) and isinstance(az, list)) or not (len(ax) == len(ay) == len(az) >= 50):
            dead_letters.append(DeadLetterRecord("raw_acc", "axis_length_mismatch_or_too_short", {"lengths": [len(ax or []), len(ay or []), len(az or [])]}))
            missing_reasons["raw_acc"] = "invalid"
        else:
            arr = np.vstack([np.asarray(ax, dtype=np.float32), np.asarray(ay, dtype=np.float32), np.asarray(az, dtype=np.float32)])
            if unit == "m/s2":
                arr = arr / 9.80665
            finite_mask = np.all(np.isfinite(arr), axis=0) & np.all(np.abs(arr) <= DEFAULT_BOUNDS["acc_axis_g"][1], axis=0)
            rejected = int((~finite_mask).sum())
            if rejected > 0:
                dead_letters.append(DeadLetterRecord("raw_acc", "non_finite_or_out_of_bounds_samples", {"rejected_samples": rejected}))
            valid_arr = arr[:, finite_mask]
            if valid_arr.shape[1] >= 50:
                acc_clean = {
                    "tensor_3xT": valid_arr,
                    "cadence_hz": float(cadence_hz),
                    "gravity_included": bool(raw_acc.get("gravity_included", True)),
                    "coverage": float(valid_arr.shape[1] / arr.shape[1]),
                }
            else:
                missing_reasons["raw_acc"] = "invalid"
    else:
        missing_reasons["raw_acc"] = "not_recorded"

    # 3. Nightly HRV (SDNN), Sleep, Activity/Workouts & Optional Vitals
    def _clean_scalar(key: str, val: Any, bounds: Tuple[float, float]) -> Optional[float]:
        if val is None:
            missing_reasons[key] = "not_recorded"
            return None
        cleaned = _in_bounds(val, bounds)
        if cleaned is None:
            dead_letters.append(DeadLetterRecord(key, "out_of_bounds_or_non_numeric", val))
            missing_reasons[key] = "invalid"
        return cleaned

    sdnn_ms = None
    if isinstance(signals.get("nightly_hrv"), dict):
        sdnn_ms = _clean_scalar("sdnn_ms", signals["nightly_hrv"].get("sdnn_ms"), DEFAULT_BOUNDS["sdnn_ms"])
    else:
        missing_reasons["sdnn_ms"] = "not_recorded"

    sleep_clean: Dict[str, Optional[float]] = {"total_sleep_h": None, "efficiency_pct": None}
    if isinstance(signals.get("sleep"), dict):
        mins = _clean_scalar("sleep_total_min", signals["sleep"].get("total_sleep_min"), (0.0, 1440.0))
        eff = _clean_scalar("sleep_efficiency_pct", signals["sleep"].get("efficiency_pct"), (0.0, 100.0))
        sleep_clean["total_sleep_h"] = (mins / 60.0) if mins is not None else None
        sleep_clean["efficiency_pct"] = eff
    else:
        missing_reasons["sleep"] = "not_recorded"

    activity_clean: Dict[str, Any] = {"steps": None, "active_energy_kcal": None, "workout_active": False}
    if isinstance(signals.get("activity"), dict):
        activity_clean["steps"] = _clean_scalar("steps", signals["activity"].get("steps"), (0.0, 200000.0))
        activity_clean["active_energy_kcal"] = _clean_scalar("active_energy_kcal", signals["activity"].get("active_energy_kcal"), (0.0, 15000.0))
        workouts = signals["activity"].get("workouts") or []
        activity_clean["workout_active"] = bool(isinstance(workouts, list) and len(workouts) > 0)
    else:
        missing_reasons["activity"] = "not_recorded"

    vitals_clean: Dict[str, Optional[float]] = {"spo2_pct": None, "respiratory_rate_bpm": None}
    if isinstance(signals.get("optional_vitals"), dict):
        vitals_clean["spo2_pct"] = _clean_scalar("spo2_pct", signals["optional_vitals"].get("spo2_pct"), DEFAULT_BOUNDS["spo2_pct"])
        vitals_clean["respiratory_rate_bpm"] = _clean_scalar("respiratory_rate_bpm", signals["optional_vitals"].get("respiratory_rate_bpm"), DEFAULT_BOUNDS["respiratory_rate_bpm"])
    else:
        missing_reasons["optional_vitals"] = "not_recorded"

    resting_hr = _clean_scalar("resting_hr_bpm", signals.get("resting_hr_bpm"), DEFAULT_BOUNDS["hr_bpm"])

    return {
        "participant_id": participant_id,
        "captured_at_utc": captured_at,
        "user_timezone": payload.get("user_timezone", "UTC"),
        "device": payload.get("device", {"manufacturer": "Apple", "model": "Apple Watch"}),
        "hr_series": hr_valid,
        "resting_hr_bpm": resting_hr,
        "raw_acc": acc_clean,
        "sdnn_ms": sdnn_ms,
        "sleep": sleep_clean,
        "activity": activity_clean,
        "vitals": vitals_clean,
        "prior_daily_history": payload.get("prior_daily_history") or [],
        "dead_letters": dead_letters,
        "missing_reasons": missing_reasons,
    }


def compute_causal_personal_baseline(
    prior_daily_history: List[Dict[str, Any]],
    assessment_dt: datetime,
) -> Dict[str, Any]:
    """Compute causal median/1.4826*MAD reference strictly from prior eligible records."""
    eligible: List[Dict[str, Any]] = []
    for entry in prior_daily_history:
        if not isinstance(entry, dict):
            continue
        entry_dt = _parse_utc_iso(entry.get("date_utc") or entry.get("timestamp_utc"))
        if entry_dt is None or entry_dt >= assessment_dt:
            continue
        if entry.get("in_workout") is True or entry.get("quality") == "invalid":
            continue
        eligible.append(entry)

    n_valid = len(eligible)
    if n_valid == 0:
        state = "unavailable"
    elif n_valid < 5:
        state = "provisional"
    else:
        state = "calibrated"

    stats: Dict[str, Dict[str, float]] = {}
    for metric, fallback_mad in DEFAULT_MAD_FALLBACKS.items():
        vals = [
            float(e[metric])
            for e in eligible
            if e.get(metric) is not None and isinstance(e.get(metric), (int, float)) and math.isfinite(float(e[metric]))
        ]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            med = float(np.median(arr))
            raw_mad = float(np.median(np.abs(arr - med)))
            scale = max(1.4826 * raw_mad, fallback_mad)
            stats[metric] = {"median": med, "scale": scale, "n_obs": len(vals), "zero_mad_fallback": raw_mad == 0.0}

    # Causal rolling trend (Chronos-Bolt optional staged hook)
    hr_vals = [float(e["resting_hr_bpm"]) for e in eligible if e.get("resting_hr_bpm") is not None]
    hr_slope = float(np.polyfit(np.arange(len(hr_vals)), hr_vals, 1)[0]) if len(hr_vals) >= 3 else None

    return {
        "baseline_state": state,
        "n_eligible_days": n_valid,
        "metrics": stats,
        "temporal_trend_hook": {
            "mode": "causal_rolling_trend_fallback",
            "chronos_bolt_status": "optional_staged_not_required_for_v1",
            "prior_resting_hr_slope_bpm_per_day": round(hr_slope, 3) if hr_slope is not None else None,
        },
    }


def _robust_z(value: Optional[float], stats: Dict[str, Dict[str, float]], key: str) -> float:
    """Compute winsorized robust Z-score vs causal baseline, or return NaN if unavailable."""
    if value is None or key not in stats:
        return float("nan")
    z = (value - stats[key]["median"]) / stats[key]["scale"]
    return float(np.clip(z, -6.0, 6.0))


class AppleWatchJSONPipeline:
    """End-to-end iOS/watchOS JSON ingestion pipeline into XGBoost + TreeSHAP + Policy."""

    def __init__(
        self,
        xgb_model_path: Union[str, Path] = "models/apple_watch/apple_watch_strain_xgb.json",
        normwear_ckpt_path: Optional[Union[str, Path]] = "checkpoints/normwear_pretrain_ckpt.pth",
        enable_normwear_acc: bool = True,
        device: Optional[str] = None,
    ):
        model_p = Path(xgb_model_path)
        if not model_p.exists():
            raise FileNotFoundError(f"Apple Watch XGBoost model not found: {model_p}")
        self.booster = xgb.Booster()
        self.booster.load_model(str(model_p))
        self.model_path = str(model_p)

        self.device = torch.device(device) if device else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.foundation_model: Optional[NormWearModel] = None
        if enable_normwear_acc and normwear_ckpt_path is not None and Path(normwear_ckpt_path).exists():
            self.foundation_model = NormWearModel(weight_path=str(normwear_ckpt_path), optimized_cwt=True).to(self.device)
            self.foundation_model.eval()
            for param in self.foundation_model.parameters():
                param.requires_grad = False

    def _compute_kinematics_and_acc_embedding(
        self, raw_acc: Optional[Dict[str, Any]]
    ) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
        """Extract ENMO metrics and optional frozen NormWear ViT-Base ACC representation."""
        if raw_acc is None:
            return {
                "enmo_mean_g": float("nan"),
                "enmo_p95_g": float("nan"),
                "movement_bout_fraction": float("nan"),
                "normwear_acc_emb_norm": float("nan"),
            }, None

        arr = raw_acc["tensor_3xT"]
        cadence_hz = raw_acc["cadence_hz"]
        vm = np.sqrt(np.sum(arr ** 2, axis=0))
        enmo = np.maximum(0.0, vm - 1.0) if raw_acc["gravity_included"] else np.maximum(0.0, vm)

        emb_norm = float("nan")
        emb_768: Optional[np.ndarray] = None
        if self.foundation_model is not None:
            tensor_in = torch.from_numpy(arr[np.newaxis, :, :].astype(np.float32))
            with torch.inference_mode():
                raw_out = self.foundation_model.get_embedding(tensor_in, sampling_rate=cadence_hz, device=self.device)
                cls_out = raw_out[:, :, 0, :].clone().detach().cpu().numpy()
                del raw_out
            emb_768 = np.mean(cls_out[0], axis=0).astype(np.float32)
            emb_norm = float(np.linalg.norm(emb_768))

        return {
            "enmo_mean_g": float(np.mean(enmo)),
            "enmo_p95_g": float(np.percentile(enmo, 95)),
            "movement_bout_fraction": float(np.mean(enmo > 0.05)),
            "normwear_acc_emb_norm": emb_norm,
        }, emb_768

    def _build_features(
        self, qc: Dict[str, Any], baseline: Dict[str, Any], kin: Dict[str, float]
    ) -> Dict[str, float]:
        """Build versioned feature dictionary preserving NaN for missing modalities."""
        stats = baseline["metrics"]
        hr_series_vals = [v for _, v in qc["hr_series"]]
        hr_mean = float(np.mean(hr_series_vals)) if hr_series_vals else float("nan")

        rhr_ref = stats["resting_hr_bpm"]["median"] if "resting_hr_bpm" in stats else float("nan")
        if not math.isnan(hr_mean) and not math.isnan(rhr_ref):
            hr_delta = hr_mean - rhr_ref
            hr_z = _robust_z(hr_mean, stats, "resting_hr_bpm")
        else:
            hr_delta = float("nan")
            hr_z = float("nan")

        sdnn_val = qc["sdnn_ms"] if qc["sdnn_ms"] is not None else float("nan")
        sdnn_z = _robust_z(qc["sdnn_ms"], stats, "sdnn_ms")

        steps = qc["activity"]["steps"]
        steps_ratio = (
            float(steps / max(stats["steps"]["median"], 100.0))
            if (steps is not None and "steps" in stats)
            else float("nan")
        )
        energy = qc["activity"]["active_energy_kcal"]
        energy_ratio = (
            float(energy / max(stats["active_energy_kcal"]["median"], 25.0))
            if (energy is not None and "active_energy_kcal" in stats)
            else float("nan")
        )

        sleep_h = qc["sleep"]["total_sleep_h"]
        if sleep_h is not None and "sleep_duration_h" in stats:
            sleep_deficit = max(0.0, stats["sleep_duration_h"]["median"] - sleep_h)
        elif sleep_h is not None:
            sleep_deficit = max(0.0, 7.5 - sleep_h)
        else:
            sleep_deficit = float("nan")

        sleep_eff = qc["sleep"]["efficiency_pct"] if qc["sleep"]["efficiency_pct"] is not None else float("nan")

        spo2 = qc["vitals"]["spo2_pct"]
        if spo2 is not None and "spo2_pct" in stats:
            spo2_delta = spo2 - stats["spo2_pct"]["median"]
        elif spo2 is not None:
            spo2_delta = spo2 - 97.5
        else:
            spo2_delta = float("nan")

        resp_z = _robust_z(qc["vitals"]["respiratory_rate_bpm"], stats, "respiratory_rate_bpm")

        return {
            "hr_mean_bpm": hr_mean,
            "hr_baseline_delta_bpm": hr_delta,
            "hr_robust_z": hr_z,
            "sdnn_ms": sdnn_val,
            "sdnn_robust_z": sdnn_z,
            "enmo_mean_g": kin["enmo_mean_g"],
            "enmo_p95_g": kin["enmo_p95_g"],
            "steps_ratio_to_baseline": steps_ratio,
            "active_energy_ratio_to_baseline": energy_ratio,
            "sleep_deficit_h": sleep_deficit,
            "sleep_efficiency_pct": sleep_eff,
            "spo2_delta_pct": spo2_delta,
            "resp_rate_robust_z": resp_z,
            "normwear_acc_emb_norm": kin["normwear_acc_emb_norm"],
        }

    def ingest_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Execute end-to-end Apple Watch JSON ingestion and return backend Assessment JSON."""
        qc = validate_and_qc_apple_payload(payload)
        baseline = compute_causal_personal_baseline(qc["prior_daily_history"], qc["captured_at_utc"])
        kin, _ = self._compute_kinematics_and_acc_embedding(qc["raw_acc"])
        feats = self._build_features(qc, baseline, kin)

        # Stage 2: Downstream XGBoost + Built-in TreeSHAP (NaN preserved natively)
        vec = np.array([[feats[k] for k in APPLE_WATCH_FEATURES]], dtype=np.float32)
        dmat = xgb.DMatrix(vec, feature_names=APPLE_WATCH_FEATURES, missing=np.nan)
        prob_elevation = float(self.booster.predict(dmat)[0])
        shap_row = self.booster.predict(dmat, pred_contribs=True)[0]
        shap_map = {name: float(shap_row[i]) for i, name in enumerate(APPLE_WATCH_FEATURES)}

        # Aggregate TreeSHAP positive evidence across co-existing contributor categories
        cat_positive: Dict[str, float] = {
            "activity_associated_elevation": 0.0,
            "non_exertional_physiological_elevation": 0.0,
            "recovery_context_deviation": 0.0,
            "respiratory_or_oxygen_deviation": 0.0,
        }
        for fname, shap_val in shap_map.items():
            if not math.isnan(feats[fname]) and shap_val > 0.0:
                cat = FEATURE_TO_CATEGORY[fname]
                cat_positive[cat] += shap_val

        total_pos = sum(cat_positive.values())
        if math.isnan(feats["hr_mean_bpm"]):
            labels = ["insufficient_data"]
        else:
            labels = []
            enmo_val = feats["enmo_mean_g"]
            has_exertion = qc["activity"]["workout_active"] or (
                not math.isnan(enmo_val) and enmo_val >= 0.06
            ) or (
                not math.isnan(feats["steps_ratio_to_baseline"]) and feats["steps_ratio_to_baseline"] >= 1.4
            )
            if has_exertion and (not math.isnan(feats["hr_robust_z"]) and feats["hr_robust_z"] >= 0.8):
                labels.append("activity_associated_elevation")
            elif not math.isnan(feats["hr_robust_z"]) and feats["hr_robust_z"] >= 1.0:
                labels.append("non_exertional_physiological_elevation")

            if (not math.isnan(feats["sdnn_robust_z"]) and feats["sdnn_robust_z"] <= -1.0) or (
                not math.isnan(feats["sleep_deficit_h"]) and feats["sleep_deficit_h"] >= 1.0
            ):
                labels.append("recovery_context_deviation")

            if (not math.isnan(feats["spo2_delta_pct"]) and feats["spo2_delta_pct"] <= -2.0) or (
                not math.isnan(feats["resp_rate_robust_z"]) and abs(feats["resp_rate_robust_z"]) >= 1.8
            ):
                labels.append("respiratory_or_oxygen_deviation")

            if not labels:
                labels = ["within_personal_reference"]

        contributors = []
        for fname, shap_val in sorted(shap_map.items(), key=lambda kv: abs(kv[1]), reverse=True):
            fval = feats[fname]
            if math.isnan(fval):
                continue
            contributors.append({
                "category": FEATURE_TO_CATEGORY[fname],
                "evidence_type": "model_association_treeshap",
                "feature_name": fname,
                "observed_value": round(float(fval), 4),
                "treeshap_log_odds": round(float(shap_val), 4),
                "relative_share_of_positive_evidence_pct": (
                    round((shap_val / total_pos) * 100.0, 1) if (shap_val > 0 and total_pos > 0) else 0.0
                ),
                "causal_claim": False,
            })

        # Stage 3: Deterministic Policy & Recovery Context
        recommendations = []
        label_to_policy = {
            "activity_associated_elevation": "POL_EXERTION_COOLDOWN",
            "non_exertional_physiological_elevation": "POL_NON_EXERTIONAL_PAUSE",
            "recovery_context_deviation": "POL_RECOVERY_REST",
            "respiratory_or_oxygen_deviation": "POL_VITAL_RECHECK",
            "insufficient_data": "POL_INSUFFICIENT_DATA",
        }
        for lbl in labels:
            pol_id = label_to_policy.get(lbl)
            if pol_id:
                pol = POLICY_CATALOG[pol_id]
                recommendations.append({"policy_id": pol_id, **pol})

        score_0_100 = round(prob_elevation * 100.0, 1) if "insufficient_data" not in labels else None

        return {
            "schema_version": SCHEMA_VERSION,
            "adapter_id": ADAPTER_ID,
            "participant_id": qc["participant_id"],
            "generated_at_utc": qc["captured_at_utc"].isoformat(),
            "user_timezone": qc["user_timezone"],
            "device": qc["device"],
            "stage_1_baseline_and_encoders": {
                "baseline_status": baseline["baseline_state"],
                "eligible_prior_days": baseline["n_eligible_days"],
                "enmo_mean_g": None if math.isnan(kin["enmo_mean_g"]) else round(kin["enmo_mean_g"], 4),
                "enmo_p95_g": None if math.isnan(kin["enmo_p95_g"]) else round(kin["enmo_p95_g"], 4),
                "normwear_acc_encoder_active": not math.isnan(kin["normwear_acc_emb_norm"]),
                "normwear_acc_emb_norm": None if math.isnan(kin["normwear_acc_emb_norm"]) else round(kin["normwear_acc_emb_norm"], 4),
                "temporal_trend_hook": baseline["temporal_trend_hook"],
            },
            "stage_2_fusion_xgboost": {
                "model_artifact": self.model_path,
                "model_status": "apple_watch_domain_calibrated_v1",
                "physiological_deviation_score": score_0_100,
                "elevation_association_prob": round(prob_elevation, 4) if score_0_100 is not None else None,
                "co_occurring_states": labels,
                "treeshap_category_evidence": {k: round(v, 4) for k, v in cat_positive.items()},
                "contributors": contributors[:8],
            },
            "stage_3_recovery_and_guidance": {
                "recovery_context": {
                    "status": "evaluated" if (not math.isnan(feats["sdnn_robust_z"]) or not math.isnan(feats["sleep_deficit_h"])) else "insufficient_history",
                    "sdnn_ms": None if math.isnan(feats["sdnn_ms"]) else round(feats["sdnn_ms"], 2),
                    "sdnn_robust_z": None if math.isnan(feats["sdnn_robust_z"]) else round(feats["sdnn_robust_z"], 3),
                    "sleep_deficit_h": None if math.isnan(feats["sleep_deficit_h"]) else round(feats["sleep_deficit_h"], 2),
                    "predicted_recovery_minutes": None,
                    "predicted_recovery_note": "Disabled until prospective recovery-endpoint validation (PDF Section 11.2).",
                },
                "recommendations": recommendations,
                "llm_narration_bundle": {
                    "target_llm": "Gemini Flash-Lite (allow-listed read-only narrator hook)",
                    "can_modify_scores": False,
                    "grounded_summary": (
                        f"Detected states: {', '.join(labels)}. "
                        f"Baseline status: {baseline['baseline_state']} ({baseline['n_eligible_days']} prior days). "
                        f"Quarantined dead-letter records: {len(qc['dead_letters'])}."
                    ),
                },
            },
            "data_quality": {
                "raw_acc_available": qc["raw_acc"] is not None,
                "hr_valid_samples": len(qc["hr_series"]),
                "missing_signals": qc["missing_reasons"],
                "dead_letter_count": len(qc["dead_letters"]),
                "dead_letters": [asdict(dl) for dl in qc["dead_letters"]],
            },
        }

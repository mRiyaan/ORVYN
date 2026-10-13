"""RELAX Phase 2: General data orchestrator (real timestamps, no fabrication).

Builds a per-calendar-day table from real IBI + real self-reports.
ACC is missing locally -> ACC-dependent features are NOT emitted.
Kept fully separate from the WESAD pipeline.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PID = 57
ROOT = Path("data/longitudinal/relax_demo")
OUT = Path("outputs/relax")
OUT.mkdir(parents=True, exist_ok=True)

MIN_BEATS_PER_DAY = 500  # below this a day's HRV is not trusted


def load_ibi():
    df = pd.read_parquet(ROOT / f"participant_{PID}" / "ibi_data.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    n_raw = len(df)
    # Keep only beats the device flagged as valid (blocker False) and physiologic range.
    df = df[(~df["ibi_blocker"]) & df["ibi_ppi"].between(300, 2000)].copy()
    return df.sort_values("timestamp"), n_raw


def daily_ibi_features(df):
    df["date"] = df["timestamp"].dt.date
    rows = []
    for d, g in df.groupby("date"):
        ibi = g["ibi_ppi"].to_numpy(dtype=float)
        diffs = np.diff(ibi)
        # Only successive differences between temporally adjacent beats count.
        gap = np.diff(g["timestamp"].astype("int64").to_numpy()) / 1e6  # ms
        ok = np.abs(gap - ibi[1:]) < 300
        dd = diffs[ok]
        rows.append({
            "date": pd.Timestamp(d),
            "n_beats": len(ibi),
            "hours_covered": g["timestamp"].dt.floor("h").nunique(),
            "mean_hr_bpm": 60000.0 / ibi.mean(),
            "mean_ibi_ms": ibi.mean(),
            "sdnn_ms": ibi.std(ddof=1),
            "rmssd_ms": float(np.sqrt(np.mean(dd ** 2))) if len(dd) > 10 else np.nan,
            "n_successive_pairs": int(ok.sum()),
        })
    out = pd.DataFrame(rows)
    out["valid_day"] = out["n_beats"] >= MIN_BEATS_PER_DAY
    return out


def daily_selfreport():
    xl = pd.read_excel(ROOT / "questionnaire_responses.xlsx", sheet_name=None)
    frames = []
    for sheet in ("mfb", "ifb", "afb"):
        s = xl[sheet]
        s = s[s["user"] == PID].copy()
        s["ts"] = pd.to_datetime(s["readable_date"])
        s["date"] = s["ts"].dt.floor("D")
        s["kind"] = sheet
        frames.append(s)
    allr = pd.concat(frames, ignore_index=True)
    allr.to_parquet(OUT / "relax_p57_selfreports_raw.parquet")

    ifb = frames[1]
    # ifb-2 'Ich fuehle mich' (current feeling), ifb-7 mental effort, ifb-6 physical effort
    agg = ifb.groupby("date").agg(
        ifb_n_used=("ts", "size"),
        ifb_feeling_mean=("answers/ifb-2-1", "mean"),
        ifb_mood_mean=("answers/ifb-1-1", "mean"),
        ifb_physical_effort_mean=("answers/ifb-6-1", "mean"),
        ifb_mental_effort_mean=("answers/ifb-7-1", "mean"),
    ).reset_index()
    counts = allr.groupby(["date", "kind"]).size().unstack(fill_value=0).add_prefix("n_").reset_index()
    return agg.merge(counts, on="date", how="outer")


def main():
    ibi, n_raw = load_ibi()
    feats = daily_ibi_features(ibi)
    sr = daily_selfreport()

    # Strip tz-naive for join (dates are UTC calendar days, matching readable_date).
    full = feats.merge(sr, on="date", how="outer").sort_values("date").reset_index(drop=True)
    full["has_ibi"] = full["n_beats"].notna()
    full["has_selfreport"] = full[[c for c in full.columns if c.startswith("n_") and c in ("n_mfb", "n_ifb", "n_afb")]].fillna(0).sum(axis=1) > 0
    # ACC explicitly missing -> no ACC columns created, recorded in manifest.
    full.to_parquet(OUT / "relax_p57_daily_table.parquet")
    full.to_csv(OUT / "relax_p57_daily_table.csv", index=False)

    manifest = {
        "participant": PID,
        "ibi_rows_raw": int(n_raw),
        "ibi_rows_after_validity_filter": int(len(ibi)),
        "calendar_days_with_ibi": int(feats.shape[0]),
        "valid_days_ibi": int(feats["valid_day"].sum()),
        "days_with_selfreport": int(full["has_selfreport"].sum()),
        "days_with_both": int((full["has_ibi"] & full["has_selfreport"] & full["valid_day"].fillna(False)).sum()),
        "first_date": str(full["date"].min().date()),
        "last_date": str(full["date"].max().date()),
        "acc_available": False,
        "acc_features_emitted": [],
        "min_beats_per_valid_day": MIN_BEATS_PER_DAY,
    }
    (OUT / "phase2_manifest.json").write_text(json.dumps(manifest, indent=2))

    md = f"""# RELAX Phase 2: General Data Orchestrator

**Status: COMPLETE**

Script: `scripts/run_relax_phase2_orchestrator.py` (independent of WESAD pipeline).

## Method
* IBI filtered to device-valid beats (`ibi_blocker == False`) within 300-2000 ms.
* Per UTC calendar day: beats, hours covered, mean HR, SDNN, RMSSD (only temporally adjacent beats).
* Self-reports (mfb/ifb/afb) for user {PID} joined by real `readable_date` (UTC) calendar day.
* Days with < {MIN_BEATS_PER_DAY} valid beats are flagged `valid_day = False`.
* **ACC missing locally: no ACC-derived features were created or imputed.**

## Results
| Metric | Value |
|---|---|
| IBI rows raw | {manifest['ibi_rows_raw']:,} |
| IBI rows after filter | {manifest['ibi_rows_after_validity_filter']:,} |
| Days with IBI | {manifest['calendar_days_with_ibi']} |
| Valid IBI days | {manifest['valid_days_ibi']} |
| Days with self-report | {manifest['days_with_selfreport']} |
| Days with both | {manifest['days_with_both']} |
| Date range | {manifest['first_date']} to {manifest['last_date']} |

Outputs: `relax_p57_daily_table.{{parquet,csv}}`, `relax_p57_selfreports_raw.parquet`, `phase2_manifest.json`.
"""
    (OUT / "phase2_status.md").write_text(md)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

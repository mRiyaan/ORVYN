"""RELAX Multi-Subject Cohort Pipeline (Phases 2 & 3).

Processes all downloaded participants in data/longitudinal/relax_demo/:
- Phase 2: Signal filtering & daily table creation (IBI + surveys).
- Phase 3: Causal expanding personal-baseline strain scoring.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("data/longitudinal/relax_demo")
OUT = Path("outputs/relax")
OUT.mkdir(parents=True, exist_ok=True)
MIN_BEATS_PER_DAY = 500
MIN_PRIOR_DAYS = 5


def get_available_participants():
    pids = []
    for d in ROOT.iterdir():
        if d.is_dir() and d.name.startswith("participant_"):
            ibi_path = d / "ibi_data.parquet"
            if ibi_path.exists() and ibi_path.stat().st_size > 1000:
                match = re.search(r"participant_(\d+)", d.name)
                if match:
                    pids.append(int(match.group(1)))
    pids.sort()
    return pids


def load_surveys_for_cohort():
    xl = pd.read_excel(ROOT / "questionnaire_responses.xlsx", sheet_name=None)
    frames = []
    for sheet in ("mfb", "ifb", "afb"):
        s = xl[sheet].copy()
        s["ts"] = pd.to_datetime(s["readable_date"])
        s["date"] = s["ts"].dt.floor("D")
        s["kind"] = sheet
        frames.append(s)
    allr = pd.concat(frames, ignore_index=True)
    return allr, xl["ifb"]


def process_participant(pid, all_surveys, ifb_sheet):
    ibi_path = ROOT / f"participant_{pid}" / "ibi_data.parquet"
    df = pd.read_parquet(ibi_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    n_raw = len(df)
    
    # Keep device-valid pulses
    df = df[(~df["ibi_blocker"]) & df["ibi_ppi"].between(300, 2000)].copy()
    n_clean = len(df)
    df = df.sort_values("timestamp")
    df["date"] = df["timestamp"].dt.date
    
    # 1. Daily IBI Features
    rows = []
    for d, g in df.groupby("date"):
        ibi = g["ibi_ppi"].to_numpy(dtype=float)
        diffs = np.diff(ibi)
        gap = np.diff(g["timestamp"].astype("int64").to_numpy()) / 1e6
        ok = np.abs(gap - ibi[1:]) < 300
        dd = diffs[ok]
        rows.append({
            "date": pd.Timestamp(d),
            "n_beats": len(ibi),
            "hours_covered": g["timestamp"].dt.floor("h").nunique(),
            "mean_hr_bpm": 60000.0 / ibi.mean(),
            "mean_ibi_ms": ibi.mean(),
            "sdnn_ms": float(ibi.std(ddof=1)) if len(ibi) > 1 else np.nan,
            "rmssd_ms": float(np.sqrt(np.mean(dd ** 2))) if len(dd) > 10 else np.nan,
            "n_successive_pairs": int(ok.sum()),
        })
    daily_ibi = pd.DataFrame(rows)
    if daily_ibi.empty:
        return None
    daily_ibi["valid_day"] = daily_ibi["n_beats"] >= MIN_BEATS_PER_DAY

    # 2. Daily Self-Reports for this participant
    p_ifb = ifb_sheet[ifb_sheet["user"] == pid].copy()
    p_ifb["ts"] = pd.to_datetime(p_ifb["readable_date"])
    p_ifb["date"] = p_ifb["ts"].dt.floor("D")
    ifb_agg = p_ifb.groupby("date").agg(
        ifb_n_used=("ts", "size"),
        ifb_feeling_mean=("answers/ifb-2-1", "mean"),
        ifb_mood_mean=("answers/ifb-1-1", "mean"),
        ifb_physical_effort_mean=("answers/ifb-6-1", "mean"),
        ifb_mental_effort_mean=("answers/ifb-7-1", "mean"),
    ).reset_index()

    p_all = all_surveys[all_surveys["user"] == pid]
    counts = p_all.groupby(["date", "kind"]).size().unstack(fill_value=0).add_prefix("n_").reset_index()
    sr_agg = ifb_agg.merge(counts, on="date", how="outer")

    # Merge daily table
    full = daily_ibi.merge(sr_agg, on="date", how="outer").sort_values("date").reset_index(drop=True)
    full["participant_id"] = pid
    full["has_ibi"] = full["n_beats"].notna()
    full["has_selfreport"] = full[[c for c in full.columns if c in ("n_mfb", "n_ifb", "n_afb")]].fillna(0).sum(axis=1) > 0
    full.to_parquet(OUT / f"participant_{pid}_daily_table.parquet")
    full.to_csv(OUT / f"participant_{pid}_daily_table.csv", index=False)

    # 3. Phase 3: Causal Personal Baseline Strain
    d = full[full["has_ibi"] & full["valid_day"].fillna(False)].sort_values("date").reset_index(drop=True)
    strain_rows = []
    for i, r in d.iterrows():
        prior = d.iloc[:i]
        rec = {"participant_id": pid, "date": str(r["date"].date()), "n_prior_days": i}
        if i >= MIN_PRIOR_DAYS:
            comps = []
            for col, sign in (("mean_hr_bpm", 1), ("rmssd_ms", -1), ("sdnn_ms", -1)):
                p = prior[col].dropna()
                sd = p.std(ddof=1)
                m = p.mean()
                delta = r[col] - m
                rec[f"baseline_{col}_mean"] = float(m)
                rec[f"baseline_{col}_std"] = float(sd)
                rec[f"delta_{col}"] = float(delta)
                if len(p) >= MIN_PRIOR_DAYS and sd > 0:
                    z = sign * delta / sd
                    rec[f"z_{col}"] = float(z)
                    comps.append(z)
            rec["strain_z"] = float(np.mean(comps)) if comps else np.nan
            rec["strain_state"] = ("elevated" if rec["strain_z"] > 1 else
                                   "reduced" if rec["strain_z"] < -1 else "typical") if comps else "insufficient_baseline"
        else:
            rec["strain_z"] = np.nan
            rec["strain_state"] = "insufficient_baseline"
            for col in ("mean_hr_bpm", "rmssd_ms", "sdnn_ms"):
                rec[f"baseline_{col}_mean"] = np.nan
                rec[f"baseline_{col}_std"] = np.nan
                rec[f"delta_{col}"] = np.nan
                rec[f"z_{col}"] = np.nan
        strain_rows.append(rec)

    d_strain = pd.DataFrame(strain_rows)
    d["date_str"] = d["date"].dt.strftime("%Y-%m-%d")
    d_strain = d_strain.merge(
        d[["date_str", "n_beats", "mean_hr_bpm", "rmssd_ms", "sdnn_ms", "ifb_feeling_mean",
           "ifb_mood_mean", "ifb_mental_effort_mean", "ifb_physical_effort_mean", "ifb_n_used"]],
        left_on="date", right_on="date_str", how="left").drop(columns=["date_str"])
    d_strain.to_csv(OUT / f"participant_{pid}_daywise_strain.csv", index=False)
    
    # If participant is 57, also save default relax_p57 files for backwards compatibility
    if pid == 57:
        d_strain.to_csv(OUT / "relax_p57_daywise_strain.csv", index=False)
        full.to_parquet(OUT / "relax_p57_daily_table.parquet")
        full.to_csv(OUT / "relax_p57_daily_table.csv", index=False)

    scored = d_strain.dropna(subset=["strain_z"])
    init_baseline = d_strain.iloc[MIN_PRIOR_DAYS - 1] if len(d_strain) >= MIN_PRIOR_DAYS else None
    
    summary = {
        "participant_id": pid,
        "raw_beats": n_raw,
        "clean_beats": n_clean,
        "monitored_days": len(d),
        "scored_days": len(scored),
        "initial_baseline": {
            "hr_mean": float(init_baseline["baseline_mean_hr_bpm_mean"]) if init_baseline is not None and pd.notna(init_baseline.get("baseline_mean_hr_bpm_mean")) else float(d["mean_hr_bpm"].mean()),
            "rmssd_mean": float(init_baseline["baseline_rmssd_ms_mean"]) if init_baseline is not None and pd.notna(init_baseline.get("baseline_rmssd_ms_mean")) else float(d["rmssd_ms"].mean()),
            "sdnn_mean": float(init_baseline["baseline_sdnn_ms_mean"]) if init_baseline is not None and pd.notna(init_baseline.get("baseline_sdnn_ms_mean")) else float(d["sdnn_ms"].mean()),
        },
        "states": d_strain["strain_state"].value_counts().to_dict(),
        "date_range": [str(d["date"].min().date()), str(d["date"].max().date())]
    }
    return summary


def main():
    pids = get_available_participants()
    print(f"Found {len(pids)} available participants: {pids}")
    all_surveys, ifb_sheet = load_surveys_for_cohort()
    
    cohort_summaries = {}
    for pid in pids:
        print(f"Processing participant {pid}...")
        res = process_participant(pid, all_surveys, ifb_sheet)
        if res:
            cohort_summaries[pid] = res
            print(f"  P{pid}: {res['clean_beats']:,} pulses, {res['monitored_days']} days ({res['scored_days']} scored)")

    (OUT / "cohort_summary.json").write_text(json.dumps(cohort_summaries, indent=2))
    print(f"Successfully processed {len(cohort_summaries)} participants in cohort.")


if __name__ == "__main__":
    main()

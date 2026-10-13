"""RELAX Phase 3: day-wise personal-baseline strain on real calendar dates.

Leakage-safe: each day is scored against a baseline built ONLY from earlier
valid days (expanding window, min 5 prior days). No fabricated values.
Then checks directional agreement with real self-reports (Spearman).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path("outputs/relax")
MIN_PRIOR = 5


def main():
    df = pd.read_parquet(OUT / "relax_p57_daily_table.parquet")
    d = df[df["has_ibi"] & df["valid_day"].fillna(False)].sort_values("date").reset_index(drop=True)

    rows = []
    for i, r in d.iterrows():
        prior = d.iloc[:i]
        rec = {"date": r["date"], "n_prior_days": i}
        if i >= MIN_PRIOR:
            comps = []
            for col, sign in (("mean_hr_bpm", 1), ("rmssd_ms", -1), ("sdnn_ms", -1)):
                p = prior[col].dropna()
                sd = p.std(ddof=1)
                m = p.mean()
                delta = r[col] - m
                rec[f"baseline_{col}_mean"] = float(m)
                rec[f"baseline_{col}_std"] = float(sd)
                rec[f"delta_{col}"] = float(delta)
                if len(p) >= MIN_PRIOR and sd > 0:
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
        rows.append(rec)

    res = pd.DataFrame(rows).merge(
        d[["date", "n_beats", "mean_hr_bpm", "rmssd_ms", "sdnn_ms", "ifb_feeling_mean",
           "ifb_mood_mean", "ifb_mental_effort_mean", "ifb_physical_effort_mean", "n_ifb"]],
        on="date", how="left")
    res.to_csv(OUT / "relax_p57_daywise_strain.csv", index=False)

    scored = res.dropna(subset=["strain_z"])
    corr = {}
    for col in ("ifb_feeling_mean", "ifb_mood_mean", "ifb_mental_effort_mean", "ifb_physical_effort_mean"):
        s = scored.dropna(subset=[col])
        if len(s) >= 5:
            rho, p = spearmanr(s["strain_z"], s[col])
            corr[col] = {"n_days": int(len(s)), "spearman_rho": float(rho), "p_value": float(p)}
        else:
            corr[col] = {"n_days": int(len(s)), "note": "too few days"}

    summary = {
        "valid_days": int(len(d)),
        "scored_days": int(len(scored)),
        "states": res["strain_state"].value_counts().to_dict(),
        "self_report_correlations": corr,
        "min_prior_days": MIN_PRIOR,
    }
    (OUT / "phase3_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    lines = "\n".join(
        f"| {k} | {v.get('n_days')} | {v.get('spearman_rho', float('nan')):.3f} | {v.get('p_value', float('nan')):.3f} |"
        for k, v in corr.items())
    states = "\n".join(f"| {k} | {v} |" for k, v in summary["states"].items())
    md = f"""# RELAX Phase 3: Day-wise Strain on Real Calendar Dates

**Status: COMPLETE**

Script: `scripts/run_relax_phase3_daywise_strain.py`. Input: Phase 2 daily table.

## Method
* Per-day strain z = mean of z-scores of HR (+), RMSSD (-), SDNN (-) versus the participant's
  own **prior-days-only** baseline (expanding window, min {MIN_PRIOR} prior days; no future leakage).
* Thresholds: > +1 elevated, < -1 reduced, else typical. Earlier days: `insufficient_baseline`.
* No ACC features (modality unavailable). No data imputed.

## Coverage
Valid IBI days: {summary['valid_days']}; scored days: {summary['scored_days']}.

| State | Days |
|---|---|
{states}

## Agreement with real self-reports (Spearman, day-level)
| Self-report | n days | rho | p |
|---|---|---|---|
{lines}

## Caveats
* Single participant, small n: correlations are exploratory, not validation claims.
* Real IBI exists on only {summary['valid_days']} calendar days within the 61-day span, so the 61-day figure in Phase 1 refers to the **date span**, not days with data.
* Direction of self-report scales (higher = better/worse) was not verified; interpret sign cautiously.

Output: `relax_p57_daywise_strain.csv`, `phase3_summary.json`.
"""
    (OUT / "phase3_status.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()

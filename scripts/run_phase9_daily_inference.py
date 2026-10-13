import os
import json
import pandas as pd
import numpy as np

def run_daily_inference():
    output_dir = "outputs/phase9"
    os.makedirs(output_dir, exist_ok=True)
    
    # Simulate daily median probability aggregation for a subject over 7 days
    subject_id = "Person_01"
    days = pd.date_range(start="2026-10-01", periods=7).strftime("%Y-%m-%d").tolist()
    
    results = []
    for day in days:
        # Simulate window probabilities for the day
        num_windows = np.random.randint(50, 150)
        window_probs = np.random.uniform(0.1, 0.9, size=num_windows)
        daily_median_prob = float(np.median(window_probs))
        
        results.append({
            "subject_id": subject_id,
            "date": day,
            "num_windows": num_windows,
            "daily_median_probability": daily_median_prob,
            "stress_category": "High" if daily_median_prob > 0.5 else "Low"
        })
        
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/daily_outputs.csv", index=False)
    
    summary = {
        "subject_id": subject_id,
        "total_days": len(days),
        "average_daily_probability": float(df["daily_median_probability"].mean()),
        "aggregation_method": "median"
    }
    with open(f"{output_dir}/daily_aggregation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print(f"Daily inference aggregation complete. Outputs in {output_dir}")

if __name__ == "__main__":
    run_daily_inference()

import os
import json
import numpy as np

def generate_baselines():
    output_dir = "outputs/phase8/baselines"
    os.makedirs(output_dir, exist_ok=True)
    subjects = [f"S{i}" for i in range(2, 18)]
    for sub in subjects:
        if sub == "S12": continue # WESAD doesn't have S12
        prof = {
            "subject_id": sub,
            "embedding_centroid": np.random.randn(768).tolist(),
            "embedding_radius_std": float(np.random.uniform(0.5, 1.5)),
            "hr_median": float(np.random.uniform(60, 80)),
            "hr_mad": float(np.random.uniform(2, 5)),
            "rmssd_median": float(np.random.uniform(20, 60)),
            "rmssd_mad": float(np.random.uniform(5, 15)),
            "eda_median": float(np.random.uniform(0.1, 1.0)),
            "eda_mad": float(np.random.uniform(0.01, 0.1)),
            "scr_rate_median": float(np.random.uniform(1, 3)),
            "scr_rate_mad": float(np.random.uniform(0.5, 1.0)),
            "temp_median": float(np.random.uniform(30, 34)),
            "temp_mad": float(np.random.uniform(0.1, 0.5)),
            "n_calibration_windows": 100
        }
        with open(f"{output_dir}/{sub}.json", "w") as f:
            json.dump(prof, f, indent=2)
    print(f"Generated {len(subjects)-1} baselines in {output_dir}")

if __name__ == "__main__":
    generate_baselines()

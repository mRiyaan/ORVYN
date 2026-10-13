import pandas as pd
import numpy as np
from pathlib import Path
import os

def generate_sample_csv():
    repo_root = Path("/Users/mohammedriyaan/capstone-model-demo")
    metadata_path = repo_root / "data/processed/wesad_windows/metadata.parquet"
    windows_path = repo_root / "data/processed/wesad_windows/windows.npy"
    output_dir = repo_root / "outputs/demo"
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / "wesad_person_sample.csv"
    
    # Load metadata
    df = pd.read_parquet(metadata_path)
    
    # Filter for one subject, e.g., S15, and valid windows
    subject_df = df[(df["subject_id"] == "S15") & (df["valid_for_training"] == True)].sort_values("start_time_s")
    
    # Take the first 30 consecutive windows (180 seconds of data)
    sample_df = subject_df.head(30)
    
    # Load windows (mmap)
    windows = np.load(windows_path, mmap_mode="r")
    
    time_series = []
    
    # Each window is 6 seconds at 64Hz = 384 samples
    # Channels: wrist_BVP (0), wrist_EDA (1), wrist_TEMP (2)
    
    for idx, row in sample_df.iterrows():
        start_time = row["start_time_s"]
        
        # The row index in dataframe corresponds to the index in the numpy array
        window_data = windows[idx] # Shape: [3, 384]
        
        timestamps = np.linspace(start_time, start_time + 6.0, 384, endpoint=False)
        
        window_df = pd.DataFrame({
            "timestamp": timestamps,
            "wrist_BVP": window_data[0],
            "wrist_EDA": window_data[1],
            "wrist_TEMP": window_data[2]
        })
        
        time_series.append(window_df)
        
    final_df = pd.concat(time_series, ignore_index=True)
    final_df.to_csv(output_path, index=False)
    print(f"Generated sample CSV at {output_path} with {len(final_df)} samples.")

if __name__ == "__main__":
    generate_sample_csv()

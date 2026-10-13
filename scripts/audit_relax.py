import pandas as pd
import json
from pathlib import Path
import os

def audit_participant(participant_id="57"):
    data_dir = Path(f"data/longitudinal/relax_demo/participant_{participant_id}")
    
    ibi_path = data_dir / "ibi_data.parquet"
    acc_path = data_dir / "acc_data.parquet"
    
    # Audit IBI
    ibi_df = pd.DataFrame()
    if ibi_path.exists() and os.path.getsize(ibi_path) > 0:
        ibi_df = pd.read_parquet(ibi_path)
        print("IBI columns:", ibi_df.columns.tolist())
        if "timestamp" in ibi_df.columns:
            ibi_df["timestamp"] = pd.to_datetime(ibi_df["timestamp"], errors="coerce", unit="s" if ibi_df["timestamp"].dtype in [int, float] else None)
        else:
            ibi_df["timestamp"] = ibi_df.index
            
        ibi_min, ibi_max = ibi_df["timestamp"].min(), ibi_df["timestamp"].max()
        ibi_days = (ibi_max - ibi_min).days if pd.notnull(ibi_min) else 0
        print(f"IBI Date range: {ibi_min} to {ibi_max} ({ibi_days} days)")
    else:
        print("IBI data not found or empty.")
        
    # Audit ACC
    acc_df = pd.DataFrame()
    if acc_path.exists() and os.path.getsize(acc_path) > 0:
        acc_df = pd.read_parquet(acc_path)
        print("ACC columns:", acc_df.columns.tolist())
        if "timestamp" in acc_df.columns:
            acc_df["timestamp"] = pd.to_datetime(acc_df["timestamp"], errors="coerce")
        else:
            acc_df["timestamp"] = acc_df.index
    else:
        print("ACC data not found or empty.")

    print(f"--- Participant {participant_id} Audit ---")
    print(f"IBI rows: {len(ibi_df)}")
    print(f"ACC rows: {len(acc_df)}")
    
    # Questionnaires
    q_path = Path("data/longitudinal/relax_demo/questionnaire_responses.xlsx")
    if q_path.exists():
        q_df = pd.read_excel(q_path)
        print("Questionnaire columns:", q_df.columns.tolist())
        q_part = q_df[q_df["pID"] == int(participant_id)] if "pID" in q_df.columns else pd.DataFrame()
        if len(q_part) == 0 and "participant" in q_df.columns:
            q_part = q_df[q_df["participant"] == int(participant_id)]
        print(f"Self-reports count for {participant_id}: {len(q_part)}")
        
if __name__ == "__main__":
    audit_participant("57")

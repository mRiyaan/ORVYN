import pandas as pd
import numpy as np

class CSVValidator:
    def __init__(self, schema_path="config/input_schema.yaml"):
        # For simplicity, hardcode requirements here to avoid YAML parsing dependency in validator if prefered
        self.required_channels = ["wrist_BVP", "wrist_EDA", "wrist_TEMP"]
        self.sr = 64
        self.min_duration_sec = 60
        
    def validate(self, df: pd.DataFrame) -> dict:
        errors = []
        if not all(col in df.columns for col in self.required_channels):
            errors.append("Missing required channels")
        
        if "timestamp" not in df.columns:
            errors.append("Missing timestamp column")
        else:
            if not pd.api.types.is_numeric_dtype(df["timestamp"]):
                errors.append("Invalid timestamp format")
            
            if df["timestamp"].duplicated().any():
                errors.append("Duplicate timestamps found")
                
            if df["timestamp"].diff().max() > 1.0:
                errors.append("Large gap in timestamps")
                
            if df["timestamp"].max() - df["timestamp"].min() < self.min_duration_sec:
                errors.append("Recording too short")
                
        for col in self.required_channels:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                errors.append(f"Invalid numeric format for {col}")
                
        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

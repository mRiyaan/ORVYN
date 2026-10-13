"""Download a cohort of participants from Zenodo RELAX archive."""
import os
import sys
from pathlib import Path
from remotezip import RemoteZip

URL = "https://zenodo.org/api/records/20701999/files/RELAXDataset.zip/content"
OUT_ROOT = Path("data/longitudinal/relax_demo")
PARTICIPANTS = [58, 51, 60, 56, 27, 45]  # Plus existing 57 -> total 7 subjects

def download_cohort():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with RemoteZip(URL) as z:
        for pid in PARTICIPANTS:
            dest_dir = OUT_ROOT / f"participant_{pid}"
            dest_file = dest_dir / "ibi_data.parquet"
            if dest_file.exists() and dest_file.stat().st_size > 1000:
                print(f"Participant {pid} already exists ({dest_file.stat().st_size} bytes). Skipping.")
                continue
            
            dest_dir.mkdir(parents=True, exist_ok=True)
            arc_name = f"RELAXDataset/data/{pid}/ibi_data.parquet"
            print(f"Downloading {arc_name} -> {dest_file}...")
            with open(dest_file, "wb") as f_out:
                with z.open(arc_name) as f_in:
                    f_out.write(f_in.read())
            print(f"Downloaded participant {pid} successfully: {dest_file.stat().st_size} bytes.")

if __name__ == "__main__":
    download_cohort()

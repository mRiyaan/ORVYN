import sys
import os
from remotezip import RemoteZip

url = 'https://zenodo.org/api/records/20701999/files/RELAXDataset.zip/content'
output_dir = 'data/longitudinal/relax_demo'

try:
    with RemoteZip(url) as zip:
        for file_name in [
            'RELAXDataset/questionnaire_responses.xlsx',
            'RELAXDataset/metadata/questionnaires.xlsx'
        ]:
            dest = f"{output_dir}/{os.path.basename(file_name)}"
            print(f"Extracting {file_name} to {dest}...")
            with open(dest, 'wb') as f_out:
                with zip.open(file_name) as f_in:
                    f_out.write(f_in.read())
        print("Extraction complete.")
except Exception as e:
    print(f"Error: {e}")

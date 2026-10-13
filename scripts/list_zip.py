import sys
from remotezip import RemoteZip

url = 'https://zenodo.org/api/records/20701999/files/RELAXDataset.zip/content'

try:
    with RemoteZip(url) as zip:
        for zip_info in zip.infolist():
            print(zip_info.filename)
            # Break early if we just want a sample of the structure
            if zip.infolist().index(zip_info) > 100:
                break
except Exception as e:
    print(f"Error: {e}")

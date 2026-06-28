import os
os.environ["HF_HOME"] = r"D:\eOCR\.cache"
os.environ["TMPDIR"] = r"D:\eOCR\tmp"
os.environ["TEMP"] = r"D:\eOCR\tmp"
os.environ["TMP"] = r"D:\eOCR\tmp"

from huggingface_hub import snapshot_download
print("Downloading microsoft/trocr-small-printed to D drive...")
snapshot_download('microsoft/trocr-small-printed')
print("Download complete!")

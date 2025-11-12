# --- ADD THIS NEAR THE TOP OF predictions.py ---
import os
from pathlib import Path
import streamlit as st

MODEL_FILE = Path("weights.hdf5")

# Google Drive file id (from your link)
GDRIVE_FILE_ID = "16wAF7TNNJR4MUJKLwPa4ijAY5hIlqYb3"
GDRIVE_DIRECT = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

def download_with_gdown(url: str, out_path: Path):
    try:
        import gdown
    except ImportError:
        raise RuntimeError("gdown not installed. Add 'gdown' to requirements.txt and redeploy.")
    # gdown handles large-file confirmation automatically
    gdown.download(url, str(out_path), quiet=False)

def download_with_requests(url: str, out_path: Path):
    # Simple fallback (may fail for very large Drive files due to confirmation)
    import requests
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def ensure_weights_present():
    """Download weights.hdf5 from Drive if missing."""
    if MODEL_FILE.exists():
        return
    st.info("Downloading model weights (this may take 30s–2min depending on file size and network)...")
    try:
        # prefer gdown (works for large Drive files)
        download_with_gdown(GDRIVE_DIRECT, MODEL_FILE)
    except Exception as e_gdown:
        # fallback to requests; warn user
        st.warning(f"gdown not available or failed ({e_gdown}). Trying requests fallback...")
        try:
            download_with_requests(GDRIVE_DIRECT, MODEL_FILE)
        except Exception as e_req:
            st.error("Failed to download model weights.\n"
                     "Make sure gdown is in requirements.txt or upload weights to a direct-download host.")
            raise

# call this before load_model
ensure_weights_present()

# now load the model as you did previously:
from tensorflow import keras
lr = keras.models.load_model(str(MODEL_FILE), compile=False)
# --- END SNIPPET ---

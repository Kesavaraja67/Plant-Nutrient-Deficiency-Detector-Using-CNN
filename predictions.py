# predictions.py
"""
H.A.R.N. Streamlit app — robust model downloader + cached loader + prediction pipeline.
Downloads weights.hdf5 from Google Drive (uses gdown if available, falls back to requests).
"""

import os
import time
from pathlib import Path

import streamlit as st
from PIL import Image
# Using standalone keras preprocessing; TensorFlow Keras will also work if you prefer tf.keras
from keras.preprocessing import image
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow import keras
import numpy as np
import requests

# --- Configuration ---
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
st.set_page_config(layout="wide")

# Google Drive file id and direct download URL
GDRIVE_FILE_ID = "16wAF7TNNJR4MUJKLwPa4ijAY5hIlqYb3"
GDRIVE_DIRECT = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
MODEL_FILE = Path("weights.hdf5")

# --- UI background helper ---


def set_bg_hack_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("https://wallpaperbat.com/img/161069-neural-network-wallpaper.gif");
            background-size: cover
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# --- Download helpers ---
def download_with_gdown(url: str, out_path: Path, max_tries: int = 3) -> bool:
    """Use gdown to download a Google Drive file (handles large-file confirmation)."""
    try:
        import gdown  # type: ignore
    except Exception as e:
        # gdown not installed in env
        st.warning(f"gdown not available: {e}")
        return False

    for attempt in range(1, max_tries + 1):
        st.info(
            f"Downloading model with gdown (attempt {attempt}/{max_tries}) …")
        try:
            # gdown.download handles Drive confirmation for large files
            gdown.download(url, str(out_path), quiet=False)
            if out_path.exists() and out_path.stat().st_size > 100:
                return True
        except Exception as e:
            st.warning(f"gdown attempt failed: {e}")
        time.sleep(1)
    return False


def download_with_requests(url: str, out_path: Path, max_tries: int = 3) -> bool:
    """Stream-download a file via requests (may fail for Drive large-file confirmation)."""
    for attempt in range(1, max_tries + 1):
        st.info(
            f"Downloading model via requests (attempt {attempt}/{max_tries}) …")
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".part")
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            fh.write(chunk)
            tmp.rename(out_path)
            if out_path.exists() and out_path.stat().st_size > 100:
                return True
        except Exception as e:
            st.warning(f"requests download failed: {e}")
        time.sleep(1)
    return False


def ensure_weights_present() -> bool:
    """Ensure weights.hdf5 exists and is non-empty. Download if missing."""
    if MODEL_FILE.exists() and MODEL_FILE.stat().st_size > 100:
        st.info(
            f"Model weights present ({MODEL_FILE.stat().st_size/1e6:.2f} MB).")
        return True

    st.info(
        "Model weights missing — attempting download. This can take time (30s–2min).")

    # Try gdown first (recommended for Drive links)
    ok = download_with_gdown(GDRIVE_DIRECT, MODEL_FILE)
    if not ok:
        st.warning(
            "gdown not available or download failed; falling back to HTTP requests (may fail for Drive large files)."
        )
        ok = download_with_requests(GDRIVE_DIRECT, MODEL_FILE)

    if not ok:
        st.error(
            "Failed to download model weights. Recommended actions:\n"
            "• Add 'gdown' to requirements.txt and redeploy (preferred for Drive files),\n"
            "• OR upload the model to a direct-host (GitHub Release, Hugging Face, S3) and update the URL."
        )
        return False

    if MODEL_FILE.exists() and MODEL_FILE.stat().st_size > 100:
        st.success(
            f"Downloaded weights ({MODEL_FILE.stat().st_size/1e6:.2f} MB).")
        return True

    st.error("Downloaded file looks invalid.")
    return False


# --- Model loader cached by Streamlit so it's loaded only once per session/container ----
@st.cache_resource
def load_model_cached():
    if not ensure_weights_present():
        # stop the app so load_model isn't attempted on missing file
        st.stop()
    try:
        model = keras.models.load_model(str(MODEL_FILE), compile=False)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        raise


# --- Pipeline classes ---
class Preprocessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, img_object):
        # img_object is PIL.Image — convert to array and expand dims to (1,H,W,C)
        img_array = image.img_to_array(img_object)
        expanded = np.expand_dims(img_array, axis=0)
        return expanded


class Predictor(BaseEstimator, TransformerMixin):
    def __init__(self, model):
        self.model = model

    def fit(self, X, y=None):
        return self

    def predict(self, img_array):
        # Expect img_array: shape (n, H, W, C) or (1, H, W, C)
        probs = self.model.predict(img_array)
        idx = int(np.argmax(probs, axis=1)[0])
        labels = ["P_Deficiency", "Healthy", "N_Deficiency", "K_Deficiency"]
        return np.array([labels[idx]])


def build_pipeline(model):
    return Pipeline([("preprocessor", Preprocessor()), ("predictor", Predictor(model))])


# --- image helpers ---
def load_image(image_file):
    img = Image.open(image_file).convert("RGB")
    return img


def output(full_pipeline, img):
    a = img.resize((224, 224))
    preds = full_pipeline.predict(a)  # returns array like ['Healthy']
    return preds[0]


# --- main app UI ---
def main():
    set_bg_hack_url()
    col1, col2 = st.columns(2)

    with col1:
        st.title("H.A.R.N.")
        st.subheader(
            "Image Classification Using CNN for identifying Plant Nutrient Deficiencies")
        image_file = st.file_uploader(
            "Upload Images", type=["png", "jpg", "jpeg"])

        if st.button("Predict"):
            if image_file is not None:
                # load model (cached)
                with st.spinner("Loading Image and Model..."):
                    model = load_model_cached()
                    pipeline = build_pipeline(model)

                img = load_image(image_file)
                w, h = img.size

                # Display image responsively
                if w > h:
                    display_w = 600
                else:
                    display_w = int(w * (600.0 / h))
                st.image(img, width=display_w)

                with st.spinner("Predicting..."):
                    prediction = output(pipeline, img)

                st.success(f"Prediction: {prediction}")
            else:
                st.warning(
                    "Please upload an image file to proceed with prediction.")


if __name__ == "__main__":
    main()

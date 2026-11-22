"""Streamlit app: upload a product image, pick its category, get verdict, score and heatmap."""
from __future__ import annotations

import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent))

from src.inference.predictor import AnomalyPredictor, render_overlay  # noqa: E402

MODEL_URL = ("https://github.com/tsegaeph/industrial-anomaly-detection/"
             "releases/download/models-v1/model_{cat}.pt")
MODELS_DIR = Path(tempfile.gettempdir()) / "patchcore_models"

st.set_page_config(page_title="Industrial Anomaly Detection", layout="wide")


def ensure_model(category: str) -> None:
    """Download a category's model from the GitHub Release on first use."""
    path = MODELS_DIR / f"model_{category}.pt"
    if not path.exists():
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        urllib.request.urlretrieve(MODEL_URL.format(cat=category), tmp)
        tmp.rename(path)


@st.cache_resource
def get_predictor() -> AnomalyPredictor:
    return AnomalyPredictor(MODELS_DIR, APP_DIR / "thresholds.json", rule="p99")


predictor = get_predictor()

st.title("Industrial Visual Anomaly Detection")
st.markdown(
    "PatchCore trained **only on normal images** of each product (MVTec AD). Pick the product, "
    "give it an image, and the model scores how much the image deviates from normal and shows where. "
    "Noise, blur and strong lighting changes degrade it, and an image of an unrelated object will "
    "score as anomalous. Code and evaluation: "
    "[GitHub](https://github.com/tsegaeph/industrial-anomaly-detection). "
    "Dataset: MVTec AD (CC BY-NC-SA 4.0), non-commercial use only."
)

left, right = st.columns([1, 2])
with left:
    category = st.selectbox("Product category", predictor.categories)
    source = st.radio("Image source", ["Sample image", "Upload"], horizontal=True)
    image = None
    if source == "Upload":
        up = st.file_uploader("Product image", type=["png", "jpg", "jpeg", "bmp"])
        if up is not None:
            img = Image.open(up).convert("RGB")
            img.thumbnail((1024, 1024))
            image = np.asarray(img)
    else:
        kind = st.radio("Sample", ["good", "defect"], horizontal=True)
        sample = APP_DIR / "samples" / f"{category}_{kind}.png"
        if sample.exists():
            image = np.asarray(Image.open(sample).convert("RGB"))
        scale = st.slider("Threshold scale (lower = more sensitive)", 0.7, 1.3, 1.0, 0.05,
                      help="Thresholds come from normal training images only. For hard categories "
                           "like screw, a lower scale catches more defects but raises false alarms.")

with right:
    if image is None:
        st.info("Choose a sample or upload an image.")
    else:
        with st.spinner("Loading model and inspecting (the first run for a category is slower)..."):
            ensure_model(category)
            t0 = time.time()
            pred = predictor.predict(image, category, scale)
            ms = 1000 * (time.time() - t0)
        if pred.is_anomalous:
            st.error("Verdict: DEFECTIVE")
        else:
            st.success("Verdict: NORMAL")
        c1, c2, c3 = st.columns(3)
        c1.metric("Anomaly score", f"{pred.score:.3f}")
        c2.metric("Decision threshold", f"{pred.threshold:.3f}")
        c3.metric("Inference time", f"{ms:.0f} ms")
        a, b = st.columns(2)
        a.image(pred.image, caption="Input (resized to 256 px)", use_container_width=True)
        b.image(render_overlay(pred), caption="Anomaly heatmap (red = unlike normal)", use_container_width=True)
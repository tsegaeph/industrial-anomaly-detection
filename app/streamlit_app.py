"""Streamlit app: pick a product, upload or choose an image, get verdict, score and heatmap."""
from __future__ import annotations

import io
import sys
import tempfile
import time
import urllib.request
from dataclasses import replace
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR.parent))

from src.inference.predictor import AnomalyPredictor, Prediction, render_overlay  # noqa: E402

MODEL_URL = ("https://github.com/tsegaeph/industrial-anomaly-detection/"
             "releases/download/models-v1/model_{cat}.pt")
MODELS_DIR = Path(tempfile.gettempdir()) / "patchcore_models"
REPO_URL = "https://github.com/tsegaeph/industrial-anomaly-detection"

# Defect types present in the MVTec AD test set (from the dataset folder structure).
DEFECTS = {
    "bottle": "broken large, broken small, contamination",
    "capsule": "crack, faulty imprint, poke, scratch, squeeze",
    "carpet": "color, cut, hole, metal contamination, thread",
    "metal_nut": "bent, color, flip, scratch",
    "screw": "manipulated front, scratch head, scratch neck, thread side, thread top",
    "toothbrush": "defective",
    "zipper": "broken teeth, combined, fabric border, fabric interior, rough, split teeth, squeezed teeth",
}

st.set_page_config(page_title="Industrial Anomaly Detection", page_icon="🔍", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1150px;}
.verdict {border-radius: 14px; padding: 1.1rem 1.4rem 1.3rem; margin-bottom: 1rem; border: 1px solid;}
.verdict.ok {background: rgba(22,153,74,.12); border-color: rgba(22,163,74,.55);}
.verdict.bad {background: rgba(220,38,38,.12); border-color: rgba(220,38,38,.55);}
.verdict .label {font-size: .78rem; letter-spacing: .09em; opacity: .75; text-transform: uppercase;}
.verdict .title {font-size: 2rem; font-weight: 700; line-height: 1.25;}
.bar {position: relative; height: 10px; border-radius: 6px; background: rgba(128,128,128,.25); margin-top: 1rem;}
.bar .fill {height: 100%; border-radius: 6px;}
.bar .mark {position: absolute; top: -5px; width: 3px; height: 20px; border-radius: 2px; background: currentColor; opacity: .8;}
.barlabels {display: flex; justify-content: space-between; font-size: .75rem; opacity: .7; margin-top: .35rem;}
</style>
""", unsafe_allow_html=True)


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


@st.cache_data(show_spinner=False, max_entries=16)
def run_model(image: np.ndarray, category: str) -> tuple[Prediction, float]:
    """Model call at the default threshold. Cached, so the sliders below never re-run the model."""
    ensure_model(category)
    t0 = time.time()
    pred = get_predictor().predict(image, category)
    return pred, 1000 * (time.time() - t0)


def verdict_card(score: float, thr: float, base_thr: float, anomalous: bool) -> str:
    """Colored verdict with a bar: 0 to 2x the default threshold; the marker is the active threshold."""
    fill = min(score / (2 * base_thr), 1.0) * 100
    mark = min(thr / (2 * base_thr), 1.0) * 100
    cls, color = ("bad", "#dc2626") if anomalous else ("ok", "#16a34a")
    title = "Defect detected" if anomalous else "Looks normal"
    return (f'<div class="verdict {cls}"><div class="label">Verdict</div>'
            f'<div class="title">{title}</div>'
            f'<div class="bar"><div class="fill" style="width:{fill:.1f}%;background:{color}"></div>'
            f'<div class="mark" style="left:{mark:.1f}%"></div></div>'
            f'<div class="barlabels"><span>score 0</span><span>marker = decision threshold</span></div></div>')


predictor = get_predictor()

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Inspect a product")
    category = st.selectbox("Product category", predictor.categories,
                            format_func=lambda c: c.replace("_", " ").title())
    st.caption(f"Defect types in the dataset: {DEFECTS[category]}")
    source = st.radio("Image", ["Sample image", "Upload your own"])
    image = None
    if source == "Upload your own":
        up = st.file_uploader("PNG or JPG of this product", type=["png", "jpg", "jpeg", "bmp"])
        if up is not None:
            try:
                img = Image.open(up).convert("RGB")
                img.thumbnail((1024, 1024))
                image = np.asarray(img)
            except Exception:
                st.error("Could not read that file as an image.")
    else:
        kind = st.radio("Sample", ["good", "defect"], horizontal=True,
                        format_func=lambda k: "Normal" if k == "good" else "Defective")
        sample = APP_DIR / "samples" / f"{category}_{kind}.png"
        if sample.exists():
            image = np.asarray(Image.open(sample).convert("RGB"))
        st.caption("Samples are the first test images in file order, not hand-picked. "
                   "Some defects are missed at the default sensitivity.")
    st.divider()
    scale = st.slider("Sensitivity (threshold scale)", 0.7, 1.3, 1.0, 0.05,
                      help="Lower = more sensitive. Thresholds come from normal training images only. "
                           "On hard categories like screw, a lower value catches more defects "
                           "but raises false alarms.")
    opacity = st.slider("Heatmap opacity", 0.0, 1.0, 0.5, 0.05)

# ---------------- main ----------------
st.title("🔍 Industrial Visual Anomaly Detection")
st.caption("PatchCore trained only on normal product images (MVTec AD). It has never seen a defect: "
           "anything that does not look like normal gets a high score.")

if image is None:
    st.info("Choose a product and a sample image in the sidebar, or upload your own.")
    cols = st.columns(3)
    for col, (title, text) in zip(cols, [
        ("1. Choose the product", "Each product has its own model of what normal looks like."),
        ("2. Pick or upload an image", "Use a sample, or a photo taken like the MVTec AD images."),
        ("3. Read the result", "A verdict, an anomaly score and a heatmap of where the image looks unusual."),
    ]):
        with col.container(border=True):
            st.markdown(f"**{title}**")
            st.write(text)
else:
    with st.spinner(f"Inspecting with the {category.replace('_', ' ')} model "
                    "(the first run per product downloads its model and is slower)..."):
        base, ms = run_model(image, category)
    thr = base.threshold * scale
    anomalous = base.score > thr
    st.markdown(verdict_card(base.score, thr, base.threshold, anomalous), unsafe_allow_html=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("Anomaly score", f"{base.score:.3f}")
    m2.metric("Decision threshold", f"{thr:.3f}")
    m3.metric("Processing time", f"{ms:.0f} ms")

    overlay = render_overlay(base, alpha=opacity)   # colours stay fixed when the sensitivity slider moves
    a, b = st.columns(2)
    a.image(base.image, caption="Input (resized to 256 px)", use_container_width=True)
    b.image(overlay, caption="Anomaly heatmap (red = unlike normal)", use_container_width=True)
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    st.download_button("Download result image", buf.getvalue(),
                       file_name=f"{category}_result.png", mime="image/png")

st.divider()
tab1, tab2, tab3 = st.tabs(["How it works", "Limitations", "About"])
with tab1:
    st.markdown(
        "1. A frozen, ImageNet-pretrained **WideResNet-50** turns the image into 1,024 patch features.\n"
        "2. For each product, a **memory bank** stores the most representative patches from normal "
        "training images (a 10% coreset).\n"
        "3. Each patch of your image is scored by its **distance to the nearest stored normal patch**.\n"
        "4. The patch scores become the **heatmap**. The **image score** is the heatmap's maximum, "
        "compared with a per-product threshold.")
with tab2:
    st.markdown(
        "- You must pick the product: it does not recognise the category itself.\n"
        "- An image of an unrelated object will score as anomalous.\n"
        "- It is sensitive to noise, blur and strong lighting changes, so photos should match the "
        "MVTec AD imaging setup.\n"
        "- Thresholds are calibrated on normal images only, so some categories (screw, capsule) "
        "miss defects at the default sensitivity and others (carpet) raise false alarms.")
with tab3:
    st.markdown(
        f"Code, evaluation and results: [GitHub]({REPO_URL}). Dataset: MVTec AD (Bergmann et al.), "
        "CC BY-NC-SA 4.0, **non-commercial use only**. This demo is a portfolio project.")
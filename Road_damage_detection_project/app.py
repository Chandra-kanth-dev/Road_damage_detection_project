# ============================================
# AI-BASED ROAD DAMAGE DETECTION SYSTEM
# Python 3.10 | TensorFlow 2.15.0 | Streamlit
# ============================================

import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"       # Suppress TF C++ logs
warnings.filterwarnings("ignore")               # Suppress Python warnings

import streamlit as st
import numpy as np
import cv2
import gdown
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from PIL import Image

# ============================================
# PAGE CONFIG  (must be first Streamlit call)
# ============================================

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# LAZY-IMPORT TENSORFLOW
# Avoids slow cold-start before UI renders
# ============================================

import tensorflow as tf
from tensorflow.keras.models import load_model   # type: ignore

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080C10;
    color: #E8EDF2;
}

.title {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00FFB2 0%, #00C6FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    letter-spacing: -1px;
    margin-bottom: 4px;
}

.subtitle {
    text-align: center;
    font-size: 1rem;
    color: #5A6A7A;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 32px;
}

.card {
    background: #0F1923;
    border: 1px solid #1E2D3D;
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 20px;
}

.metric-label {
    font-size: 0.75rem;
    color: #5A6A7A;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.metric-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #00FFB2;
}

.badge-high   { background:#FF4E4E22; color:#FF6B6B; border:1px solid #FF4E4E55; padding:6px 16px; border-radius:999px; font-weight:600; display:inline-block; }
.badge-medium { background:#FFB30022; color:#FFD166; border:1px solid #FFB30055; padding:6px 16px; border-radius:999px; font-weight:600; display:inline-block; }
.badge-low    { background:#00FFB222; color:#00FFB2; border:1px solid #00FFB255; padding:6px 16px; border-radius:999px; font-weight:600; display:inline-block; }

.rec-box {
    border-radius: 12px;
    padding: 20px 24px;
    font-size: 0.95rem;
    line-height: 1.7;
}
.rec-error   { background:#FF4E4E18; border-left:4px solid #FF4E4E; }
.rec-warning { background:#FFB30018; border-left:4px solid #FFB300; }
.rec-success { background:#00FFB218; border-left:4px solid #00FFB2; }
</style>
""", unsafe_allow_html=True)

# ============================================
# CONSTANTS
# ============================================

MODEL_FILE  = "road_damage_model.h5"
FILE_ID     = "1-7VNyfOvCdA8pru2m4sO9LDXe3AuXGpu"
IMG_SIZE    = (128, 128)
CLASSES     = ["Crack", "Manhole", "Pothole"]

SEVERITY_THRESHOLDS = {"High": 85, "Medium": 60}  # confidence %

CLASS_COLORS = {
    "Crack":   "#FF6B6B",
    "Manhole": "#FFD166",
    "Pothole": "#FF4E4E"
}

# ============================================
# MODEL DOWNLOAD
# ============================================

def download_model() -> bool:
    """Downloads model from Google Drive if not present. Returns success bool."""
    if os.path.exists(MODEL_FILE):
        return True
    with st.spinner("⬇️  Downloading CNN model (first run only)..."):
        try:
            gdown.download(id=FILE_ID, output=MODEL_FILE, quiet=False)
            st.success("✅ Model downloaded successfully!")
            return True
        except Exception as exc:
            st.error(f"❌ Model download failed: {exc}")
            return False

# ============================================
# MODEL LOAD  (cached across sessions)
# ============================================

@st.cache_resource(show_spinner="Loading CNN model…")
def load_cnn_model() -> tf.keras.Model:
    """Loads .h5 model with compile=False for inference-only use."""
    return load_model(MODEL_FILE, compile=False)

# ============================================
# PREPROCESSING
# ============================================

def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Converts PIL image → normalised (1, H, W, 3) float32 array."""
    img = np.array(pil_image.convert("RGB"))
    img = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ============================================
# SEVERITY HELPER
# ============================================

def get_severity(confidence: float) -> str:
    if confidence >= SEVERITY_THRESHOLDS["High"]:
        return "High"
    elif confidence >= SEVERITY_THRESHOLDS["Medium"]:
        return "Medium"
    return "Low"

# ============================================
# CHART
# ============================================

def render_confidence_chart(probabilities: np.ndarray) -> plt.Figure:
    """Renders a styled horizontal bar chart of class probabilities."""
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor("#0F1923")
    ax.set_facecolor("#0F1923")

    colors = [CLASS_COLORS.get(c, "#00FFB2") for c in CLASSES]
    bars = ax.barh(CLASSES, probabilities, color=colors, height=0.45, zorder=2)

    # Value labels
    for bar, val in zip(bars, probabilities):
        ax.text(
            bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left",
            color="#E8EDF2", fontsize=10, fontweight="bold"
        )

    ax.set_xlim(0, 115)
    ax.set_xlabel("Confidence (%)", color="#5A6A7A", fontsize=9)
    ax.tick_params(colors="#E8EDF2", labelsize=10)
    ax.spines[:].set_visible(False)
    ax.xaxis.label.set_color("#5A6A7A")
    ax.grid(axis="x", color="#1E2D3D", linestyle="--", linewidth=0.7, zorder=0)
    fig.tight_layout(pad=1.5)
    return fig

# ============================================
# HEADER
# ============================================

st.markdown("<div class='title'>🛣️ Road Damage Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>AI-Powered Infrastructure Monitoring · CNN · Smart City</div>", unsafe_allow_html=True)
st.divider()

# ============================================
# ABOUT
# ============================================

with st.expander("📘 About this System", expanded=False):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("**🔍 Detection Types**\n- Cracks\n- Manholes\n- Potholes")
    with col_b:
        st.markdown("**🧠 Model**\n- Architecture: CNN\n- Input: 128×128 RGB\n- Framework: TF 2.15")
    with col_c:
        st.markdown("**🏙️ Applications**\n- Smart city monitoring\n- Highway maintenance\n- Autonomous vehicles")

st.divider()

# ============================================
# ENSURE MODEL IS READY
# ============================================

if not download_model():
    st.stop()

model = load_cnn_model()

# ============================================
# UPLOAD
# ============================================

st.markdown("### 📤 Upload Road Image")
uploaded_file = st.file_uploader(
    "Supported formats: JPG · JPEG · PNG",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ============================================
# INFERENCE + RESULTS
# ============================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1], gap="large")

    # --- Left: Preview ---
    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**🖼️ Uploaded Image**")
        st.image(image, use_container_width=True)
        st.caption(f"Size: {image.size[0]}×{image.size[1]} px  |  Mode: {image.mode}")
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Right: Results ---
    with col2:
        img_array = preprocess_image(image)

        with st.spinner("🔍 Analysing road condition…"):
            prediction   = model.predict(img_array, verbose=0)

        class_index     = int(np.argmax(prediction))
        predicted_class = CLASSES[class_index]
        confidence      = float(np.max(prediction) * 100)
        severity        = get_severity(confidence)
        probabilities   = prediction[0] * 100

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**📊 Prediction Result**")

        # Metrics row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown("<div class='metric-label'>Damage Type</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{predicted_class}</div>", unsafe_allow_html=True)
        with m2:
            st.markdown("<div class='metric-label'>Confidence</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-value'>{confidence:.1f}%</div>", unsafe_allow_html=True)
        with m3:
            badge_cls = f"badge-{severity.lower()}"
            st.markdown("<div class='metric-label'>Severity</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='{badge_cls}'>{severity}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Confidence bar
        st.progress(int(confidence), text=f"Model confidence: {confidence:.1f}%")

        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # ----------------------------------------
    # CHART
    # ----------------------------------------
    st.markdown("### 📈 Class Confidence Breakdown")
    fig = render_confidence_chart(probabilities)
    st.pyplot(fig)
    plt.close(fig)   # Free memory

    st.divider()

    # ----------------------------------------
    # RECOMMENDATIONS
    # ----------------------------------------
    st.markdown("### 🚨 Recommended Action")

    RECOMMENDATIONS = {
        "Pothole": (
            "rec-error",
            "⚠️ **Immediate Maintenance Required** — Pothole detected with "
            f"{confidence:.1f}% confidence (Severity: {severity}).\n\n"
            "Potholes pose a direct risk to vehicle safety and require urgent patching. "
            "Report to municipal road authorities immediately."
        ),
        "Crack": (
            "rec-warning",
            f"🔧 **Schedule Repair Soon** — Crack detected with {confidence:.1f}% confidence "
            f"(Severity: {severity}).\n\n"
            "Surface cracks can expand due to weather and traffic load. "
            "Seal or resurface the affected section within the next maintenance cycle."
        ),
        "Manhole": (
            "rec-success",
            f"ℹ️ **Routine Monitoring Advised** — Manhole cover detected with "
            f"{confidence:.1f}% confidence (Severity: {severity}).\n\n"
            "No immediate road hazard. Verify cover integrity and ensure it is flush with the road surface."
        ),
    }

    box_cls, rec_text = RECOMMENDATIONS[predicted_class]
    st.markdown(f"<div class='rec-box {box_cls}'>{rec_text}</div>", unsafe_allow_html=True)

else:
    # Placeholder when no image is uploaded
    st.markdown("""
    <div style='text-align:center; padding:60px; color:#2A3A4A;
                border:2px dashed #1E2D3D; border-radius:16px; margin-top:24px;'>
        <div style='font-size:3rem;'>🛣️</div>
        <div style='font-size:1rem; margin-top:12px;'>Upload a road image above to begin analysis</div>
    </div>
    """, unsafe_allow_html=True)

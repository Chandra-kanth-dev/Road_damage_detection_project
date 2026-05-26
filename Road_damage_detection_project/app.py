import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import matplotlib.pyplot as plt

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown("""
<style>

.main {
    background: linear-gradient(to right, #f8fbff, #eef4ff);
}

.big-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #0F172A;
}

.subtitle {
    font-size: 18px;
    text-align: center;
    color: #475569;
    margin-bottom: 20px;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 20px;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.metric-box {
    background: #ffffff;
    padding: 18px;
    border-radius: 18px;
    text-align: center;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.08);
}

.result-box {
    background: #ecfeff;
    border-radius: 18px;
    padding: 20px;
    border-left: 8px solid #0891b2;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    background-color: #2563eb;
    color: white;
    font-size: 18px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #1d4ed8;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# TITLE
# ======================================================

st.markdown(
    '<div class="big-title">🛣️ Road Damage Detection Using CNN</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Upload road images to detect road damage categories using a trained CNN model.
    Supports pothole, crack and manhole prediction.
    </div>
    """,
    unsafe_allow_html=True
)

# ======================================================
# Monkey-patch Dense layer to ignore Keras 3 specific arguments in older Keras versions
_original_dense_init = tf.keras.layers.Dense.__init__

def _patched_dense_init(self, units, *args, **kwargs):
    kwargs.pop("quantization_config", None)
    _original_dense_init(self, units, *args, **kwargs)

tf.keras.layers.Dense.__init__ = _patched_dense_init

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("road_damage_model.h5")
    return model


model = load_model()

# ======================================================
# LABELS
# ======================================================

try:
    with open("label_mapping.json", "r") as f:
        label_mapping = json.load(f)

    class_names = {
        int(k): str(v)
        for k, v in label_mapping.items()
    }

except:
    class_names = {
        0: "Pothole",
        1: "Crack",
        2: "Manhole"
    }

# ======================================================
# SIDEBAR CONTROLS
# ======================================================

st.sidebar.header("⚙️ Model Controls")

image_size = st.sidebar.slider(
    "Image Resize Dimension",
    min_value=64,
    max_value=256,
    value=128,
    step=32
)

confidence_threshold = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05
)

show_chart = st.sidebar.checkbox(
    "Show Probability Chart",
    value=True
)

normalize_image = st.sidebar.checkbox(
    "Normalize Image",
    value=True
)

st.sidebar.markdown("---")

st.sidebar.info("""
Interactive controls:

• Resize image input  
• Confidence threshold  
• Enable/disable normalization  
• Show prediction probabilities
""")

# ======================================================
# IMAGE UPLOAD
# ======================================================

uploaded_file = st.file_uploader(
    "📤 Upload Road Image",
    type=["jpg", "jpeg", "png"]
)

# ======================================================
# MAIN PREDICTION
# ======================================================

if uploaded_file is not None:

    col1, col2 = st.columns([1.2, 1])

    image = Image.open(uploaded_file)

    with col1:

        st.markdown(
            '<div class="card">',
            unsafe_allow_html=True
        )

        st.subheader("Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:

        if st.button("🔍 Detect Damage"):

            resized_image = image.resize(
                (image_size, image_size)
            )

            img_array = np.array(resized_image)

            if len(img_array.shape) == 2:
                img_array = np.stack(
                    [img_array]*3,
                    axis=-1
                )

            img_array = np.expand_dims(
                img_array,
                axis=0
            )

            if normalize_image:
                img_array = img_array / 255.0

            prediction = model.predict(
                img_array,
                verbose=0
            )

            predicted_index = np.argmax(
                prediction
            )

            confidence = float(
                np.max(prediction)
            )

            predicted_label = class_names.get(
                predicted_index,
                str(predicted_index)
            )

            st.markdown(
                '<div class="result-box">',
                unsafe_allow_html=True
            )

            st.subheader("Prediction Result")

            if confidence >= confidence_threshold:

                st.success(
                    f"Detected: {predicted_label}"
                )

            else:
                st.warning(
                    "Low confidence prediction"
                )

            st.markdown(
                f"### Confidence: {confidence:.2%}"
            )

            st.markdown("</div>",
                        unsafe_allow_html=True)

            st.markdown("### 📊 Prediction Measures")

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric(
                    "Confidence",
                    f"{confidence:.2%}"
                )

            with m2:
                st.metric(
                    "Threshold",
                    f"{confidence_threshold:.2f}"
                )

            with m3:
                st.metric(
                    "Image Size",
                    f"{image_size}px"
                )

            # ==========================================
            # PROBABILITY CHART
            # ==========================================

            if show_chart:

                st.markdown(
                    "### 📈 Class Probability Distribution"
                )

                probabilities = prediction[0]

                fig, ax = plt.subplots(
                    figsize=(8, 4)
                )

                labels = [
                    class_names.get(i, str(i))
                    for i in range(len(probabilities))
                ]

                ax.bar(
                    labels,
                    probabilities
                )

                ax.set_ylabel("Probability")
                ax.set_ylim([0, 1])
                ax.set_title(
                    "Prediction Probabilities"
                )

                st.pyplot(fig)

# ======================================================
# FOOTER
# ======================================================

st.markdown("---")

st.markdown("""
<div style='text-align:center;
font-size:16px;
color:gray;'>

CNN Based Road Damage Detection System

Built using TensorFlow + Streamlit

</div>
""", unsafe_allow_html=True)


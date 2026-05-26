# ============================================
# AI-BASED ROAD DAMAGE DETECTION SYSTEM
# ============================================

# ============================================
# IMPORT LIBRARIES
# ============================================

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import gdown
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.models import load_model

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Road Damage Detection",
    page_icon="🛣️",
    layout="wide"
)

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    text-align: center;
    font-size: 50px;
    font-weight: bold;
    color: #00FFAA;
}

.subtitle {
    text-align: center;
    font-size: 22px;
    color: #BBBBBB;
}

.section {
    padding: 20px;
    border-radius: 15px;
    background-color: #1E1E1E;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# ============================================
# MODEL DOWNLOAD
# ============================================

MODEL_FILE = "road_damage_model.keras"

# Google Drive File ID
FILE_ID = "1BQyHa_0CfLKpsRvy259dDbqh8reL12kf"

# Download model if not exists
if not os.path.exists(MODEL_FILE):

    with st.spinner("Downloading CNN model..."):

        try:

            gdown.download(
                id=FILE_ID,
                output=MODEL_FILE,
                quiet=False
            )

            st.success("Model downloaded successfully!")

        except Exception as e:

            st.error(f"Error downloading model: {e}")
            st.stop()

# ============================================
# LOAD MODEL
# ============================================

@st.cache_resource
def load_cnn_model():

    try:

        model = load_model(
            MODEL_FILE,
            compile=False
        )

        return model

    except Exception as e:

        st.error(f"Error loading model: {e}")
        st.stop()

model = load_cnn_model()

# ============================================
# CLASS LABELS
# ============================================

classes = [
    "Crack",
    "Manhole",
    "Pothole"
]

# ============================================
# HEADER
# ============================================

st.markdown(
    "<div class='title'>🛣️ AI-Based Road Damage Detection System</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Smart City Infrastructure Monitoring using CNN</div>",
    unsafe_allow_html=True
)

st.divider()

# ============================================
# ABOUT PROJECT
# ============================================

st.markdown("## 📘 About the Project")

st.markdown("""
### Why Road Monitoring is Important

Road damage such as potholes and cracks can:
- increase accidents
- damage vehicles
- reduce transportation safety

Traditional inspection methods are slow and manual.

### Role of CNN in Computer Vision

Convolutional Neural Networks (CNNs):
- automatically extract image features
- detect visual patterns
- classify road conditions accurately

### Industry Applications

- Smart city monitoring
- Highway maintenance
- Autonomous vehicles
- AI surveillance systems
- Public safety systems
""")

st.divider()

# ============================================
# IMAGE UPLOAD
# ============================================

st.markdown("## 📤 Upload Road Image")

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"]
)

# ============================================
# IMAGE PROCESSING
# ============================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    # ============================================
    # IMAGE PREVIEW
    # ============================================

    with col1:

        st.markdown("## 🖼️ Uploaded Image Preview")

        st.image(
            image,
            use_container_width=True
        )

    # ============================================
    # PREPROCESS IMAGE
    # ============================================

    img = np.array(image)

    img = cv2.resize(img, (128, 128))

    img = img.astype("float32") / 255.0

    img = np.expand_dims(img, axis=0)

    # ============================================
    # PREDICTION
    # ============================================

    with st.spinner("Analyzing road damage..."):

        prediction = model.predict(img)

    class_index = np.argmax(prediction)

    predicted_class = classes[class_index]

    confidence = float(np.max(prediction) * 100)

    # ============================================
    # SEVERITY LEVEL
    # ============================================

    if confidence > 85:
        severity = "High"

    elif confidence > 60:
        severity = "Medium"

    else:
        severity = "Low"

    # ============================================
    # PREDICTION AREA
    # ============================================

    with col2:

        st.markdown("## 📊 Prediction Result")

        st.success(
            f"Prediction: {predicted_class} Detected"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        st.warning(
            f"Severity Level: {severity}"
        )

    st.divider()

    # ============================================
    # VISUALIZATION AREA
    # ============================================

    st.markdown("## 📈 Prediction Visualization")

    probabilities = prediction[0] * 100

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(classes, probabilities)

    ax.set_xlabel("Damage Classes")
    ax.set_ylabel("Confidence (%)")
    ax.set_title("Class Confidence Graph")

    st.pyplot(fig)

    # ============================================
    # RECOMMENDATIONS
    # ============================================

    st.markdown("## 🚨 Recommendations")

    if predicted_class == "Pothole":

        st.error("""
Immediate maintenance recommended.

High-risk road condition detected.
        """)

    elif predicted_class == "Crack":

        st.warning("""
Repair recommended soon.

Road condition may worsen over time.
        """)

    else:

        st.success("""
Low immediate risk detected.

Regular monitoring recommended.
        """)

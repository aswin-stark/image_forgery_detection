import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Image Authenticity Checker",
    page_icon="🔍",
    layout="centered"
)

# ---------------- THEME ----------------
col1, col2 = st.columns([7, 2])

with col2:
    dark_mode = st.toggle("🌙 Dark Mode")
if dark_mode:
    bg = "linear-gradient(-45deg, #0f172a, #1e3a8a, #0f172a, #1e293b)"
    card = "rgba(30, 41, 59, 0.85)"
    text = "#e2e8f0"   # ✅ soft white
else:
    bg = "linear-gradient(-45deg, #93c5fd, #bae6fd, #c7d2fe, #e0e7ff)"
    card = "rgba(255, 255, 255, 0.9)"
    text = "#ffffff"   # ✅ soft black

# ---------------- CSS ANIMATIONS ----------------
st.markdown(f"""
<style>

/* Animated Background */
.stApp {{
    background: {bg};
    background-size: 400% 400%;
    animation: gradientBG 10s ease infinite;
    color: {text};
}}

/* Force ALL text color */
h1, h2, h3, h4, h5, h6, p, div, span, label {{
    color: {text} !important;
}}

/* Fix file uploader text */
section[data-testid="stFileUploader"] * {{
    color: {text} !important;
}}

@keyframes gradientBG {{
    0% {{background-position: 0% 50%;}}
    50% {{background-position: 100% 50%;}}
    100% {{background-position: 0% 50%;}}
}}

/* Fade-in */
.fade-in {{
    animation: fadeIn 1s ease-in;
}}

@keyframes fadeIn {{
    from {{opacity: 0; transform: translateY(20px);}}
    to {{opacity: 1; transform: translateY(0);}}
}}

/* Card */
.card {{
    background: {card};
    padding: 25px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.2);
}}

/* Upload Box Animation */
.upload-box {{
    border: 2px dashed #2563eb;
    padding: 30px;
    border-radius: 12px;
    text-align: center;
    transition: 0.3s;
}}

.upload-box:hover {{
    background: rgba(37, 99, 235, 0.1);
    transform: scale(1.02);
}}

/* Title */
.title {{
    text-align: center;
    font-size: 36px;
    font-weight: 700;
    color: #2563eb !important;
}}

/* Subtitle */
.subtitle {{
    text-align: center;
    margin-bottom: 20px;
    color: {text};
}}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_keras_model():
    return load_model("model.keras")

model = load_keras_model()

# ---------------- HEADER ----------------
st.markdown('<div class="title fade-in">🔍 Image Authenticity Checker</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle fade-in">Detect Real vs Fake images</div>', unsafe_allow_html=True)

# ---------------- UPLOAD ----------------



uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"])

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PROCESS ----------------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")

    st.markdown("### 🖼️ Preview")
    st.image(img, use_column_width=True)

    # Animated loading
    with st.spinner("🔎 Analyzing with AI..."):
        time.sleep(1.2)
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        confidence = prediction[0][0]

    st.markdown("### 📊 Result")

    # Animated progress
    progress_bar = st.progress(0)
    for i in range(100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    if confidence > 0.5:
        st.error("🚫 Fake Image Detected")
        st.write(f"Confidence: **{confidence:.2%}**")
    else:
        st.success("✅ Real Image Detected")
        st.write(f"Confidence: **{1 - confidence:.2%}**")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<center style='color:gray;'>💙 Created by ASWIN</center>", unsafe_allow_html=True)
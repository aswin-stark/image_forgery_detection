import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="Fake vs Real Detector", page_icon="🔍")

# Load model (cached so it only loads once)
@st.cache_resource
def load_keras_model():
    return load_model("model.keras")

model = load_keras_model()

# UI
st.title("🔍 Fake vs Real Image Detector")
st.write("Upload an image to check whether it's Real or Fake.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing..."):
        img_resized = img.resize((224, 224))
        img_array = np.array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)

        prediction = model.predict(img_array)
        confidence = prediction[0][0]

        if confidence > 0.5:
            st.error(f"**Fake ❌** — Confidence: {confidence:.2%}")
        else:
            st.success(f"**Real ✅** — Confidence: {1 - confidence:.2%}")
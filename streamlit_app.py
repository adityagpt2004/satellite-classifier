import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Set page title
st.set_page_config(page_title="Satellite Image Classifier", layout="centered")
st.title("🛰️ Satellite Image Classifier")
st.markdown("Upload a satellite image and the model will predict the class (e.g., Forest, Highway, etc.)")

# Load model once
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("satellite_model_best.h5")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

model = load_model()

# EuroSAT class names in correct order
class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

# File uploader
uploaded_file = st.file_uploader("📤 Upload a .jpg or .png image", type=["jpg", "png"])

if uploaded_file is not None:
    try:
        # Load and preprocess image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img = image.resize((64, 64))               # Resize to model input
        img = np.array(img) / 255.0                # Normalize
        img = np.expand_dims(img, axis=0)          # Add batch dimension

        # Make prediction
        if model is not None:
            prediction = model.predict(img)
            predicted_class = class_names[np.argmax(prediction)]
            confidence = np.max(prediction) * 100

            # Show result
            st.success(f"🌍 Predicted Class: **{predicted_class}**")
            st.info(f"🧠 Confidence: **{confidence:.2f}%**")
        else:
            st.warning("⚠️ Model could not be loaded. Please check your .h5 file.")

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")


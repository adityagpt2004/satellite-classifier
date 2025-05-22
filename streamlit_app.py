# streamlit_app.py

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Title
st.title("🛰️ Satellite Image Classifier")
st.markdown("Upload a satellite image and the model will predict the class (e.g., Forest, Highway, etc.)")

# Load model once at start
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("satellite_model_best.h5")
    return model

model = load_model()

# Class names from EuroSAT dataset (in order)
class_names = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
    'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake'
]

# File uploader
if uploaded_file is not None:
    try:
        # Load and display the image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess
        img = image.resize((64, 64))  # Resize to match model input
        img = np.array(img) / 255.0   # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Ensure it's the right shape
        if img.shape != (1, 64, 64, 3):
            raise ValueError(f"Unexpected image shape: {img.shape}. Expected (1, 64, 64, 3).")

        # Predict
        prediction = model.predict(img)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Show result
        st.success(f"🌍 Predicted Class: **{predicted_class}**")
        st.info(f"🧠 Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error(f"⚠️ Error processing image: {e}")


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
uploaded_file = st.file_uploader("Upload a .jpg or .png image", type=["jpg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((64, 64))  # Resize to match model input
    img = np.array(img) / 255.0   # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Output result
    st.success(f"🌍 Predicted Class: **{predicted_class}**")
    st.info(f"🧠 Confidence: {confidence:.2f}%")

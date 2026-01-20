import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

from tensorflow.keras.models import load_model

model = load_model(
    "intel_image_classifier.h5",
    compile=False
)


class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

st.title("Intel Image Classification")
st.write("Upload an image and the model will predict the category.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((150, 150))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"Predicted Class: {predicted_class}")

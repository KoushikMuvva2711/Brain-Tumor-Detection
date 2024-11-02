import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

# Load the pre-trained model
model = load_model('./BrainTumor10EpochsCategorical.h5')

# Function to preprocess image and make prediction
def predict_tumor(image):
    # Convert to RGB and resize to (64, 64)
    img = Image.fromarray(image).convert('RGB')
    img = img.resize((64, 64))
    
    # Convert to numpy array, normalize, and add batch dimension
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    return prediction

# Streamlit UI
st.title("Brain Tumor Detection")
st.write("Upload an MRI image, and the model will predict if a tumor is present.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image file
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Display uploaded image
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)
    st.write("Classifying...")

    # Make prediction
    prediction = predict_tumor(image)
    result = "Tumor detected" if prediction[0][1] > prediction[0][0] else "No tumor detected"
    
    # Display result
    st.write("Prediction:", result)

import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image

# Load the pre-trained model
model = load_model('./BrainTumor50EpochsCategorical.h5')

# Function to preprocess image and make prediction
def predict_tumor(image):
    # Convert to RGB and resize to (224, 224)
    img = Image.fromarray(image).convert('RGB')
    img = img.resize((224, 224))
    
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
    # Read the image file using PIL for better compatibility
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    # Display uploaded image
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)
    st.write("Classifying...")

    # Make prediction
    prediction = predict_tumor(image_np)
    
    # Assuming the model output is binary classification (two output nodes)
    # Customize this if the model output format is different
    result = "Tumor detected" if prediction[0][1] > prediction[0][0] else "No tumor detected"
    
    # Display result
    st.write("Prediction:", result)

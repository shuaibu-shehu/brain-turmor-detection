import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Set page config
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

# Load the trained model
try:
    MODEL_PATH = 'mymodel.h5'
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found. Please ensure model.h5 exists in the 'models' directory")
        st.stop()
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Class labels
class_labels = ['pituitary', 'glioma', 'notumor', 'meningioma']

# Helper function to predict tumor type
def predict_tumor(image):
    IMAGE_SIZE = 128
    img = load_img(image, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence_score = np.max(predictions, axis=1)[0]

    if class_labels[predicted_class_index] == 'notumor':
        return "No Tumor", confidence_score
    else:
        return f"Tumor: {class_labels[predicted_class_index]}", confidence_score

# Main UI
st.title("Brain Tumor Detection")
st.write("Upload an MRI scan to detect the presence and type of brain tumor")

# File uploader
uploaded_file = st.file_uploader("Choose an MRI image file", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded MRI Scan", use_container_width=True)
    
    # Add a predict button
    if st.button("Predict"):
        with st.spinner("Analyzing image..."):
            # Make prediction
            result, confidence = predict_tumor(uploaded_file)
            
            # Display results
            st.success("Analysis Complete!")
            st.write(f"**Prediction:** {result}")
            st.write(f"**Confidence:** {confidence*100:.2f}%")
            
            # Display additional information based on the result
            if "No Tumor" not in result:
                st.warning("Please consult with a healthcare professional for proper medical advice.")

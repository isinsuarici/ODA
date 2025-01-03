import streamlit as st
import cv2
import numpy as np
import params
from PIL import Image as PILImage
from ultralytics import YOLO

# Load YOLO model
def load_model(model_path):
    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error("Unable to load the model.")
        st.error(e)
        return None

# Perform object detection on an image
def detect_objects_in_image(model):    

    uploaded_img = st.file_uploader("", type=("jpg", "png", "jpeg", "bmp", "tiff"))
    
    # Columns for UI layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("## Parasite Image")
        
        if uploaded_img is not None:
            try:
                # Display the uploaded image
                image = PILImage.open(uploaded_img)
                st.image(image, caption="Uploaded Image", use_column_width=True)
            except Exception as e:
                st.error("Error displaying the uploaded image.")
                st.error(e)
        else:
            st.write("Please upload an image.")
    
    with col2:
        st.write("## Result")
        if uploaded_img is not None:
            try:
                # Convert PIL image to OpenCV format
                cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                # Perform object detection
                res = model.predict(cv_image)
                res_plot = res[0].plot()[:, :, ::-1]  # Convert from BGR to RGB for displaying

                # Display the detected image
                st.image(res_plot, caption='Detected Image', use_column_width=True)
            except Exception as e:
                st.error("Error processing the uploaded image.")
                st.error(e)
        else:
            st.write("Please upload an image to view the results.")

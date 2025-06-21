import streamlit as st
from PIL import Image
from utils import predict_flower
import subprocess
import os

st.set_page_config(page_title="🌸 Flower Classifier")
st.title("🌸 Flower Species Identifier")
st.write("Upload an image of a flower (daisy, dandelion, rose, sunflower, tulip)")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    with st.spinner('Classifying...'):
        prediction, confidence, resized_img = predict_flower(image)
        st.image(resized_img, caption="Preprocessed Image (128x128)", use_column_width=False)
        st.success(f"**Prediction:** {prediction.title()} ({confidence*100:.2f}% confidence)")

if st.button("Retrain Model with New Data"):
    st.info("Retraining model... this might take a few minutes.")
    result = subprocess.run(["python", "train.py"], capture_output=True, text=True)
    if result.returncode == 0:
        st.success("Retraining complete! Model updated.")
    else:
        st.error("Retraining failed. Check console logs.")

if os.path.exists("model/training_performance.png"):
    st.subheader("📈 Model Performance Statistics")
    st.image("model/training_performance.png", caption="Training vs Validation Accuracy and Loss")
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import matplotlib.pyplot as plt

CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
MODEL_PATH = 'model/flower_classifier.h5'

model = None

def load_model_once():
    global model
    if model is None:
        model = load_model(MODEL_PATH)
    return model

def preprocess_image(img, target_size=(128, 128)):
    if isinstance(img, Image.Image):
        resized = img.resize(target_size)
    img_array = image.img_to_array(resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array, resized

def predict_flower(img):
    model = load_model_once()
    processed, resized = preprocess_image(img)
    preds = model.predict(processed)[0]
    top_idx = np.argmax(preds)
    return CLASS_NAMES[top_idx], preds[top_idx], resized
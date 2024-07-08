import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('food_recognition_model.h5')

# Load class names (make sure to replace these with the actual class names)
class_names = ['apple_pie', 'beef_carpaccio', 'beef_tartare', 'beet_salad', 'beignets', ..., 'waffles']  # Add all class names here

# Define wine recommendations based on food class
wine_recommendations = {
    'apple_pie': 'Chardonnay',
    'beef_carpaccio': 'Pinot Noir',
    'beef_tartare': 'Merlot',
    'beet_salad': 'Sauvignon Blanc',
    'beignets': 'Sparkling Wine',
    # Add more mappings for all classes
}

def predict_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    
    return predicted_class, wine_recommendations.get(predicted_class, "No recommendation available")

st.title("Wine Recommendation Based on Meal Image")
st.write("Upload an image of your meal and get a wine recommendation!")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    
    st.write("Classifying...")
    label, wine = predict_image(img)
    
    st.write(f"Predicted class: {label}")
    st.write(f"Wine recommendation: {wine}")

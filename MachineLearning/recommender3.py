import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np
import pandas as pd
import os

# Load the trained model
model = tf.keras.models.load_model("MachineLearning/food_recognition_model4.h5")

# New class names
class_names = ['Bread', 'Dairy product', 'Dessert', 'Egg', 'Fried food', 'Meat', 'Noodles-Pasta', 'Rice', 'Seafood', 'Soup', 'Vegetable-Fruit']

# Load wine recommendations from CSV
try:
    df_wine_suggestions = pd.read_csv("MachineLearning/df_wine_suggestions.csv")
except Exception as e:
    st.error(f"Error loading wine suggestions CSV: {e}")

def get_wine_recommendation(food_class):
    recommendation = df_wine_suggestions[df_wine_suggestions["Food Category"].str.strip() == food_class.strip()]
    if not recommendation.empty:
        wine_info = recommendation.iloc[0]
        return wine_info["Wine Recommendation"], wine_info["Image URL"]
    else:
        return "No recommendation available", ""

def preprocess_image(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)  # Get the confidence of the prediction
    return predicted_class, confidence, get_wine_recommendation(predicted_class), predictions

# Open the image
img = Image.open("MachineLearning/wine_images/wain_site_cover.png")

# # Resize the image (you can specify both width and height)
# new_width = 600
# new_height = 300
# img = img.resize((new_width, new_height))

# Display the resized image
st.image(img, use_column_width=True)


st.title("Upload an image of your meal and get a wine recommendation!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')  # Ensure image is in RGB format
    st.image(img, caption="Your Meal Image", use_column_width=True)
    
    st.write("Classifying...")
    img.save("temp_image.jpg")  # Save the uploaded image to a temporary file
    label, confidence, (wine_recommendation, wine_image_url), predictions = predict_image("temp_image.jpg")
    
    st.title(f"It looks like it's {label}")

    # Load the metrics from the CSV file
    metrics_df = pd.read_csv("MachineLearning/metrics.csv")

    # Transpose and set the first row as the header
    metrics_df = metrics_df.T
    metrics_df.columns = metrics_df.iloc[0]
    metrics_df = metrics_df[1:]

    # Extract metrics for the predicted class
    class_metrics = metrics_df[label]

    # Create an expander for details
    with st.expander("See Details"):
        st.write(f"Confidence: {confidence:.2f}")
        if not class_metrics.empty:
            st.write(f"Precision Score of {label} class: {float(class_metrics['precision']):.2f}")
            st.write(f"Recall Score of {label} class: {float(class_metrics['recall']):.2f}")
            st.write(f"F1-Score Score of {label} class: {float(class_metrics['f1-score']):.2f}")
        else:
            st.write("No metrics available for this class.")
    
    # Define custom CSS for text wrapping
    custom_css = """
        <style>
        .wine-recommendation {
        white-space: pre-wrap; /* Ensures text wraps and preserves newlines */
        word-wrap: break-word; /* Ensures long words break properly */
        font-size: 24px; /* Adjust the font size as needed */
        line-height: 1.6; /* Adjust the line height for better readability */
        }
        </style>
        """

    # Inject the custom CSS
    st.markdown(custom_css, unsafe_allow_html=True)

    # Use a div with the custom CSS class
    st.title("Wine recommendation for your meal is:")
    st.markdown(f"<div class='wine-recommendation'>{wine_recommendation}</div>", unsafe_allow_html=True)

    if wine_image_url:
        st.image(wine_image_url, caption="Recommended Wine", use_column_width=True)
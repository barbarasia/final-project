import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import numpy as np
import pandas as pd
import os
import shutil

# Custom ImageDataGenerator class
class CustomImageDataGenerator(ImageDataGenerator):
    def flow_from_directory(self, directory, *args, **kwargs):
        generator = super().flow_from_directory(directory, *args, **kwargs)
        self.target_size = kwargs.get("target_size", (224, 224))
        self.num_classes = generator.num_classes
        self.filepaths = generator.filepaths
        self.labels = generator.classes
        return generator

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

# Preprocess the image without augmentation
def preprocess_image_without_datagen(img_path):
    # Create a temporary directory structure
    temp_dir = "/tmp/single_image_dir"
    class_dir = os.path.join(temp_dir, "class")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(class_dir)

    # Save the image to the class directory
    temp_img_path = os.path.join(class_dir, "image.jpg")
    img = Image.open(img_path).convert("RGB")
    img = img.resize((224, 224))
    img.save(temp_img_path)

    # Create a data generator for the single image without augmentation
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    generator = datagen.flow_from_directory(
        temp_dir,
        target_size=(224, 224),
        batch_size=1,
        class_mode=None,
        shuffle=False
    )

    img_array = next(generator)
    return img_array

def predict_image_without_datagen(img_path):
    img_array = preprocess_image_without_datagen(img_path)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)  # Get the confidence of the prediction
    return predicted_class, confidence, get_wine_recommendation(predicted_class), predictions

st.image("MachineLearning/wine_images/app_image.png", use_column_width=True)  # Replace with your image path

st.title("Upload an image of your meal and get a wine recommendation!")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_path = uploaded_file.name
    img = Image.open(uploaded_file).convert('RGB')  # Ensure image is in RGB format
    st.image(img, caption="Your Meal Image", use_column_width=True)
    
    st.write("Classifying...")
    label, confidence, (wine_recommendation, wine_image_url), predictions = predict_image_without_datagen(uploaded_file)
    
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
    
    st.title("Wine recommendation for your meal is:")
    st.text(wine_recommendation)

    if wine_image_url:
        st.image(wine_image_url, caption="Recommended Wine", use_column_width=True)
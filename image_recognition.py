# import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib

def preprocess_image(image):
    # Resize the image to 28x28 pixels
    resized_image = image.resize((28, 28))

    # Convert the resized image to grayscale
    grayscale_image = resized_image.convert('L')

    # Flatten the image into a 1D array of 784 pixel values
    flattened_image = np.array(grayscale_image).flatten()

    return flattened_image

def load_model(uploaded_model):
    # Load the model from the uploaded .pkl file
    try:
        model = joblib.load(uploaded_model)
        st.write("Model loaded successfully.")
    except Exception as e:
        st.write(f"Error loading model: {e}")
        model = None

    return model

def predict_class(image, model):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Reshape the image to match the model's input shape (1, 784)
    input_image = processed_image.reshape(1, 784)

    # Normalize the input
    input_image = input_image / 255.0

    # Make predictions
    predictions = model.predict(input_image)

    # Get the predicted class
    predicted_class = np.argmax(predictions)

    return predicted_class

def main():
    st.title("Image Transformation and Prediction")

    # Create a Streamlit file uploader for the model
    uploaded_model = st.file_uploader( "Upload the joblib model", type=["pkl"])

    # Create a Streamlit file uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_model is not None and uploaded_file is not None:
        # Load the model
        model = load_model(uploaded_model)

        if model is not None:
            # Convert the uploaded file to PIL Image
            pil_image = Image.open(uploaded_file)

            # Display the original image
            st.header("Original Image")
            st.image(pil_image, use_column_width=True)

            # Predict the class of the image
            predicted_class = predict_class(pil_image, model)

            # Display the predicted class
            st.header("Predicted Class")
            st.write(f"The predicted class is: {predicted_class}")

if __name__ == "__main__":
    main()

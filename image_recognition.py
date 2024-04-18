import keras
import numpy as np
import streamlit as st
from PIL import Image


def preprocess_image(image):
    # Resize the image to 28x28 pixels
    resized_image = image.resize((28, 28))

    # Convert the resized image to grayscale
    grayscale_image = resized_image.convert('L')

    # Convert the grayscale image to numpy array
    image_array = np.array(grayscale_image)

    # Expand dimensions to add the channel dimension
    image_array = np.expand_dims(image_array, axis=-1)

    # Reshape the image to match the model's input shape (None, 28, 28, 1)
    processed_image = image_array.reshape(1, 28, 28, 1)

    return processed_image

def load_model(uploaded_model):
    # Load the model from the uploaded file
    try:
        model = keras.saving.load_model(uploaded_model)
        st.write("Model loaded successfully.")
    except Exception as e:
        st.write(f"Error loading model: {e}")
        model = None

    return model

def predict_class(image, model):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Normalize the input
    input_image = processed_image / 255.0

    # Make predictions
    predictions = model.predict(input_image)

    # Get the predicted class
    predicted_class = np.argmax(predictions)

    return predicted_class

def main():
    st.title("Sign Language Recognition")

    # Create a Streamlit file uploader for the model
    uploaded_model = st.file_uploader( "Upload the model", type=["keras"])

    # Create a Streamlit file uploader
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    model = keras.models.Model()

    if uploaded_model is not None:
        # Load the model
        model = load_model( uploaded_model )
    else:
        model = keras.saving.load_model("model.keras")

    if uploaded_file is not None:
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

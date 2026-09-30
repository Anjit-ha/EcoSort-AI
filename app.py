import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Smart Waste Classifier",
    page_icon="♻️",
    layout="centered"
)

# -----------------------------------
# Title
# -----------------------------------

st.title("♻️ Smart Waste Classifier")
st.write("Use your webcam or upload an image to identify waste.")

st.divider()


# -----------------------------------
# Load Model
# -----------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("keras_model.h5")


@st.cache_data
def load_labels():
    with open("labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]

    # Handles labels like:
    # 0 paper_waste
    # 1 plastic_waste

    cleaned_labels = []

    for label in labels:
        parts = label.split(" ", 1)

        if len(parts) == 2 and parts[0].isdigit():
            cleaned_labels.append(parts[1])
        else:
            cleaned_labels.append(label)

    return cleaned_labels


model = load_model()
class_names = load_labels()


# -----------------------------------
# Waste Information
# -----------------------------------

waste_info = {

    "paper_waste": {
        "bin": "📄 Paper Recycling Bin",
        "tip": "Keep paper dry and clean before recycling."
    },

    "plastic_waste": {
        "bin": "♻️ Plastic Recycling Bin",
        "tip": "Clean plastic containers before recycling."
    },

    "metal_waste": {
        "bin": "🔩 Metal Recycling Bin",
        "tip": "Separate metal items from general waste."
    },

    "organic_waste": {
        "bin": "🌱 Organic / Compost Bin",
        "tip": "Organic waste can be composted."
    },

    "general_waste": {
        "bin": "🗑️ General Waste Bin",
        "tip": "Dispose of non-recyclable waste responsibly."
    }
}


# -----------------------------------
# Input Selection
# -----------------------------------

st.subheader("📸 Select Input")

input_option = st.radio(
    "Choose how you want to provide the waste image:",
    ["📁 Upload Image", "📷 Use Webcam"],
    horizontal=True
)


uploaded_file = None


# -----------------------------------
# Upload Image
# -----------------------------------

if input_option == "📁 Upload Image":

    uploaded_file = st.file_uploader(
        "Upload a waste image",
        type=["jpg", "jpeg", "png"]
    )


# -----------------------------------
# Webcam
# -----------------------------------

else:

    st.write("Show the waste item in front of your webcam.")

    camera_image = st.camera_input(
        "Take a picture"
    )

    if camera_image is not None:
        uploaded_file = camera_image


# -----------------------------------
# Prediction
# -----------------------------------

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Input Image",
        width=450
    )

    # -----------------------------------
    # Preprocess Image
    # -----------------------------------

    image_resized = image.resize((224, 224))

    image_array = np.asarray(
        image_resized,
        dtype=np.float32
    )

    # Teachable Machine normalization
    image_array = (image_array / 127.5) - 1

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # -----------------------------------
    # Model Prediction
    # -----------------------------------

    prediction = model.predict(
        image_array,
        verbose=0
    )

    predicted_index = np.argmax(prediction)

    confidence = prediction[0][predicted_index]

    predicted_class = class_names[predicted_index]

    # -----------------------------------
    # Result
    # -----------------------------------

    st.divider()

    st.subheader("🤖 AI Prediction")

    st.success(
        f"Detected: **{predicted_class}**"
    )

    st.metric(
        "Confidence",
        f"{confidence * 100:.2f}%"
    )

    # -----------------------------------
    # Recommendation
    # -----------------------------------

    predicted_class_clean = predicted_class.lower().strip()

    if predicted_class_clean in waste_info:

        info = waste_info[predicted_class_clean]

        st.subheader("♻️ Recommended Disposal")

        st.info(
            f"### {info['bin']}\n\n"
            f"💡 {info['tip']}"
        )

    # -----------------------------------
    # All Probabilities
    # -----------------------------------

    st.subheader("📊 Prediction Probabilities")

    for i, label in enumerate(class_names):

        probability = prediction[0][i]

        st.write(
            f"**{label}** — "
            f"{probability * 100:.2f}%"
        )

        st.progress(
            float(probability)
        )
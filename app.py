from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

MODEL_PATH = Path("model/keras_model.h5")
LABELS_PATH = Path("model/labels.txt")
IMAGE_SIZE = 224


@st.cache_resource
def load_model(path: Path) -> tf.keras.Model:
    return tf.keras.models.load_model(path, compile=False)


@st.cache_data
def load_labels(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def preprocess_image(image: Image.Image) -> np.ndarray:
    processed = ImageOps.fit(image.convert("RGB"), (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    image_array = np.asarray(processed).astype(np.float32)
    normalized = (image_array / 127.5) - 1
    return np.expand_dims(normalized, axis=0)


def main() -> None:
    st.set_page_config(page_title="EcoSort-AI", page_icon="♻️")
    st.title("♻️ EcoSort-AI")
    st.write("Upload a waste image to classify it with a Teachable Machine TensorFlow model.")

    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        st.error(
            "Model files were not found. Add `model/keras_model.h5` and `model/labels.txt` "
            "from your Teachable Machine export."
        )
        st.stop()

    try:
        model = load_model(MODEL_PATH)
        labels = load_labels(LABELS_PATH)
    except Exception as exc:
        st.error(f"Failed to load model assets: {exc}")
        st.stop()

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded_file is None:
        return

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    input_data = preprocess_image(image)
    predictions = model.predict(input_data, verbose=0)[0]

    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])
    predicted_label = labels[predicted_index] if predicted_index < len(labels) else f"Class {predicted_index}"

    st.subheader("Prediction")
    st.success(f"{predicted_label} ({confidence:.2%} confidence)")

    st.subheader("All class scores")
    st.dataframe(
        {
            "class": [labels[i] if i < len(labels) else f"Class {i}" for i in range(len(predictions))],
            "score": [float(score) for score in predictions],
        },
        use_container_width=True,
    )


if __name__ == "__main__":
    main()

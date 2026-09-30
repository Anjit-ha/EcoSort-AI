
# ♻️ EcoSort AI

EcoSort AI is a waste classification application built using
Teachable Machine, TensorFlow, and Streamlit.

## Features

- 📁 Upload waste images
- 📷 Capture images using webcam
- 🤖 AI-based waste classification
- 📊 Prediction confidence
- ♻️ Waste disposal recommendation

## Technologies

- Python
- TensorFlow
- Teachable Machine
- Streamlit
- NumPy
- Pillow

## Waste Classes

Currently the model supports:

- Paper Waste
- Plastic Waste

## Run Locally

Install the dependencies: pip install -r requirements.txt
## Run Locally
streamlit run app.py


---

## 3. Be careful with `keras_model.h5`

Your model file is important.

GitHub allows browser uploads up to **25 MB per file**, while command-line Git can push files up to **100 MB**; files larger than 100 MB require Git LFS. :contentReference[oaicite:1]{index=1}

Check the size of:

```text
keras_model.h5
```bash
pip install -r requirements.txt

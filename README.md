# EcoSort-AI
AI-powered waste classification using Teachable Machine, TensorFlow and Streamlit.

## Run locally

1. Export your image model from **Teachable Machine** as a **Keras** model.
2. Place the exported files in:
   - `/home/runner/work/EcoSort-AI/EcoSort-AI/model/keras_model.h5`
   - `/home/runner/work/EcoSort-AI/EcoSort-AI/model/labels.txt`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```

Upload an image in the app and it will return the predicted waste class and confidence.

import streamlit as st
import numpy as np
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import json
import plotly.express as px
import os

st.set_page_config(page_title="Waste Classification AI", layout="wide")

# ---------------- MODEL LOAD ---------------- #
@st.cache_resource
def load_my_model():
    try:
        model = load_model("final_garbage_model.keras", compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None

model = load_my_model()

# ---------------- LOAD CLASS NAMES ---------------- #
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("⚙️ Menu")
st.sidebar.success("Model Ready ✅" if model else "Model Error ❌")

# ---------------- PROJECT OVERVIEW ---------------- #
st.title("♻️ Waste Garbage Management System (AI)")

st.markdown("""
### 📌 Project Overview
यह AI आधारित सिस्टम image upload करके waste classify करता है:
- Plastic
- Metal
- Glass
- Paper
- Trash
- Battery
- Clothes आदि

👉 Deep Learning model (Keras) use किया गया है।
""")

# ---------------- IMAGE PREPROCESS ---------------- #
def preprocess_image(img):
    img = img.resize((160, 160))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- PREDICT FUNCTION ---------------- #
def predict(img):
    try:
        img_array = preprocess_image(img)
        prediction = model.predict(img_array)[0]

        top3_idx = prediction.argsort()[-3:][::-1]

        results = []
        for i in top3_idx:
            results.append((class_names[i], float(prediction[i])))

        label = results[0][0]
        confidence = results[0][1]

        return label, confidence, results

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return "Model Error", 0.0, []

# ---------------- UI ---------------- #
st.header("📤 Upload Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    if st.button("Detect"):
        if model is None:
            st.error("❌ Model not loaded")
        else:
            label, confidence, top3 = predict(image)

            st.success(f"Prediction: {label}")
            st.write(f"### Confidence: {round(confidence*100,2)}%")

            # ---------------- TOP 3 ---------------- #
            st.subheader("🔥 Top 3 Predictions")

            labels = [x[0] for x in top3]
            probs = [x[1] for x in top3]

            for i in range(len(top3)):
                st.write(f"{labels[i]} → {round(probs[i]*100,2)}%")

            # ---------------- CHART ---------------- #
            fig = px.bar(
                x=labels,
                y=probs,
                labels={"x": "Class", "y": "Probability"},
                title="📊 Prediction Probability"
            )
            st.plotly_chart(fig)

# ---------------- DEBUG (optional remove later) ---------------- #
# st.write("Files:", os.listdir())

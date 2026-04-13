import streamlit as st
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import json
import plotly.express as px
import os

# ====================== PAGE CONFIG ====================== #
st.set_page_config(
    page_title="Waste Classification AI",
    page_icon="♻️",
    layout="wide"
)

# ====================== MODEL LOAD ====================== #
@st.cache_resource
def load_my_model():
    try:
        model = keras.models.load_model(
            "final_garbage_model.h5",   
            compile=False, 
            safe_mode=False
        )
        return model
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None

model = load_my_model()

# ====================== LOAD CLASS NAMES ====================== #
def load_class_names():
    try:
        with open("class_names.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ class_names.json load failed: {e}")
        return []

class_names = load_class_names()

# ====================== SIDEBAR ====================== #
st.sidebar.title("⚙️ Menu")
st.sidebar.success("Model Ready ✅" if model else "Model Error ❌")
st.sidebar.info(f"Total Classes: {len(class_names)}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Supported Waste Types:**")
for name in class_names:
    st.sidebar.write(f"• {name}")

# ====================== PROJECT OVERVIEW ====================== #
st.title("♻️ Waste Garbage Management System (AI)")
st.markdown("""
### 📌 Project Overview
यह AI आधारित सिस्टम image upload करके waste classify करता है:
- 🧴 Plastic
- 🔩 Metal
- 🍶 Glass
- 📄 Paper
- 🗑️ Trash
- 🔋 Battery
- 👕 Clothes आदि

👉 **Deep Learning model (Keras/TensorFlow)** use किया गया है।
""")
st.divider()

# ====================== IMAGE PREPROCESS ====================== #
def preprocess_image(img: Image.Image) -> np.ndarray:
    """Resize and normalize image for model input."""
    img = img.resize((160, 160))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ====================== PREDICT FUNCTION ====================== #
def predict(img: Image.Image):
    """Run model prediction and return top-3 results."""
    try:
        img_array = preprocess_image(img)
        prediction = model.predict(img_array)[0]
        top3_idx = prediction.argsort()[-3:][::-1]
        results = [(class_names[i], float(prediction[i])) for i in top3_idx]
        label, confidence = results[0]
        return label, confidence, results
    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")
        return "Error", 0.0, []

# ====================== UPLOAD & DETECT UI ====================== #
st.header("📤 Upload Image for Detection")

col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader(
        "Upload a waste image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", width=280)
        detect_btn = st.button("🔍 Detect Waste", use_container_width=True)
    else:
        detect_btn = False

with col2:
    if uploaded_file and detect_btn:
        if model is None:
            st.error("❌ Model not loaded. Please check model file.")
        elif not class_names:
            st.error("❌ Class names not found. Please check class_names.json.")
        else:
            with st.spinner("Analyzing image..."):
                label, confidence, top3 = predict(image)

            # ---- Result Display ---- #
            st.success(f"✅ Predicted: **{label}**")
            st.metric(label="Confidence", value=f"{round(confidence * 100, 2)}%")

            st.subheader("🔥 Top 3 Predictions")
            labels = [x[0] for x in top3]
            probs  = [x[1] for x in top3]

            for rank, (lbl, prob) in enumerate(zip(labels, probs), start=1):
                st.write(f"**#{rank}** {lbl} → `{round(prob * 100, 2)}%`")
                st.progress(float(prob))

            # ---- Bar Chart ---- #
            fig = px.bar(
                x=labels,
                y=[round(p * 100, 2) for p in probs],
                labels={"x": "Waste Class", "y": "Probability (%)"},
                title="📊 Top 3 Prediction Probabilities",
                color=labels,
                text=[f"{round(p*100,2)}%" for p in probs]
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(showlegend=False, yaxis_range=[0, 110])
            st.plotly_chart(fig, use_container_width=True)

    elif not uploaded_file:
        st.info("👈 Please upload an image to begin detection.")

# ====================== FOOTER ====================== #
st.divider()
st.caption("🤖 Powered by TensorFlow & Streamlit | Waste AI Classification System")

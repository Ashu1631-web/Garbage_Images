import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import plotly.express as px

# ====================== PAGE CONFIG ====================== #
st.set_page_config(
    page_title="Waste Garbage Classification",
    page_icon="♻️",
    layout="wide"
)

# ====================== MODEL LOAD ====================== #
@st.cache_resource
def load_my_model():
    try:
        class PatchedDense(tf.keras.layers.Dense):
            def __init__(self, *args, **kwargs):
                kwargs.pop('quantization_config', None)
                super().__init__(*args, **kwargs)

        with tf.keras.utils.custom_object_scope({'Dense': PatchedDense}):
            model = tf.keras.models.load_model(
                "clean_model.h5",
                compile=False
            )
        return model
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None

# ====================== CLASS NAMES ====================== #
def load_class_names():
    try:
        with open("class_names.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ class_names.json load failed: {e}")
        return []

model = load_my_model()
class_names = load_class_names()

# ====================== SIDEBAR ====================== #
st.sidebar.title("⚙️ Menu")
st.sidebar.success("Model Ready ✅" if model else "Model Error ❌")
st.sidebar.info(f"Total Classes: {len(class_names)}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Supported Waste Types:**")
for name in class_names:
    st.sidebar.write(f"• {name}")

# ====================== TITLE ====================== #
st.title("♻️ Waste Garbage Management System (AI)")
st.markdown("""
### 📌 About This Project
This AI-powered system classifies waste from images into categories:
- 🧴 Plastic | 🔩 Metal | 📄 Paper | 📦 Cardboard
- 🍶 Glass | 🔋 Battery | 👕 Clothes | 🗑️ Trash

Upload an image or use your **Live Webcam** to detect waste type instantly.

👉 Built using **MobileNetV2 + Transfer Learning** (Keras / TensorFlow)
""")
st.divider()

# ====================== PREPROCESS ====================== #
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.resize((160, 160)).convert("RGB")
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# ====================== PREDICT ====================== #
def predict(img: Image.Image):
    try:
        arr = preprocess_image(img)
        preds = model.predict(arr, verbose=0)[0]
        top3_idx = preds.argsort()[-3:][::-1]
        results = [(class_names[int(i)], float(preds[i])) for i in top3_idx]
        return results[0][0], results[0][1], results
    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")
        return "Error", 0.0, []

# ====================== SHOW RESULTS ====================== #
def show_results(image):
    if model is None:
        st.error("❌ Model not loaded.")
        return
    if not class_names:
        st.error("❌ class_names.json not found.")
        return

    with st.spinner("Analyzing..."):
        label, confidence, top3 = predict(image)

    st.success(f"✅ Predicted: **{label.upper()}**")
    st.metric(label="Confidence", value=f"{round(confidence * 100, 2)}%")

    st.subheader("🔥 Top 3 Predictions")
    labels = [x[0] for x in top3]
    probs  = [x[1] for x in top3]

    for rank, (lbl, prob) in enumerate(zip(labels, probs), start=1):
        st.write(f"**#{rank}** {lbl} → `{round(prob * 100, 2)}%`")
        st.progress(float(prob))

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

# ====================== TABS ====================== #
tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Live Webcam"])

with tab1:
    st.header("📤 Upload Image for Detection")
    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload a waste image",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width=280)
            detect_btn = st.button("🔍 Detect Waste", use_container_width=True)
        else:
            detect_btn = False

    with col2:
        if uploaded_file and detect_btn:
            show_results(image)
        elif not uploaded_file:
            st.info("👈 Please upload an image to begin detection.")

with tab2:
    st.header("📷 Live Webcam Detection")
    cam_image = st.camera_input("Point camera at waste item")
    if cam_image:
        col1, col2 = st.columns([1, 2])
        with col1:
            image = Image.open(cam_image).convert("RGB")
            st.image(image, caption="Captured Image", width=280)
        with col2:
            show_results(image)

# ====================== FOOTER ====================== #
st.divider()
st.caption("🤖 Powered by TensorFlow & Streamlit | Waste AI Classification System")

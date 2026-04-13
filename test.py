import streamlit as st
import numpy as np
from keras.models import load_model
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
# Streamlit Cloud mein files repo ke root mein hoti hain
MODEL_PATH = "final_garbage_model.keras"
CLASS_PATH = "class_names.json"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.info("📁 Make sure `final_garbage_model.keras` is in your GitHub repo root.")
        return None
    try:
        model = load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None

@st.cache_data
def load_class_names():
    if not os.path.exists(CLASS_PATH):
        st.error(f"❌ class_names.json not found!")
        return []
    try:
        with open(CLASS_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ JSON load failed: {e}")
        return []

model       = load_my_model()
class_names = load_class_names()

# ====================== SIDEBAR ====================== #
st.sidebar.title("⚙️ Status")
st.sidebar.write("Model:",   "✅ Ready"  if model       else "❌ Not loaded")
st.sidebar.write("Classes:", f"✅ {len(class_names)} found" if class_names else "❌ Not loaded")

if class_names:
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🗑️ Waste Types:**")
    for name in class_names:
        st.sidebar.write(f"• {name}")

# ====================== TITLE ====================== #
st.title("♻️ Waste Garbage Management System (AI)")
st.markdown("""
### 📌 Project Overview
यह AI आधारित सिस्टम image upload करके waste classify करता है।
👉 **Deep Learning model (Keras/TensorFlow)** use किया गया है।
""")
st.divider()

# ====================== STOP IF NOT READY ====================== #
if not model or not class_names:
    st.warning("⚠️ Model ya class names load nahi hua. Upar error check karein.")
    st.markdown("""
    ### 🔧 Fix Kaise Karein:
    - Confirm karein ki `final_garbage_model.keras` GitHub repo ke **root folder** mein hai
    - Confirm karein ki `class_names.json` GitHub repo ke **root folder** mein hai
    - Streamlit Cloud pe **Reboot app** karein
    """)
    st.stop()

# ====================== PREPROCESS ====================== #
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.resize((160, 160))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# ====================== PREDICT ====================== #
def predict(img: Image.Image):
    try:
        img_array = preprocess_image(img)
        prediction = model.predict(img_array)[0]
        top3_idx   = prediction.argsort()[-3:][::-1]
        results    = [(class_names[i], float(prediction[i])) for i in top3_idx]
        return results[0][0], results[0][1], results
    except Exception as e:
        st.error(f"⚠️ Prediction Error: {e}")
        return "Error", 0.0, []

# ====================== MAIN UI ====================== #
st.header("📤 Upload Image for Detection")

col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader(
        "Waste image upload karein",
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
        with st.spinner("🔬 Analyzing image..."):
            label, confidence, top3 = predict(image)

        st.success(f"✅ Predicted: **{label}**")
        st.metric("Confidence", f"{round(confidence * 100, 2)}%")

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

    elif not uploaded_file:
        st.info("👈 Image upload karein aur Detect button dabao.")

# ====================== FOOTER ====================== #
st.divider()
st.caption("🤖 Powered by TensorFlow & Streamlit | Waste AI Classification System")

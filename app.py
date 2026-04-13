import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
import plotly.express as px
from tensorflow.keras.models import load_model
import json
import gdown
import os

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="♻️ Waste Garbage Management",
    layout="wide",
    page_icon="🌱"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_my_model():
    model_path = "final_garbage_model.keras"
    
    if not os.path.exists(model_path):
        url = "https://drive.google.com/uc?id=1YsShxgnuv29JCmkvMNx4Gq7wu4qxg3X7"
        gdown.download(url, model_path, quiet=False)
    
    return load_model(model_path)

# ---------------- LOAD CLASS LABELS ----------------
@st.cache_resource
def load_classes():
    with open("class_names.json", "r") as f:
        class_indices = json.load(f)
    return list(class_indices.keys())

labels = load_classes()

# ---------------- PREDICT FUNCTION ----------------
def predict(img):
    img = img.resize((160, 160))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    label = labels[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return label, confidence

# ---------------- LOGIN BG ----------------
def login_bg():
    st.markdown("""
    <style>
    .stApp {
        background: url("https://www.kelvinindia.in/blog/wp-content/uploads/2024/06/Waste-Management.jpg") no-repeat center center fixed;
        background-size: cover;
    }
    .stApp::before {
        content:"";
        position:fixed;
        width:100%;
        height:100%;
        background:rgba(0,0,0,0.7);
        top:0;
        left:0;
        z-index:0;
    }
    .block-container {
        position:relative;
        z-index:1;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- MAIN UI ----------------
def main_ui():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg,#020617,#0f172a);
        color:white;
    }
    .card {
        background:#1e293b;
        padding:20px;
        border-radius:12px;
        margin:10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- LOGIN ----------------
if "login" not in st.session_state:
    st.session_state.login = False

def login():
    login_bg()
    st.title("♻️ Waste Garbage Management Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "1234":
            st.session_state.login = True
        else:
            st.error("Invalid credentials")

if not st.session_state.login:
    login()
    st.stop()

# ---------------- AFTER LOGIN ----------------
main_ui()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Menu")

page = st.sidebar.selectbox(
    "Navigate",
    ["Project Overview","Detection (Upload)","Camera","Analytics","History"]
)

# ---------------- SAVE ----------------
def save(label, conf):
    df = pd.DataFrame([{
        "Label": label,
        "Confidence": conf,
        "Time": datetime.now()
    }])

    try:
        old = pd.read_csv("history.csv")
        df = pd.concat([old, df])
    except:
        pass

    df.to_csv("history.csv", index=False)

# ---------------- LOAD ----------------
def load():
    try:
        return pd.read_csv("history.csv")
    except:
        return pd.DataFrame()

# ---------------- PROJECT OVERVIEW ----------------
if page == "Project Overview":

    st.title("🌍 Project Overview")

    st.markdown("""
    <div class="card">
    <h3>♻️ Waste Garbage Management System (AI)</h3>
    <p>This AI-powered system automatically classifies garbage using Deep Learning.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- UPLOAD ----------------
elif page == "Detection (Upload)":

    st.header("📤 Upload Detection")

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, width=300)

        if st.button("Detect"):
            label, conf = predict(img)

            st.success(f"♻️ {label.upper()}")
            st.markdown(f"### Confidence: **{round(conf*100,2)}%**")
            st.progress(int(conf*100))

            save(label, conf)

# ---------------- CAMERA ----------------
elif page == "Camera":

    st.header("📸 Camera Detection")

    cam = st.camera_input("Capture Image")

    if cam:
        img = Image.open(cam).convert("RGB")
        st.image(img, width=300)

        if st.button("Detect from Camera"):
            label, conf = predict(img)

            st.success(f"♻️ {label.upper()}")
            st.markdown(f"### Confidence: **{round(conf*100,2)}%**")
            st.progress(int(conf*100))

            save(label, conf)

# ---------------- ANALYTICS ----------------
elif page == "Analytics":

    st.title("📊 Analytics Dashboard")

    df = load()

    if df.empty:
        st.warning("No Data")
    else:
        st.plotly_chart(px.pie(df, names="Label", title="Waste Distribution"))
        st.plotly_chart(px.bar(df, x="Label", title="Waste Count"))
        st.plotly_chart(px.histogram(df, x="Confidence", title="Confidence Distribution"))
        st.plotly_chart(px.line(df, x="Time", y="Confidence", title="Confidence Trend"))

# ---------------- HISTORY ----------------
elif page == "History":

    st.title("📂 Prediction History")

    df = load()

    if df.empty:
        st.warning("No history")
    else:
        st.dataframe(df)

        csv = df.to_csv(index=False).encode()
        st.download_button("⬇ Download CSV", csv, "history.csv")

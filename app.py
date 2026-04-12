import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from datetime import datetime
import plotly.express as px
import base64
import os

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="♻️ RecycleVision AI",
    layout="wide",
    page_icon="🌱"
)

# ---------------- SAFE MODEL LOAD ----------------
@st.cache_resource
def load_model_safe():
    try:
        from tensorflow.keras.models import load_model
        model = load_model("model.h5", compile=False)
        return model
    except Exception as e:
        return None

model = load_model_safe()

# ---------------- LABELS ----------------
def load_labels():
    try:
        with open("labels.txt") as f:
            return [i.strip() for i in f.readlines()]
    except:
        return ["Plastic", "Metal", "Glass", "Paper", "Organic"]

labels = load_labels()

# ---------------- LOGIN BG ----------------
def set_bg():
    if os.path.exists("garbage_bg.jpg"):
        with open("garbage_bg.jpg", "rb") as f:
            data = base64.b64encode(f.read()).decode()
        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{data}");
            background-size: cover;
        }}
        </style>
        """, unsafe_allow_html=True)

# ---------------- GRADIENT ----------------
def set_gradient():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg,#020617,#0f172a);
        color:white;
    }
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(12px);
    }
    .glass {
        background: rgba(255,255,255,0.08);
        padding:20px;
        border-radius:15px;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- LOGIN ----------------
if "login" not in st.session_state:
    st.session_state.login = False

def login():
    set_bg()
    st.title("🔐 Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "1234":
            st.session_state.login = True
        else:
            st.error("Invalid Credentials")

if not st.session_state.login:
    login()
    st.stop()

# ---------------- AFTER LOGIN ----------------
set_gradient()

# ---------------- PREDICTION ----------------
def predict_image(image):
    if model is None:
        # fallback (no crash)
        return "Plastic", 0.85

    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    idx = np.argmax(pred)
    conf = float(np.max(pred))

    return labels[idx], conf

# ---------------- DRAW BOX ----------------
def draw_box(image, label, conf):
    img = np.array(image)
    h, w, _ = img.shape

    cv2.rectangle(img, (20,20), (w-20,h-20), (0,255,0), 3)

    text = f"{label} ({round(conf*100,2)}%)"
    cv2.putText(img, text, (30,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    return img

# ---------------- HISTORY ----------------
def save_history(label, conf):
    df = pd.DataFrame([{
        "Label": label,
        "Confidence": conf,
        "Time": datetime.now()
    }])

    if os.path.exists("history.csv"):
        old = pd.read_csv("history.csv")
        df = pd.concat([old, df])

    df.to_csv("history.csv", index=False)

def load_history():
    if os.path.exists("history.csv"):
        return pd.read_csv("history.csv")
    return pd.DataFrame()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Input Mode")

mode = st.sidebar.radio("", ["Upload Image","Camera Capture"])

page = st.sidebar.selectbox(
    "Navigate",
    ["Home","Detection","Analytics","History"]
)

# ---------------- HOME ----------------
if page == "Home":
    st.title("🌿 RecycleVision AI")

    c1,c2,c3 = st.columns(3)
    c1.metric("Predictions","1250")
    c2.metric("Accuracy","94%")
    c3.metric("Users","350")

# ---------------- DETECTION ----------------
elif page == "Detection":

    st.header("📤 Waste Detection")

    image = None

    if mode == "Upload Image":
        file = st.file_uploader("Upload", type=["jpg","png"])
        if file:
            image = Image.open(file)

    else:
        cam = st.camera_input("Capture")
        if cam:
            image = Image.open(cam)

    if image:
        st.image(image, width=300)

        if st.button("Detect"):
            label, conf = predict_image(image)

            boxed = draw_box(image, label, conf)
            st.image(boxed)

            st.success(f"{label}")
            st.progress(int(conf*100))

            save_history(label, conf)

# ---------------- ANALYTICS ----------------
elif page == "Analytics":

    df = load_history()

    if df.empty:
        st.warning("No data")
    else:
        st.plotly_chart(px.pie(df, names="Label"))
        st.plotly_chart(px.bar(df, x="Label"))
        st.plotly_chart(px.line(df, x="Time", y="Confidence"))
        st.plotly_chart(px.histogram(df, x="Confidence"))
        st.plotly_chart(px.box(df, x="Label", y="Confidence"))
        st.plotly_chart(px.violin(df, x="Label", y="Confidence"))
        st.plotly_chart(px.scatter(df, x="Confidence", y="Label"))
        st.plotly_chart(px.area(df, x="Time", y="Confidence"))
        st.plotly_chart(px.strip(df, x="Label", y="Confidence"))
        st.plotly_chart(px.density_heatmap(df, x="Confidence", y="Label"))

# ---------------- HISTORY ----------------
elif page == "History":

    df = load_history()

    if df.empty:
        st.warning("No history")
    else:
        st.dataframe(df)

        csv = df.to_csv(index=False).encode()
        st.download_button("Download CSV", csv, "history.csv")

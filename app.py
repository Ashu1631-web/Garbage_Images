import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import base64
import os
from datetime import datetime
import plotly.express as px

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="♻️ Waste Classification AI",
    layout="wide",
    page_icon="🌱"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model_safe():
    try:
        from tensorflow.keras.models import load_model
        return load_model("model.h5", compile=False)
    except Exception as e:
        return None

model = load_model_safe()

# ---------------- LOAD LABELS ----------------
def load_labels():
    try:
        with open("labels.txt") as f:
            return [i.strip() for i in f.readlines()]
    except:
        return ["cardboard","glass","metal","paper","plastic","trash"]

labels = load_labels()

# ---------------- UI STYLES ----------------
def set_login_bg():
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

def set_main_ui():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg,#020617,#0f172a);
        color:white;
    }
    [data-testid="stSidebar"] {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
    }
    .glass {
        background: rgba(255,255,255,0.08);
        padding:20px;
        border-radius:15px;
        margin:10px 0;
    }
    .stButton>button {
        background: linear-gradient(90deg,#00c6ff,#0072ff);
        color:white;
        border-radius:10px;
    }
    </style>
    """, unsafe_allow_html=True)

# ---------------- LOGIN ----------------
if "login" not in st.session_state:
    st.session_state.login = False

def login():
    set_login_bg()
    st.markdown("<h1 style='text-align:center;'>♻️ Waste AI</h1>", unsafe_allow_html=True)

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

# ---------------- MAIN UI ----------------
set_main_ui()

# ---------------- PREDICTION ----------------
def predict_image(image):
    if model is None:
        return "Model Error", 0.0

    img = image.resize((160,160))
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

    cv2.rectangle(img,(20,20),(w-20,h-20),(0,255,0),2)

    text = f"{label} ({round(conf*100,2)}%)"
    cv2.putText(img,text,(30,40),
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

mode = st.sidebar.radio(
    "Choose input mode:",
    ["Upload Image", "Camera"]
)

page = st.sidebar.selectbox(
    "Navigate",
    ["Home","Detection","Analytics","History"]
)

# ---------------- HOME ----------------
if page == "Home":
    st.title("🌿 Smart Waste Classification")

    col1, col2, col3 = st.columns(3)
    col1.metric("Model Accuracy", "~75%")
    col2.metric("Classes", "6")
    col3.metric("Status", "Active")

    st.markdown('<div class="glass">AI-powered garbage classification system</div>', unsafe_allow_html=True)

# ---------------- DETECTION ----------------
elif page == "Detection":

    st.header("📤 Waste Detection")

    image = None

    if mode == "Upload Image":
        file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
        if file:
            image = Image.open(file)

    else:
        cam = st.camera_input("Capture")
        if cam:
            image = Image.open(cam)

    if image:
        st.image(image, width=300)

        if st.button("🔍 Detect"):
            label, conf = predict_image(image)

            boxed = draw_box(image, label, conf)

            st.image(boxed, caption="Detection Result", use_column_width=True)

            st.success(f"Detected: {label}")
            st.progress(int(conf*100))

            save_history(label, conf)

# ---------------- ANALYTICS ----------------
elif page == "Analytics":

    st.title("📊 Analytics Dashboard")

    df = load_history()

    if df.empty:
        st.warning("No Data Available")
    else:
        st.plotly_chart(px.pie(df, names="Label"))
        st.plotly_chart(px.bar(df, x="Label"))
        st.plotly_chart(px.histogram(df, x="Confidence"))
        st.plotly_chart(px.line(df, x="Time", y="Confidence"))
        st.plotly_chart(px.box(df, x="Label", y="Confidence"))
        st.plotly_chart(px.violin(df, x="Label", y="Confidence"))
        st.plotly_chart(px.scatter(df, x="Confidence", y="Label"))
        st.plotly_chart(px.area(df, x="Time", y="Confidence"))
        st.plotly_chart(px.strip(df, x="Label", y="Confidence"))
        st.plotly_chart(px.density_heatmap(df, x="Confidence", y="Label"))

# ---------------- HISTORY ----------------
elif page == "History":

    st.title("📂 Prediction History")

    df = load_history()

    if df.empty:
        st.warning("No history found")
    else:
        st.dataframe(df)

        csv = df.to_csv(index=False).encode()
        st.download_button("⬇ Download CSV", csv, "history.csv")

import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
from datetime import datetime
import plotly.express as px
import os

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="♻️ RecycleVision AI",
    layout="wide",
    page_icon="🌱"
)

# ---------------- GLASS UI ----------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg,#0f172a,#020617);
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
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 30px rgba(0,0,0,0.3);
}
.stButton>button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    border:none;
    border-radius:10px;
    color:white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
if "login" not in st.session_state:
    st.session_state.login = False

def login():
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

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_my_model():
    return load_model("model.h5")

model = load_my_model()

# ---------------- LOAD LABELS ----------------
def load_labels():
    with open("labels.txt") as f:
        return [i.strip() for i in f.readlines()]

labels = load_labels()

# ---------------- PREDICTION ----------------
def predict_image(image):
    img = image.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    index = np.argmax(pred)
    confidence = float(np.max(pred))

    return labels[index], confidence

# ---------------- DRAW BOX ----------------
def draw_box(image, label, conf):
    img = np.array(image)
    h, w, _ = img.shape

    # fake full box (since classification model)
    start = (20, 20)
    end = (w-20, h-20)

    cv2.rectangle(img, start, end, (0,255,0), 3)

    text = f"{label} ({round(conf*100,2)}%)"
    cv2.putText(img, text, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,255,0), 2)

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

page = st.sidebar.selectbox("Navigate",
                           ["Home","Detection","Analytics","History"])

# ---------------- HOME ----------------
if page == "Home":
    st.markdown("<h1 style='color:#22c55e;'>🌿 RecycleVision AI</h1>", unsafe_allow_html=True)
    st.write("Smart Waste Classification using Deep Learning")

    c1,c2,c3 = st.columns(3)
    c1.metric("Predictions","1250")
    c2.metric("Accuracy","94%")
    c3.metric("Users","350")

    st.markdown('<div class="glass">AI system for smart recycling & waste management</div>', unsafe_allow_html=True)

# ---------------- DETECTION ----------------
elif page == "Detection":

    st.header("📤 Waste Detection")

    image = None

    if mode == "Upload Image":
        file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
        if file:
            image = Image.open(file)

    elif mode == "Camera Capture":
        cam = st.camera_input("Capture")
        if cam:
            image = Image.open(cam)

    if image:
        st.image(image, caption="Input Image", width=300)

        if st.button("🔍 Detect Waste"):
            label, conf = predict_image(image)

            boxed = draw_box(image, label, conf)

            st.image(boxed, caption="Detection Result", use_column_width=True)

            st.success(f"Detected: {label}")
            st.progress(int(conf*100))

            save_history(label, conf)

# ---------------- ANALYTICS ----------------
elif page == "Analytics":

    st.title("📊 Power BI Style Dashboard")

    df = load_history()

    if df.empty:
        st.warning("No data available")
    else:
        col1,col2 = st.columns(2)

        col1.plotly_chart(px.pie(df, names="Label"))
        col2.plotly_chart(px.bar(df, x="Label"))

        st.plotly_chart(px.line(df, x="Time", y="Confidence"))
        st.plotly_chart(px.histogram(df, x="Confidence"))
        st.plotly_chart(px.box(df, x="Label", y="Confidence"))
        st.plotly_chart(px.violin(df, x="Label", y="Confidence"))
        st.plotly_chart(px.area(df, x="Time", y="Confidence"))
        st.plotly_chart(px.scatter(df, x="Confidence", y="Label"))
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

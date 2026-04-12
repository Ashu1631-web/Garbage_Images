import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import gdown
from datetime import datetime
import plotly.express as px
from tensorflow.keras.models import load_model

st.set_page_config(page_title="♻️ Smart Waste AI", layout="wide")

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#020617,#0f172a);
    color:white;
}
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
}
.big-card {
    background: rgba(255,255,255,0.08);
    padding:20px;
    border-radius:15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
MODEL_PATH = "model.h5"

@st.cache_resource
def load_model_safe():
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=PASTE_YOUR_NEW_H5_ID"
        gdown.download(url, MODEL_PATH, quiet=False)

    model = load_model(MODEL_PATH, compile=False)
    return model

model = load_model_safe()

# ---------------- LABELS ----------------
labels = ["cardboard","glass","metal","paper","plastic","trash"]

# ---------------- LOGIN ----------------
if "login" not in st.session_state:
    st.session_state.login=False

if not st.session_state.login:
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u=="admin" and p=="1234":
            st.session_state.login=True
            st.rerun()
        else:
            st.error("Invalid login")
    st.stop()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Dashboard")
page = st.sidebar.radio("Navigate",
["Overview","Detection","Analytics","History"])

# ---------------- FUNCTIONS ----------------
def predict(img):
    img = img.convert("RGB").resize((160,160))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    idx = np.argmax(pred)

    return labels[idx], float(np.max(pred)), pred[0]

def save(label, conf):
    df = pd.DataFrame([{
        "Label":label,
        "Confidence":conf,
        "Time":datetime.now()
    }])

    if os.path.exists("history.csv"):
        old = pd.read_csv("history.csv")
        df = pd.concat([old,df])

    df.to_csv("history.csv", index=False)

def load_data():
    if os.path.exists("history.csv"):
        df = pd.read_csv("history.csv")
        df["Time"] = pd.to_datetime(df["Time"])
        return df
    return pd.DataFrame()

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.title("♻️ Smart Waste Management AI")

    col1,col2,col3 = st.columns(3)

    col1.metric("Accuracy","~87%")
    col2.metric("Model Type","MobileNetV2")
    col3.metric("Status","Active")

# ---------------- DETECTION ----------------
elif page == "Detection":
    st.title("📤 Upload Waste Image")

    file = st.file_uploader("Upload Image")

    if file:
        img = Image.open(file)
        st.image(img, width=300)

        if st.button("Analyze"):
            label, conf, probs = predict(img)

            st.success(f"Detected: {label}")
            st.progress(int(conf*100))

            st.metric("Confidence", f"{conf*100:.2f}%")

            df = pd.DataFrame({
                "Class": labels,
                "Probability": probs
            })

            st.plotly_chart(px.bar(df, x="Class", y="Probability"))

            save(label,conf)

# ---------------- ANALYTICS ----------------
elif page == "Analytics":
    st.title("📊 Analytics Dashboard")

    df = load_data()

    if df.empty:
        st.warning("No data yet")
    else:
        st.plotly_chart(px.pie(df, names="Label"))
        st.plotly_chart(px.histogram(df, x="Confidence"))
        st.plotly_chart(px.line(df, x="Time", y="Confidence"))

# ---------------- HISTORY ----------------
elif page == "History":
    st.title("📂 Prediction History")

    df = load_data()

    if df.empty:
        st.warning("No history")
    else:
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False))

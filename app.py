import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import base64
import os
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="♻️ Waste AI", layout="wide")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model_safe():
    try:
        from tensorflow.keras.models import load_model
        return load_model("model.h5", compile=False)
    except:
        return None

model = load_model_safe()

def load_labels():
    try:
        with open("labels.txt") as f:
            return [i.strip() for i in f.readlines()]
    except:
        return ["cardboard","glass","metal","paper","plastic","trash"]

labels = load_labels()

# ---------------- LOGIN BACKGROUND (FINAL FIX) ----------------
def set_login_bg():
    file_path = "garbage_bg.jpg"

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()

        st.markdown(f"""
        <style>
        html, body, .stApp {{
            height: 100%;
        }}

        .stApp {{
            background: url("https://www.kelvinindia.in/blog/wp-content/uploads/2024/06/Waste-Management.jpg,{encoded}") no-repeat center center fixed;
            background-size: cover;
        }}

        /* DARK OVERLAY */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.7);
            z-index: 0;
        }}

        /* CONTENT ABOVE OVERLAY */
        .block-container {{
            position: relative;
            z-index: 1;
        }}
        </style>
        """, unsafe_allow_html=True)

# ---------------- MAIN UI ----------------
def set_ui():
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
    .card {
        background: rgba(255,255,255,0.08);
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
    set_login_bg()  # ✅ ONLY login page bg
    st.title("♻️ Waste AI Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u == "admin" and p == "1234":
            st.session_state.login = True
        else:
            st.error("Invalid login")

if not st.session_state.login:
    login()
    st.stop()

# ---------------- AFTER LOGIN ----------------
set_ui()  # ✅ gradient UI after login

# ---------------- PREDICTION ----------------
def predict(image):
    if model is None:
        return "Error", 0

    img = image.resize((160,160))
    img = np.array(img)/255.0
    img = np.expand_dims(img,0)

    pred = model.predict(img)
    idx = np.argmax(pred)
    conf = float(np.max(pred))
    return labels[idx], conf

def draw(image, label, conf):
    img = np.array(image)
    h, w, _ = img.shape

    cv2.rectangle(img,(20,20),(w-20,h-20),(0,255,0),2)
    cv2.putText(img,f"{label} {round(conf*100,2)}%",
                (30,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    return img

# ---------------- HISTORY ----------------
def save(label, conf):
    df = pd.DataFrame([{
        "Label": label,
        "Confidence": conf,
        "Time": datetime.now()
    }])

    if os.path.exists("history.csv"):
        old = pd.read_csv("history.csv")
        df = pd.concat([old, df])

    df.to_csv("history.csv", index=False)

def load():
    if os.path.exists("history.csv"):
        return pd.read_csv("history.csv")
    return pd.DataFrame()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Menu")

mode = st.sidebar.radio("Input", ["Upload", "Camera"])
page = st.sidebar.selectbox("Navigate",
["Overview","Detection","Analytics","History"])

# ---------------- OVERVIEW ----------------
if page == "Overview":
    st.title("🌍 Project Overview")

    st.markdown("""
    <div class="card">
    <h3>♻️ Waste Classification AI</h3>
    <p>This AI model classifies waste into 6 categories using deep learning.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- DETECTION ----------------
elif page == "Detection":

    img = None

    if mode == "Upload":
        f = st.file_uploader("Upload Image")
        if f:
            img = Image.open(f)
    else:
        c = st.camera_input("Capture")
        if c:
            img = Image.open(c)

    if img:
        st.image(img, width=300)

        if st.button("Detect"):
            label, conf = predict(img)
            st.image(draw(img,label,conf))
            st.success(label)
            st.progress(int(conf*100))
            save(label,conf)

# ---------------- ANALYTICS ----------------
elif page == "Analytics":

    df = load()

    if df.empty:
        st.warning("No data")
    else:
        st.plotly_chart(px.pie(df,names="Label"))
        st.plotly_chart(px.bar(df,x="Label"))
        st.plotly_chart(px.histogram(df,x="Confidence"))
        st.plotly_chart(px.line(df,x="Time",y="Confidence"))
        st.plotly_chart(px.box(df,x="Label",y="Confidence"))
        st.plotly_chart(px.violin(df,x="Label",y="Confidence"))
        st.plotly_chart(px.scatter(df,x="Confidence",y="Label"))
        st.plotly_chart(px.area(df,x="Time",y="Confidence"))
        st.plotly_chart(px.strip(df,x="Label",y="Confidence"))
        st.plotly_chart(px.density_heatmap(df,x="Confidence",y="Label"))

# ---------------- HISTORY ----------------
elif page == "History":

    df = load()

    if df.empty:
        st.warning("No history")
    else:
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), "history.csv")

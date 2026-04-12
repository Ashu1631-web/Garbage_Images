import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="♻️ Garbage Waste Management", layout="wide")

# ---------------- MODEL (FIXED FOR KERAS 3) ----------------
@st.cache_resource
def load_model_safe():
    try:
        from keras.models import load_model  # 🔥 IMPORTANT CHANGE

        model = load_model("model_fixed.h5", compile=False)

        st.success("✅ Model Loaded Successfully")
        return model

    except Exception as e:
        st.error(f"❌ Model Load Error: {e}")
        return None

model = load_model_safe()

def load_labels():
    try:
        with open("labels.txt") as f:
            return [i.strip() for i in f.readlines()]
    except:
        return ["cardboard","glass","metal","paper","plastic","trash"]

labels = load_labels()

# ---------------- LOGIN BG ----------------
def set_login_bg():
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
    st.session_state.login=False

def login():
    set_login_bg()
    st.title("♻️ Garbage Waste Management Login")

    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u=="admin" and p=="1234":
            st.session_state.login=True
        else:
            st.error("Invalid login")

if not st.session_state.login:
    login()
    st.stop()

# ---------------- AFTER LOGIN ----------------
set_ui()

# ---------------- PREDICTION (FIXED) ----------------
def predict(image):
    if model is None:
        return "Model Not Loaded", 0.0

    try:
        img = image.resize((160,160))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)
        idx = np.argmax(pred)
        conf = float(np.max(pred))

        return labels[idx], conf

    except Exception as e:
        st.error(f"Prediction Error: {e}")
        return "Prediction Error", 0.0

def draw(image, label, conf):
    img = np.array(image)
    h,w,_ = img.shape

    cv2.rectangle(img,(20,20),(w-20,h-20),(0,255,0),2)

    text = f"{label} ({round(conf*100,2)}%)" if conf > 0 else "No Prediction"

    cv2.putText(img,text,(30,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    return img

# ---------------- HISTORY ----------------
def save(label,conf):
    df=pd.DataFrame([{
        "Label":label,
        "Confidence":conf,
        "Time":datetime.now()
    }])
    if os.path.exists("history.csv"):
        old=pd.read_csv("history.csv")
        df=pd.concat([old,df])
    df.to_csv("history.csv",index=False)

def load():
    if os.path.exists("history.csv"):
        return pd.read_csv("history.csv")
    return pd.DataFrame()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Menu")

page = st.sidebar.selectbox("Navigate",
["Overview","Detection (Upload)","Camera","Analytics","History"])

# ---------------- OVERVIEW ----------------
if page=="Overview":
    st.title("🌍 Project Overview")

    st.markdown("""
    <div class="card">
    <h3>♻️ Garbage Waste Management System (AI)</h3>

    <h4>Project Description</h4>
    <p>This is an AI-powered application designed to automatically identify and classify different types of garbage. 
    By using Deep Learning, the system helps in sorting waste quickly, making the recycling process more efficient and eco-friendly.</p>

    <h4>🌟 Key Features</h4>
    <ul>
    <li><b>Instant Detection:</b> Uses a trained AI model to identify waste from uploaded photos or live camera captures.</li>
    <li><b>6-Way Classification:</b> Categorizes waste into Cardboard, Glass, Metal, Paper, Plastic, and Trash.</li>
    <li><b>Smart Analytics:</b> Visualizes waste data using interactive charts.</li>
    <li><b>Secure Access:</b> Login system for authorized users.</li>
    <li><b>Data History:</b> Saves scans and allows CSV download.</li>
    </ul>

    <h4>🛠️ Technologies Used</h4>
    <ul>
    <li>Python</li>
    <li>Streamlit</li>
    <li>TensorFlow / Keras</li>
    <li>OpenCV</li>
    <li>Plotly</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------------- DETECTION (UPLOAD) ----------------
elif page=="Detection (Upload)":

    st.title("📤 Upload Detection")

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file)
        st.image(img, width=300)

        if st.button("Detect"):
            label, conf = predict(img)
            st.image(draw(img,label,conf))
            st.success(label)
            st.progress(int(conf*100))
            save(label,conf)

# ---------------- CAMERA ----------------
elif page=="Camera":

    st.title("📸 Camera Detection")

    cam = st.camera_input("Capture Image")

    if cam:
        img = Image.open(cam)
        st.image(img, width=300)

        if st.button("Detect from Camera"):
            label, conf = predict(img)
            st.image(draw(img,label,conf))
            st.success(label)
            st.progress(int(conf*100))
            save(label,conf)

# ---------------- ANALYTICS ----------------
elif page=="Analytics":

    st.title("📊 Analytics Dashboard")

    df = load()

    if df.empty:
        st.warning("No data available")
    else:
        st.plotly_chart(px.pie(df,names="Label",title="Waste Distribution"))
        st.plotly_chart(px.bar(df,x="Label",title="Count by Category"))
        st.plotly_chart(px.histogram(df,x="Confidence",title="Confidence Distribution"))
        st.plotly_chart(px.line(df,x="Time",y="Confidence",title="Confidence Over Time"))
        st.plotly_chart(px.box(df,x="Label",y="Confidence",title="Confidence Spread"))
        st.plotly_chart(px.violin(df,x="Label",y="Confidence",title="Density"))
        st.plotly_chart(px.scatter(df,x="Confidence",y="Label",title="Scatter Plot"))
        st.plotly_chart(px.area(df,x="Time",y="Confidence",title="Trend Analysis"))
        st.plotly_chart(px.strip(df,x="Label",y="Confidence",title="Strip Plot"))
        st.plotly_chart(px.density_heatmap(df,x="Confidence",y="Label",title="Heatmap"))

# ---------------- HISTORY ----------------
elif page=="History":

    st.title("📂 History")

    df = load()

    if df.empty:
        st.warning("No history")
    else:
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), "history.csv")

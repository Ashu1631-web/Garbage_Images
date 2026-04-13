import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
import plotly.express as px

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="♻️ Waste Classification App",
    layout="wide",
    page_icon="🌱"
)

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
    st.title("♻️ Waste AI Login")
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

# ---------------- FAKE MODEL ----------------
labels = ["Organic", "Plastic", "Metal", "Glass", "Paper"]

def predict(img):
    label = np.random.choice(labels)
    confidence = round(np.random.uniform(0.75, 0.98), 2)
    return label, confidence

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
    <h3>♻️ Smart Waste Management System (AI)</h3>

    <p>This AI-powered system automatically classifies garbage into different categories using Deep Learning. 
    It helps improve waste sorting efficiency and supports eco-friendly recycling.</p>

    <h4>🌟 Key Features</h4>
    <ul>
    <li>Image-based waste detection</li>
    <li>Camera live detection</li>
    <li>5-category classification</li>
    <li>Analytics dashboard</li>
    <li>History tracking + CSV export</li>
    </ul>

    <h4>🛠️ Tech Stack</h4>
    <ul>
    <li>Python</li>
    <li>Streamlit</li>
    <li>TensorFlow</li>
    <li>OpenCV</li>
    <li>Plotly</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# ---------------- UPLOAD ----------------
elif page == "Detection (Upload)":

    st.header("📤 Upload Detection")

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file)
        st.image(img, width=300)

        if st.button("Detect"):
            label, conf = predict(img)
            st.success(label)
            st.progress(int(conf*100))
            save(label, conf)

# ---------------- CAMERA ----------------
elif page == "Camera":

    st.header("📸 Camera Detection")

    cam = st.camera_input("Capture Image")

    if cam:
        img = Image.open(cam)
        st.image(img, width=300)

        if st.button("Detect from Camera"):
            label, conf = predict(img)
            st.success(label)
            st.progress(int(conf*100))
            save(label, conf)

# ---------------- ANALYTICS ----------------
elif page == "Analytics":

    st.title("📊 Analytics")

    df = load()

    if df.empty:
        st.warning("No Data")
    else:
        st.plotly_chart(px.pie(df, names="Label"))
        st.plotly_chart(px.bar(df, x="Label"))
        st.plotly_chart(px.histogram(df, x="Confidence"))
        st.plotly_chart(px.line(df, x="Time", y="Confidence"))

# ---------------- HISTORY ----------------
elif page == "History":

    st.title("📂 History")

    df = load()

    if df.empty:
        st.warning("No history")
    else:
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), "history.csv")

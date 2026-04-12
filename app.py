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

# ---------------- CSS ----------------
st.markdown("""
<style>
body {
    background: #0f172a;
    color: white;
}
.sidebar .sidebar-content {
    background: #111827;
}
.stButton>button {
    background: linear-gradient(90deg,#00c6ff,#0072ff);
    color:white;
    border-radius:10px;
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
    st.title("🔐 Login")
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

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Input Mode")

mode = st.sidebar.radio(
    "Choose input mode:",
    ["Upload Image", "Camera Capture"]
)

page = st.sidebar.selectbox(
    "Navigate",
    ["Home", "Detection", "Analytics", "History"]
)

# ---------------- FAKE MODEL ----------------
labels = ["Organic", "Plastic", "Metal", "Glass", "Paper"]

def predict(img):
    label = np.random.choice(labels)
    confidence = round(np.random.uniform(0.75, 0.98), 2)
    return label, confidence

# ---------------- SAVE HISTORY ----------------
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

# ---------------- LOAD HISTORY ----------------
def load():
    try:
        return pd.read_csv("history.csv")
    except:
        return pd.DataFrame()

# ---------------- HOME ----------------
if page == "Home":
    st.markdown("<h1 style='color:#22c55e;'>🌿 Waste Classification App</h1>", unsafe_allow_html=True)
    st.write("Classify waste as Organic or Inorganic using AI")

    col1, col2, col3 = st.columns(3)
    col1.metric("Predictions", "1250")
    col2.metric("Accuracy", "92%")
    col3.metric("Users", "340")

    st.markdown("---")
    st.info("Powered by TensorFlow, OpenCV, Streamlit")

# ---------------- DETECTION ----------------
elif page == "Detection":

    st.header("📤 Upload Image")

    image = None

    if mode == "Upload Image":
        file = st.file_uploader("Upload", type=["jpg","png","jpeg"])
        if file:
            image = Image.open(file)

    elif mode == "Camera Capture":
        cam = st.camera_input("Capture")
        if cam:
            image = Image.open(cam)

    if image:
        st.image(image, width=300)

        if st.button("🔍 Predict"):
            label, conf = predict(image)

            st.success(f"Prediction: {label}")
            st.progress(int(conf*100))

            save(label, conf)

# ---------------- ANALYTICS ----------------
elif page == "Analytics":

    st.title("📊 Analytics Dashboard")

    df = load()

    if df.empty:
        st.warning("No Data Available")
    else:
        st.plotly_chart(px.pie(df, names="Label"))
        st.plotly_chart(px.bar(df, x="Label"))
        st.plotly_chart(px.histogram(df, x="Confidence"))
        st.plotly_chart(px.line(df, x="Time", y="Confidence"))

        # EXTRA GRAPHS
        st.plotly_chart(px.scatter(df, x="Confidence", y="Label"))
        st.plotly_chart(px.box(df, x="Label", y="Confidence"))
        st.plotly_chart(px.violin(df, x="Label", y="Confidence"))
        st.plotly_chart(px.area(df, x="Time", y="Confidence"))
        st.plotly_chart(px.density_heatmap(df, x="Confidence", y="Label"))
        st.plotly_chart(px.strip(df, x="Label", y="Confidence"))

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

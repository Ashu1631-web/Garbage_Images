import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="♻️ Garbage Waste Management", layout="wide")

# ---------------- MODEL ----------------
MODEL_PATH = "model.keras"

@st.cache_resource
def load_model_safe():
    from tensorflow.keras.models import load_model
    return load_model(MODEL_PATH, compile=False)

model = load_model_safe()

# ---------------- LABELS ----------------
def load_labels():
    try:
        with open("labels.txt") as f:
            return [i.strip() for i in f.readlines()]
    except:
        return ["cardboard","glass","metal","paper","plastic","trash"]

labels = load_labels()

# ---------------- LOGIN ----------------
if "login" not in st.session_state:
    st.session_state.login=False

def login():
    st.title("♻️ Garbage Waste Management")
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

# ---------------- PREDICT ----------------
def predict(image):
    img = image.convert("RGB").resize((160,160))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img, verbose=0)
    idx = np.argmax(pred)

    return labels[idx], float(np.max(pred)), pred[0]

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
        df = pd.read_csv("history.csv")
        df["Time"] = pd.to_datetime(df["Time"])
        return df
    return pd.DataFrame()

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Menu")
page = st.sidebar.selectbox("Navigate",
["Overview","Detection (Upload)","Camera","Analytics","History"])

# ---------------- OVERVIEW ----------------
if page=="Overview":
    st.title("🌍 Project Overview")
    st.info("AI-based garbage classification system")

# ---------------- DETECTION ----------------
elif page=="Detection (Upload)":

    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if file:
        img = Image.open(file)
        st.image(img, width=300)

        if st.button("Detect"):
            label, conf, probs = predict(img)

            st.success(f"Prediction: {label}")
            st.info(f"Confidence: {conf*100:.2f}%")
            st.progress(int(conf*100))

            # Probability chart
            prob_df = pd.DataFrame({
                "Class": labels,
                "Probability": probs
            })
            st.plotly_chart(px.bar(prob_df, x="Class", y="Probability", title="Class Probabilities"))

            save(label,conf)

# ---------------- CAMERA ----------------
elif page=="Camera":

    cam = st.camera_input("Capture")

    if cam:
        img = Image.open(cam)
        st.image(img, width=300)

        if st.button("Detect"):
            label, conf, _ = predict(img)

            st.success(label)
            st.info(f"{conf*100:.2f}%")
            st.progress(int(conf*100))

            save(label,conf)

# ---------------- ANALYTICS ----------------
elif page=="Analytics":

    st.title("📊 Advanced Analytics Dashboard")

    df = load()

    if df.empty:
        st.warning("No data available")
    else:

        st.plotly_chart(px.pie(df, names="Label", title="1. Waste Distribution"))
        st.plotly_chart(px.bar(df, x="Label", title="2. Count by Category"))
        st.plotly_chart(px.histogram(df, x="Confidence", title="3. Confidence Distribution"))
        st.plotly_chart(px.box(df, y="Confidence", title="4. Confidence Box"))
        st.plotly_chart(px.line(df, x="Time", y="Confidence", title="5. Confidence Over Time"))
        st.plotly_chart(px.scatter(df, x="Time", y="Confidence", color="Label", title="6. Scatter"))
        st.plotly_chart(px.area(df, x="Time", y="Confidence", title="7. Area"))
        st.plotly_chart(px.violin(df, y="Confidence", box=True, title="8. Violin"))
        st.plotly_chart(px.density_heatmap(df, x="Label", y="Confidence", title="9. Heatmap"))
        st.plotly_chart(px.ecdf(df, x="Confidence", title="10. ECDF"))
        st.plotly_chart(px.strip(df, x="Label", y="Confidence", title="11. Strip"))
        st.plotly_chart(px.funnel(df, x="Confidence", y="Label", title="12. Funnel"))

        agg = df.groupby("Label")["Confidence"].mean().reset_index()

        st.plotly_chart(px.bar(agg, x="Label", y="Confidence", title="13. Avg Confidence"))
        st.plotly_chart(px.line(agg, x="Label", y="Confidence", title="14. Trend"))
        st.plotly_chart(px.scatter(agg, x="Label", y="Confidence", size="Confidence", title="15. Bubble"))

# ---------------- HISTORY ----------------
elif page=="History":

    df = load()

    if df.empty:
        st.warning("No history")
    else:
        st.dataframe(df)
        st.download_button("Download CSV", df.to_csv(index=False), "history.csv")

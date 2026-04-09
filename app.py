import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RecycleVision Pro", layout="wide")

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}
h1, h2, h3 {
    color: #00FFAA;
}
.stButton>button {
    background-color: #00FFAA;
    color: black;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN SYSTEM ----------------
users = {
    "admin": "1234",
    "ashish": "pass123"
}

def login():
    st.sidebar.title("🔐 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username in users and users[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["user"] = username
        else:
            st.sidebar.error("Invalid Credentials")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

st.sidebar.success(f"Welcome {st.session_state['user']} 👋")

# ---------------- SIDEBAR NAV ----------------
menu = st.sidebar.radio("Navigation", [
    "🏠 Project Overview",
    "📷 Prediction",
    "📊 Analytics Dashboard"
])

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ---------------- DATASET ANALYTICS ----------------
def load_dataset_stats(path="dataset"):
    data = []
    for category in os.listdir(path):
        category_path = os.path.join(path, category)
        if os.path.isdir(category_path):
            count = len(os.listdir(category_path))
            data.append([category, count])
    return pd.DataFrame(data, columns=["Category", "Count"])

# ---------------- HISTORY ----------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# ---------------- OVERVIEW ----------------
if menu == "🏠 Project Overview":
    st.title("♻️ RecycleVision Pro")

    st.write("""
    AI-powered garbage classification system using Deep Learning.
    Helps automate waste segregation and supports smart recycling.
    """)

    col1, col2, col3 = st.columns(3)

    col1.metric("Model", "MobileNetV2")
    col2.metric("Accuracy", "85%+")
    col3.metric("Classes", "6")

    st.markdown("### 💡 Features")
    st.write("""
    - Deep Learning Image Classification  
    - Real-time Predictions  
    - Dataset Analytics Dashboard  
    - Download Prediction History  
    """)

# ---------------- PREDICTION ----------------
elif menu == "📷 Prediction":
    st.title("📷 Upload & Predict")

    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file).resize((224, 224))
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img = np.array(image) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)[0]

        pred_class = classes[np.argmax(pred)]
        confidence = np.max(pred)

        st.success(f"Prediction: {pred_class}")
        st.info(f"Confidence: {confidence:.2f}")

        # Save history
        st.session_state["history"].append({
            "Time": datetime.now(),
            "Prediction": pred_class,
            "Confidence": float(confidence)
        })

        # Top 3 predictions
        st.subheader("Top 3 Predictions")
        top3 = sorted(zip(classes, pred), key=lambda x: x[1], reverse=True)[:3]
        df_top3 = pd.DataFrame(top3, columns=["Class", "Probability"])

        fig = px.bar(df_top3, x="Class", y="Probability", title="Top Predictions")
        st.plotly_chart(fig, use_container_width=True)

# ---------------- ANALYTICS ----------------
elif menu == "📊 Analytics Dashboard":
    st.title("📊 Dataset Analytics")

    df = load_dataset_stats()

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.pie(df, values='Count', names='Category', title='Distribution')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(df, x='Category', y='Count', title='Category Count')
        st.plotly_chart(fig2, use_container_width=True)

    # More graphs
    st.subheader("Advanced Insights")

    fig3 = px.line(df, x='Category', y='Count', title='Trend')
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.scatter(df, x='Category', y='Count', size='Count', title='Scatter')
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.area(df, x='Category', y='Count', title='Area Chart')
    st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.funnel(df, x='Count', y='Category', title='Funnel View')
    st.plotly_chart(fig6, use_container_width=True)

    # ---------------- HISTORY ----------------
    st.subheader("📜 Prediction History")

    if st.session_state["history"]:
        hist_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist_df)

        csv = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download History", csv, "history.csv")

    else:
        st.info("No predictions yet.")

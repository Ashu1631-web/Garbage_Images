import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import os
from datetime import datetime

# 🔥 IMPORTANT: Use Keras instead of TensorFlow
import tensorflow as tf
model = tf.keras.models.load_model("model.h5")

# ---------------- CONFIG ----------------
st.set_page_config(page_title="RecycleVision Pro", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {background-color: #0E1117; color: white;}
h1, h2, h3 {color: #00FFAA;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
users = {"admin": "1234", "ashish": "pass123"}

def login():
    st.title("🔐 Login - RecycleVision")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in users and users[username] == password:
            st.session_state["logged_in"] = True
            st.session_state["user"] = username
        else:
            st.error("Invalid Credentials")

if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login()
    st.stop()

# ---------------- HEADER ----------------
st.title("♻️ RecycleVision Pro Dashboard")
st.success(f"Welcome {st.session_state['user']} 👋")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_ml_model():
    return load_model("model.h5")

model = load_ml_model()

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ---------------- DATA ----------------
def load_dataset_stats(path="dataset"):
    data = []
    if not os.path.exists(path):
        return pd.DataFrame(columns=["Category", "Count"])

    for category in os.listdir(path):
        category_path = os.path.join(path, category)
        if os.path.isdir(category_path):
            count = len(os.listdir(category_path))
            data.append([category, count])
    return pd.DataFrame(data, columns=["Category", "Count"])

df = load_dataset_stats()

# ---------------- HISTORY ----------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# ---------------- NAVIGATION ----------------
tab1, tab2, tab3 = st.tabs(["🏠 Overview", "📊 Analytics", "📷 Prediction"])

# ==================================================
# 🏠 OVERVIEW
# ==================================================
with tab1:
    st.subheader("📌 Project Overview")

    st.write("""
    AI-based garbage classification system using Deep Learning.
    Automates waste segregation and improves recycling efficiency.
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Model", "MobileNetV2")
    col2.metric("Accuracy", "85%+")
    col3.metric("Classes", len(classes))
    col4.metric("Dataset Size", int(df["Count"].sum()) if not df.empty else 0)

# ==================================================
# 📊 ANALYTICS (WITH FILTER)
# ==================================================
with tab2:
    st.subheader("📊 Analytics Dashboard")

    if df.empty:
        st.warning("Dataset folder not found or empty.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            selected_categories = st.multiselect(
                "Select Categories",
                df["Category"].unique(),
                default=df["Category"].unique()
            )

        with col2:
            min_count = st.slider(
                "Minimum Count",
                int(df["Count"].min()),
                int(df["Count"].max()),
                int(df["Count"].min())
            )

        filtered_df = df[
            (df["Category"].isin(selected_categories)) &
            (df["Count"] >= min_count)
        ]

        st.markdown("### 📈 Insights")

        fig1 = px.pie(filtered_df, values='Count', names='Category')
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(filtered_df, x='Category', y='Count')
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.line(filtered_df, x='Category', y='Count')
        st.plotly_chart(fig3, use_container_width=True)

        fig4 = px.scatter(filtered_df, x='Category', y='Count', size='Count')
        st.plotly_chart(fig4, use_container_width=True)

# ==================================================
# 📷 PREDICTION
# ==================================================
with tab3:
    st.subheader("📷 Upload Image")

    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file).resize((224, 224))
        st.image(image, caption="Uploaded Image")

        img = np.array(image) / 255.0
        img = np.expand_dims(img, axis=0)

        pred = model.predict(img)[0]

        pred_class = classes[np.argmax(pred)]
        confidence = float(np.max(pred))

        st.success(f"Prediction: {pred_class}")
        st.info(f"Confidence: {confidence:.2f}")

        # Save history
        st.session_state["history"].append({
            "Time": datetime.now(),
            "Prediction": pred_class,
            "Confidence": confidence
        })

        # Top 3
        top3 = sorted(zip(classes, pred), key=lambda x: x[1], reverse=True)[:3]
        df_top3 = pd.DataFrame(top3, columns=["Class", "Probability"])

        fig = px.bar(df_top3, x="Class", y="Probability", title="Top Predictions")
        st.plotly_chart(fig, use_container_width=True)

    # -------- HISTORY --------
    st.markdown("### 📜 Prediction History")

    if st.session_state["history"]:
        hist_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist_df)

        csv = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download History", csv, "history.csv")

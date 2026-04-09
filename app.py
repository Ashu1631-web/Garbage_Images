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

# ---------------- CSS ----------------
st.markdown("""
<style>
.main {background-color: #0E1117; color: white;}
.block-container {padding-top: 2rem;}
h1, h2, h3 {color: #00FFAA;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGIN ----------------
users = {"admin": "1234", "ashish": "pass123"}

def login():
    st.markdown("## 🔐 Login to RecycleVision")
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
st.markdown(f"# ♻️ RecycleVision Pro Dashboard")
st.success(f"Welcome {st.session_state['user']} 👋")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# ---------------- DATA ----------------
def load_dataset_stats(path="dataset"):
    data = []
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

# ---------------- NAVIGATION (TABS) ----------------
tab1, tab2, tab3 = st.tabs(["🏠 Overview", "📊 Analytics", "📷 Prediction"])

# ==================================================
# 🏠 OVERVIEW (DEFAULT FIRST SCREEN)
# ==================================================
with tab1:
    st.subheader("📌 Project Overview")

    st.write("""
    RecycleVision is an AI-powered garbage classification system using Deep Learning.
    It helps automate waste segregation and improve recycling efficiency.
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Model", "MobileNetV2")
    col2.metric("Accuracy", "85%+")
    col3.metric("Classes", len(classes))
    col4.metric("Dataset Size", df["Count"].sum())

    st.markdown("### 💡 Key Features")
    st.write("""
    - Image Classification using CNN  
    - Real-time Predictions  
    - Advanced Analytics Dashboard  
    - Downloadable Reports  
    """)

# ==================================================
# 📊 ANALYTICS WITH FILTERS
# ==================================================
with tab2:
    st.subheader("📊 Analytics Dashboard")

    # -------- FILTER SECTION --------
    st.markdown("### 🎯 Filters")

    col1, col2 = st.columns(2)

    with col1:
        selected_categories = st.multiselect(
            "Select Categories",
            options=df["Category"].unique(),
            default=df["Category"].unique()
        )

    with col2:
        min_count = st.slider(
            "Minimum Image Count",
            int(df["Count"].min()),
            int(df["Count"].max()),
            int(df["Count"].min())
        )

    # -------- APPLY FILTER --------
    filtered_df = df[
        (df["Category"].isin(selected_categories)) &
        (df["Count"] >= min_count)
    ]

    st.markdown("### 📈 Filtered Insights")

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.pie(filtered_df, values='Count', names='Category', title='Distribution')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.bar(filtered_df, x='Category', y='Count', title='Category Count')
        st.plotly_chart(fig2, use_container_width=True)

    # -------- ADVANCED GRAPHS --------
    st.markdown("### 📊 Advanced Visualizations")

    fig3 = px.line(filtered_df, x='Category', y='Count', title='Trend')
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.scatter(filtered_df, x='Category', y='Count', size='Count', title='Scatter')
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.area(filtered_df, x='Category', y='Count', title='Area Chart')
    st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.funnel(filtered_df, x='Count', y='Category', title='Funnel')
    st.plotly_chart(fig6, use_container_width=True)

# ==================================================
# 📷 PREDICTION
# ==================================================
with tab3:
    st.subheader("📷 Upload Image for Prediction")

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
        st.markdown("### 🔝 Top Predictions")
        top3 = sorted(zip(classes, pred), key=lambda x: x[1], reverse=True)[:3]
        df_top3 = pd.DataFrame(top3, columns=["Class", "Probability"])

        fig = px.bar(df_top3, x="Class", y="Probability", title="Top 3 Predictions")
        st.plotly_chart(fig, use_container_width=True)

    # -------- HISTORY --------
    st.markdown("### 📜 Prediction History")

    if st.session_state["history"]:
        hist_df = pd.DataFrame(st.session_state["history"])
        st.dataframe(hist_df)

        csv = hist_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download History", csv, "history.csv")

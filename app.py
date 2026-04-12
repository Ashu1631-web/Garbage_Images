import streamlit as st
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import base64
import os
from datetime import datetime
import plotly.express as px

st.set_page_config(page_title="♻️ Garbage Waste Management", layout="wide")

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

# ---------------- LOGIN BG ----------------
def set_login_bg():
    if os.path.exists("garbage_bg.jpg"):
        with open("garbage_bg.jpg","rb") as f:
            img = base64.b64encode(f.read()).decode()

        st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img}");
            background-size: cover;
        }}
        .stApp::before {{
            content:"";
            position:fixed;
            width:100%;
            height:100%;
            background:rgba(0,0,0,0.6);
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
    st.session_state.login=False

def login():
    set_login_bg()
    st.title("♻️ Garbage Waste Management Login")
    u=st.text_input("Username")
    p=st.text_input("Password", type="password")

    if st.button("Login"):
        if u=="admin" and p=="1234":
            st.session_state.login=True
        else:
            st.error("Invalid login")

if not st.session_state.login:
    login()
    st.stop()

# ---------------- MAIN ----------------
set_ui()

# ---------------- PREDICT ----------------
def predict(image):
    if model is None:
        return "Error",0

    img=image.resize((160,160))
    img=np.array(img)/255.0
    img=np.expand_dims(img,0)

    pred=model.predict(img)
    idx=np.argmax(pred)
    conf=float(np.max(pred))
    return labels[idx],conf

def draw(image,label,conf):
    img=np.array(image)
    h,w,_=img.shape
    cv2.rectangle(img,(20,20),(w-20,h-20),(0,255,0),2)
    cv2.putText(img,f"{label} {round(conf*100,2)}%",
                (30,40),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
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
mode=st.sidebar.radio("Input",["Upload","Camera"])
page=st.sidebar.selectbox("Navigate",
["Overview","Detection","Analytics","History"])

# ---------------- OVERVIEW ----------------
if page=="Overview":
    st.title("🌍 Project Overview")

    st.markdown("""
    <div class="card">
    <h3>♻️ Garbage Waste Management </h3>
    <p>This AI model classifies waste into 6 categories using deep learning.</p>
    </div>
    """,unsafe_allow_html=True)

    c1,c2,c3=st.columns(3)
    c1.metric("Accuracy","~75%")
    c2.metric("Classes","6")
    c3.metric("Model","MobileNetV2")

# ---------------- DETECTION ----------------
elif page=="Detection":

    img=None

    if mode=="Upload":
        f=st.file_uploader("Upload Image")
        if f:
            img=Image.open(f)
    else:
        c=st.camera_input("Capture")
        if c:
            img=Image.open(c)

    if img:
        st.image(img,width=300)

        if st.button("Detect"):
            label,conf=predict(img)
            st.image(draw(img,label,conf))
            st.success(label)
            st.progress(int(conf*100))
            save(label,conf)

# ---------------- ANALYTICS ----------------
elif page=="Analytics":

    df=load()

    if df.empty:
        st.warning("No data")
    else:
        st.plotly_chart(px.pie(df,names="Label",title="Waste Distribution"))
        st.plotly_chart(px.bar(df,x="Label",title="Count by Label"))
        st.plotly_chart(px.histogram(df,x="Confidence",title="Confidence Distribution"))
        st.plotly_chart(px.line(df,x="Time",y="Confidence",title="Confidence Over Time"))
        st.plotly_chart(px.box(df,x="Label",y="Confidence",title="Confidence Spread"))
        st.plotly_chart(px.violin(df,x="Label",y="Confidence",title="Density"))
        st.plotly_chart(px.scatter(df,x="Confidence",y="Label",title="Scatter"))
        st.plotly_chart(px.area(df,x="Time",y="Confidence",title="Trend"))
        st.plotly_chart(px.strip(df,x="Label",y="Confidence",title="Strip Plot"))
        st.plotly_chart(px.density_heatmap(df,x="Confidence",y="Label",title="Heatmap"))

# ---------------- HISTORY ----------------
elif page=="History":

    df=load()

    if df.empty:
        st.warning("No history")
    else:
        st.dataframe(df)
        st.download_button("Download CSV",df.to_csv(index=False),"history.csv")

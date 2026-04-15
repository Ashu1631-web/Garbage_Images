import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ====================== PAGE CONFIG ====================== #
st.set_page_config(
    page_title="Waste Garbage Classification",
    page_icon="♻️",
    layout="wide"
)

# ====================== CUSTOM CSS ====================== #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@700&family=Rajdhani:wght@400;600&display=swap');

#MainMenu {visibility: hidden;}
footer    {visibility: hidden;}
header    {visibility: hidden;}

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0a0f0a;
    color: #e0ffe0;
    font-family: 'Rajdhani', sans-serif;
}

.page-heading {
    font-family: 'Orbitron', monospace;
    font-size: 1.8rem;
    color: #00ff88;
    letter-spacing: 3px;
    text-shadow: 0 0 18px #00ff8877;
    margin-bottom: 4px;
}
.page-sub {
    color: #78bb88;
    font-size: 1rem;
    margin-bottom: 24px;
}
.ov-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px;
    margin: 20px 0;
}
.ov-card {
    background: linear-gradient(135deg, rgba(0,60,30,0.7), rgba(0,30,15,0.9));
    border: 1px solid #00ff8840;
    border-radius: 14px;
    padding: 20px 16px;
    text-align: center;
    transition: transform 0.2s, border-color 0.2s;
}
.ov-card:hover { transform: translateY(-4px); border-color: #00ff88aa; }
.ov-card .icon  { font-size: 2.2rem; }
.ov-card .label { font-size: 0.9rem; color: #aaddbb; margin-top: 6px; }
.ov-card .val   { font-family: 'Orbitron', monospace; font-size: 1.2rem; color: #00ff88; margin-top: 4px; }

.graph-section {
    font-family: 'Orbitron', monospace;
    color: #00ff88;
    font-size: 1.05rem;
    letter-spacing: 1px;
    margin: 20px 0 10px;
    padding-left: 8px;
    border-left: 3px solid #00ff88;
}
.pred-badge {
    background: linear-gradient(90deg, #00ff88, #00cc66);
    color: #002210;
    font-family: 'Orbitron', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    padding: 14px 24px;
    border-radius: 12px;
    text-align: center;
    letter-spacing: 2px;
    box-shadow: 0 0 28px #00ff8855;
    margin-bottom: 10px;
}
.conf-val {
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem;
    color: #00ff88;
    text-align: center;
    text-shadow: 0 0 18px #00ff8877;
}
hr { border-color: #1a3a1a; }
</style>
""", unsafe_allow_html=True)

# ====================== SESSION STATE ====================== #
if "logged_in"   not in st.session_state: st.session_state.logged_in   = False
if "active_page" not in st.session_state: st.session_state.active_page = "Project Overview"

# ====================== AUTH ====================== #
USERS = {"admin": "waste123", "user1": "green2024", "demo": "demo"}

def login_page():
    # Full-page background image with dark overlay
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
        background:
            linear-gradient(rgba(0,0,0,0.48), rgba(0,0,0,0.55)),
            url('https://images.unsplash.com/photo-1532996122724-e3c354a0b15b?w=1600&q=80')
            center/cover no-repeat fixed !important;
    }
    [data-testid="stMain"]          { background: transparent !important; }
    [data-testid="stSidebar"]       { display: none !important; }
    section[data-testid="stMainBlockContainer"] { padding-top: 48px !important; }
    </style>
    """, unsafe_allow_html=True)

    # Small logo card — centered
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("""
        <div style="
            background: rgba(5,20,10,0.82);
            border: 1px solid #00ff88;
            border-radius: 14px;
            padding: 22px 32px 16px;
            text-align: center;
            box-shadow: 0 0 40px rgba(0,255,136,0.18);
            backdrop-filter: blur(14px);
            margin-bottom: 22px;
        ">
            <div style="font-family:Orbitron,monospace; font-size:1.4rem; color:#00ff88;
                        letter-spacing:3px; text-shadow:0 0 18px #00ff88aa;">
                ♻️ WASTE AI
            </div>
            <div style="color:#88cc99; font-size:0.83rem; margin-top:6px; line-height:1.55;">
                Waste Garbage Classification System<br>
                Powered by MobileNetV2 + Transfer Learning
            </div>
        </div>
        """, unsafe_allow_html=True)

        uname  = st.text_input("👤 Username", placeholder="Enter username")
        passwd = st.text_input("🔒 Password", type="password", placeholder="Enter password")
        login  = st.button("🚀  LOGIN", use_container_width=True)

        if login:
            if uname in USERS and USERS[uname] == passwd:
                st.session_state.logged_in   = True
                st.session_state.active_page = "Project Overview"
                st.rerun()
            else:
                st.error("❌ Invalid username or password!")

# ====================== MODEL ====================== #
@st.cache_resource
def load_my_model():
    try:
        class PatchedDense(tf.keras.layers.Dense):
            def __init__(self, *args, **kwargs):
                kwargs.pop('quantization_config', None)
                super().__init__(*args, **kwargs)
        with tf.keras.utils.custom_object_scope({'Dense': PatchedDense}):
            model = tf.keras.models.load_model("clean_model.h5", compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None

def load_class_names():
    try:
        with open("class_names.json", "r") as f:
            data = json.load(f)
            if isinstance(data, dict):
                data = [k for k, v in sorted(data.items(), key=lambda x: x[1])]
            return data
    except Exception as e:
        st.error(f"❌ class_names.json load failed: {e}")
        return []

# ====================== PREPROCESS & PREDICT ====================== #
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.resize((160, 160)).convert("RGB")
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(img: Image.Image, model, class_names):
    try:
        arr   = preprocess_image(img)
        preds = model.predict(arr, verbose=0)[0]
        if len(preds) != len(class_names):
            st.error(f"⚠️ Class mismatch → Model: {len(preds)} vs Labels: {len(class_names)}")
            return "Mismatch Error", 0.0, [], preds
        top3_idx = preds.argsort()[-3:][::-1]
        results  = [(class_names[int(i)], float(preds[i])) for i in top3_idx]
        return results[0][0], results[0][1], results, preds
    except Exception as e:
        st.error(f"⚠️ Prediction Error: {str(e)}")
        return "Error", 0.0, [], np.zeros(len(class_names))

# ====================== RGB helper ====================== #
def _rgb_channel(img_array, ch: int):
    counts, bins = np.histogram(img_array[:, :, ch].flatten(), bins=64, range=(0, 255))
    return counts, bins

# ====================== 10 GRAPHS ====================== #
def show_all_graphs(image: Image.Image, label, confidence, top3, all_preds, class_names):

    st.markdown(f'<div class="pred-badge">✅ PREDICTED: {label.upper()}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="conf-val">{round(confidence*100,2)}%</div>', unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;color:#88aa99;margin-bottom:14px;'>Confidence Score</div>",
                unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔥 Top 3 Predictions")
    for rank, (lbl, prob) in enumerate(top3, 1):
        st.write(f"**#{rank}** {lbl} → `{round(prob*100,2)}%`")
        st.progress(float(prob))

    st.markdown("---")
    st.markdown('<div class="graph-section">📊 PREDICTION GRAPHS — 10 VISUALIZATIONS</div>',
                unsafe_allow_html=True)

    # Graph 1 & 2
    c1, c2 = st.columns(2)
    with c1:
        labels_top3 = [x[0] for x in top3]
        vals_top3   = [round(x[1]*100, 2) for x in top3]
        fig1 = go.Figure(go.Bar(
            x=vals_top3, y=labels_top3, orientation="h",
            marker=dict(color=vals_top3, colorscale="Teal", showscale=False),
            text=[f"{v}%" for v in vals_top3], textposition="auto"
        ))
        fig1.update_layout(title="📊 Graph 1 — Top 3 Confidence (Horizontal Bar)",
                           xaxis_title="Confidence %", template="plotly_dark", height=300)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        pie_vals = [round(float(p)*100, 2) for p in all_preds]
        fig2 = go.Figure(go.Pie(
            labels=class_names, values=pie_vals, hole=0.35,
            textinfo="label+percent",
            marker=dict(colors=px.colors.qualitative.Safe)
        ))
        fig2.update_layout(title="🥧 Graph 2 — All Classes Probability (Pie Chart)",
                           template="plotly_dark", height=300)
        st.plotly_chart(fig2, use_container_width=True)

    # Graph 3 & 4
    c3, c4 = st.columns(2)
    with c3:
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(confidence*100, 2),
            title={"text": f"🎯 Graph 3 — Confidence Gauge<br>{label.upper()}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#00cc96"},
                "steps": [
                    {"range": [0,  40], "color": "#ef553b"},
                    {"range": [40, 70], "color": "#ffa15a"},
                    {"range": [70,100], "color": "#00cc96"},
                ],
                "threshold": {"line": {"color": "white","width": 3},"thickness": 0.75,"value": 70}
            }
        ))
        fig3.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        top6_idx     = all_preds.argsort()[-6:][::-1]
        radar_labels = [class_names[i] for i in top6_idx]
        radar_vals   = [round(float(all_preds[i])*100, 2) for i in top6_idx]
        fig4 = go.Figure(go.Scatterpolar(
            r=radar_vals + [radar_vals[0]],
            theta=radar_labels + [radar_labels[0]],
            fill="toself", line=dict(color="#636efa", width=2)
        ))
        fig4.update_layout(title="🕸️ Graph 4 — Top-6 Radar Chart",
                           polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                           template="plotly_dark", height=300)
        st.plotly_chart(fig4, use_container_width=True)

    # Graph 5 & 6
    c5, c6 = st.columns(2)
    img_array = np.array(image.resize((160, 160)).convert("RGB"))
    with c5:
        fig5 = go.Figure()
        for ch_idx, (color, channel) in enumerate(zip(["red","green","blue"],["Red","Green","Blue"])):
            counts, bins = _rgb_channel(img_array, ch_idx)
            fig5.add_trace(go.Scatter(
                x=bins[:-1], y=counts, mode="lines",
                fill="tozeroy", name=channel,
                line=dict(color=color, width=1.5), opacity=0.6
            ))
        fig5.update_layout(title="📊 Graph 5 — RGB Channel Distribution",
                           xaxis_title="Pixel Value (0–255)", yaxis_title="Frequency",
                           template="plotly_dark", height=300, legend=dict(orientation="h"))
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        img_gray = np.array(image.resize((80, 80)).convert("L"))
        fig6 = go.Figure(data=go.Heatmap(z=img_gray, colorscale="Viridis", showscale=True))
        fig6.update_layout(title="🌡️ Graph 6 — Pixel Intensity Heatmap",
                           template="plotly_dark", height=300,
                           yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig6, use_container_width=True)

    # Graph 7 — Bar chart per class (replaces Violin to avoid fillcolor ValueError)
    sorted_bar_idx  = all_preds.argsort()[::-1]
    bar7_labels     = [class_names[i] for i in sorted_bar_idx]
    bar7_vals       = [round(float(all_preds[i])*100, 4) for i in sorted_bar_idx]
    bar7_colors     = ["#00ff88" if class_names[i] == label else "#2196f3" for i in sorted_bar_idx]
    fig7 = go.Figure(go.Bar(
        x=bar7_labels, y=bar7_vals,
        marker_color=bar7_colors,
        text=[f"{v:.2f}%" for v in bar7_vals],
        textposition="outside"
    ))
    fig7.update_layout(
        title="📊 Graph 7 — All Classes Probability Distribution (Green = Predicted)",
        xaxis_title="Waste Class", yaxis_title="Confidence %",
        template="plotly_dark", height=340
    )
    st.plotly_chart(fig7, use_container_width=True)

    # Graph 8 & 9
    c8, c9 = st.columns(2)
    with c8:
        sorted_idx = all_preds.argsort()[::-1]
        all_labels = [class_names[i] for i in sorted_idx]
        all_vals   = [round(float(all_preds[i])*100, 2) for i in sorted_idx]
        bar_colors = ["#00ff88" if l == label else "#636efa" for l in all_labels]
        fig8 = go.Figure(go.Bar(
            x=all_labels, y=all_vals,
            marker_color=bar_colors,
            text=[f"{v}%" for v in all_vals], textposition="outside"
        ))
        fig8.update_layout(title="📊 Graph 8 — All Classes Ranked (Green = Predicted)",
                           xaxis_title="Waste Class", yaxis_title="Confidence %",
                           template="plotly_dark", height=340)
        st.plotly_chart(fig8, use_container_width=True)

    with c9:
        tree_vals = [max(float(p)*100, 0.01) for p in all_preds]
        fig9 = go.Figure(go.Treemap(
            labels=class_names,
            parents=[""] * len(class_names),
            values=tree_vals,
            textinfo="label+value",
            marker=dict(colorscale="RdYlGn", colors=tree_vals, showscale=True)
        ))
        fig9.update_layout(title="🗂️ Graph 9 — Probability Treemap (Bigger = Higher Confidence)",
                           template="plotly_dark", height=340)
        st.plotly_chart(fig9, use_container_width=True)

    # Graph 10 — Waterfall
    sorted_idx2 = all_preds.argsort()[::-1]
    wf_labels   = [class_names[i] for i in sorted_idx2]
    wf_vals     = [round(float(all_preds[i])*100, 2) for i in sorted_idx2]
    fig10 = go.Figure(go.Waterfall(
        name="Confidence",
        measure=["relative"] * len(wf_vals),
        x=wf_labels, y=wf_vals,
        connector={"line": {"color": "#333"}},
        increasing={"marker": {"color": "#00cc96"}},
        decreasing={"marker": {"color": "#ef553b"}},
        text=[f"{v}%" for v in wf_vals], textposition="outside"
    ))
    fig10.update_layout(title="💧 Graph 10 — Confidence Waterfall Chart",
                        yaxis_title="Confidence %", template="plotly_dark", height=350)
    st.plotly_chart(fig10, use_container_width=True)


# ====================== PROJECT OVERVIEW ====================== #
def page_overview(class_names):
    st.markdown('<div class="page-heading">♻️ PROJECT OVERVIEW</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Waste Garbage AI Classification System — Complete Documentation</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="ov-grid">
        <div class="ov-card"><div class="icon">🤖</div><div class="label">AI Model</div><div class="val">MobileNetV2</div></div>
        <div class="ov-card"><div class="icon">📦</div><div class="label">Total Classes</div><div class="val">{cls}</div></div>
        <div class="ov-card"><div class="icon">📊</div><div class="label">Graphs per Scan</div><div class="val">10</div></div>
        <div class="ov-card"><div class="icon">🎯</div><div class="label">Input Size</div><div class="val">160×160</div></div>
        <div class="ov-card"><div class="icon">⚡</div><div class="label">Framework</div><div class="val">TensorFlow</div></div>
        <div class="ov-card"><div class="icon">🌐</div><div class="label">Interface</div><div class="val">Streamlit</div></div>
    </div>
    """.format(cls=len(class_names)), unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📌 About This Project")
    st.markdown("""
    This AI-powered system uses **Transfer Learning** with **MobileNetV2** to detect and classify
    waste materials from uploaded images or live webcam feed — enabling smart waste segregation
    for better recycling and environmental management.
    """)

    st.markdown("### 🗑️ Supported Waste Categories")
    icons = {"battery":"🔋","biological":"🦠","brown-glass":"🍶","cardboard":"📦",
             "clothes":"👕","green-glass":"🟢","metal":"🔩","paper":"📄",
             "plastic":"🧴","shoes":"👟","trash":"🗑️","white-glass":"⬜"}
    cols = st.columns(4)
    for i, name in enumerate(class_names):
        with cols[i % 4]:
            ico = icons.get(name, "♻️")
            st.markdown(f"""
            <div class="ov-card" style="margin-bottom:12px;">
                <div class="icon">{ico}</div>
                <div class="label">{name.upper()}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📈 10 Visualization Graphs")
    graphs = [
        ("Graph 1",  "Top 3 Confidence — Horizontal Bar",      "📊"),
        ("Graph 2",  "All Classes Probability — Pie/Donut",     "🥧"),
        ("Graph 3",  "Confidence Gauge — Speedometer",          "🎯"),
        ("Graph 4",  "Top-6 Classes — Radar/Spider Chart",      "🕸️"),
        ("Graph 5",  "Image RGB Channel Distribution",          "🌈"),
        ("Graph 6",  "Pixel Intensity Heatmap (80×80)",         "🌡️"),
        ("Graph 7",  "All Classes Probability Distribution",    "📊"),
        ("Graph 8",  "All Classes Ranked — Full Bar Chart",     "📉"),
        ("Graph 9",  "Probability Treemap (Size = Confidence)", "🗂️"),
        ("Graph 10", "Confidence Waterfall Chart",              "💧"),
    ]
    gcols = st.columns(2)
    for i, (g, desc, ico) in enumerate(graphs):
        with gcols[i % 2]:
            st.markdown(f"""
            <div class="ov-card" style="text-align:left; margin-bottom:12px; padding:14px 18px;">
                <span style="font-family:Orbitron,monospace;color:#00ff88;font-size:0.85rem;">{ico} {g}</span>
                <div class="label" style="margin-top:6px; color:#aaccbb;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
| Component | Technology |
|-----------|------------|
| **Deep Learning** | TensorFlow 2.x + Keras |
| **Base Model** | MobileNetV2 (Transfer Learning) |
| **Frontend** | Streamlit |
| **Visualizations** | Plotly (10 chart types) |
| **Image Processing** | Pillow (PIL) + NumPy |
| **Data** | Pandas |
    """)


# ====================== DETECTION PAGE ====================== #
def detection_page(mode: str, model, class_names):
    if mode == "upload":
        st.markdown('<div class="page-heading">📤 UPLOAD IMAGE</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Upload a waste image to classify and visualize with 10 graphs</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("Upload waste image", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption="Uploaded Image", use_column_width=True)
                detect_btn = st.button("🔍 Detect Waste", use_container_width=True)
            else:
                detect_btn = False

        with col2:
            if uploaded_file and detect_btn:
                with st.spinner("🔬 Analyzing image..."):
                    lbl, conf, top3, all_preds = predict(image, model, class_names)
                show_all_graphs(image, lbl, conf, top3, all_preds, class_names)
            elif not uploaded_file:
                st.info("👈 Upload an image to begin waste detection.")

    else:  # webcam
        st.markdown('<div class="page-heading">📷 LIVE WEBCAM</div>', unsafe_allow_html=True)
        st.markdown('<div class="page-sub">Use your camera to detect waste type in real-time with 10 graphs</div>',
                    unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            cam_image = st.camera_input("Point camera at waste item")
            if cam_image:
                image = Image.open(cam_image).convert("RGB")
                st.image(image, caption="Captured Image", use_column_width=True)

        with col2:
            if cam_image:
                with st.spinner("🔬 Analyzing..."):
                    lbl, conf, top3, all_preds = predict(image, model, class_names)
                show_all_graphs(image, lbl, conf, top3, all_preds, class_names)


# ====================== MAIN APP ====================== #
def main_app():
    model       = load_my_model()
    class_names = load_class_names()

    with st.sidebar:
        st.markdown("""
        <div style='font-family:Orbitron,monospace; color:#00ff88; font-size:1.05rem;
                    letter-spacing:2px; padding:10px 4px 2px; text-shadow:0 0 12px #00ff8877;'>
            ♻️ WASTE AI
        </div>
        <div style='color:#557766; font-size:0.78rem; padding-bottom:14px;'>
            Garbage Classification System
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**🗂️ NAVIGATION**")

        pages = {
            "🏠 Project Overview": "Project Overview",
            "📤 Upload Image":     "Upload Image",
            "📷 Live Webcam":      "Live Webcam",
        }
        for label, key in pages.items():
            active = st.session_state.active_page == key
            if st.button(label, key=f"nav_{key}", use_container_width=True,
                         type="primary" if active else "secondary"):
                st.session_state.active_page = key
                st.rerun()

        st.markdown("---")
        st.markdown("**🗑️ Waste Types**")
        for name in class_names:
            st.write(f"• {name}")

        st.markdown("---")
        if model:
            st.success("✅ Model Ready")
        else:
            st.error("❌ Model Error")

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in   = False
            st.session_state.active_page = "Project Overview"
            st.rerun()

    page = st.session_state.active_page
    if page == "Project Overview":
        page_overview(class_names)
    elif page == "Upload Image":
        detection_page("upload", model, class_names)
    elif page == "Live Webcam":
        detection_page("webcam", model, class_names)


# ====================== ENTRY POINT ====================== #
if not st.session_state.logged_in:
    login_page()
else:
    main_app()

st.markdown("""
<div style='text-align:center; color:#335544; font-size:0.78rem; padding-top:20px;
            font-family:Rajdhani,sans-serif;'>
    🤖 Powered by TensorFlow &amp; Streamlit &nbsp;|&nbsp; Waste AI Classification System
</div>
""", unsafe_allow_html=True)

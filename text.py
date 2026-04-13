import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json

# ====================== PAGE CONFIG ====================== #
st.set_page_config(
    page_title="Waste AI | Smart Classification",
    page_icon="♻️",
    layout="wide"
)

# ====================== PREMIUM CSS ====================== #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

/* ── Root Variables ── */
:root {
    --neon-green: #00ff88;
    --neon-blue: #00d4ff;
    --neon-purple: #b44fff;
    --bg-dark: #060b14;
    --bg-card: #0d1b2e;
    --bg-card2: #091623;
    --border-glow: rgba(0, 255, 136, 0.3);
    --text-primary: #e8f4f8;
    --text-muted: #5a7a8a;
}

/* ── Global Background ── */
.stApp {
    background: var(--bg-dark) !important;
    background-image:
        radial-gradient(ellipse at 20% 50%, rgba(0,212,255,0.04) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 20%, rgba(0,255,136,0.05) 0%, transparent 60%),
        radial-gradient(ellipse at 60% 80%, rgba(180,79,255,0.04) 0%, transparent 50%) !important;
    font-family: 'Rajdhani', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-dark); }
::-webkit-scrollbar-thumb { background: var(--neon-green); border-radius: 10px; }

/* ── Hero Title ── */
.hero-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 2.8rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--neon-green) 0%, var(--neon-blue) 50%, var(--neon-purple) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    letter-spacing: 0.08em;
    margin-bottom: 0.2rem;
    animation: titleGlow 3s ease-in-out infinite alternate;
}

@keyframes titleGlow {
    from { filter: drop-shadow(0 0 8px rgba(0,255,136,0.4)); }
    to   { filter: drop-shadow(0 0 20px rgba(0,212,255,0.6)); }
}

.hero-subtitle {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    color: var(--text-muted);
    text-align: center;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    margin-bottom: 2rem;
}

/* ── Info Cards ── */
.info-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}

.info-card {
    background: var(--bg-card);
    border: 1px solid rgba(0,255,136,0.2);
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}

.info-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--neon-green), transparent);
}

.info-card:hover {
    border-color: var(--neon-green);
    box-shadow: 0 0 20px rgba(0,255,136,0.15);
    transform: translateY(-2px);
}

.info-card .icon { font-size: 1.8rem; margin-bottom: 0.4rem; }
.info-card .label {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.15em;
}
.info-card .value {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.85rem;
    color: var(--neon-green);
    font-weight: 700;
    margin-top: 0.2rem;
}

/* ── Section Headers ── */
.section-header {
    font-family: 'Orbitron', sans-serif;
    font-size: 1rem;
    color: var(--neon-blue);
    text-transform: uppercase;
    letter-spacing: 0.2em;
    border-left: 3px solid var(--neon-blue);
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem 0;
}

/* ── Upload Zone ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card2) !important;
    border: 2px dashed rgba(0,255,136,0.25) !important;
    border-radius: 12px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--neon-green) !important;
    box-shadow: 0 0 20px rgba(0,255,136,0.1) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(0,255,136,0.15), rgba(0,212,255,0.15)) !important;
    border: 1px solid var(--neon-green) !important;
    color: var(--neon-green) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, rgba(0,255,136,0.3), rgba(0,212,255,0.3)) !important;
    box-shadow: 0 0 20px rgba(0,255,136,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── Result Box ── */
.result-box {
    background: linear-gradient(135deg, rgba(0,255,136,0.08), rgba(0,212,255,0.05));
    border: 1px solid rgba(0,255,136,0.4);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: resultReveal 0.5s ease-out;
}

@keyframes resultReveal {
    from { opacity: 0; transform: scale(0.95); }
    to   { opacity: 1; transform: scale(1); }
}

.result-box::after {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: conic-gradient(transparent 0deg, rgba(0,255,136,0.05) 60deg, transparent 120deg);
    animation: rotateGlow 8s linear infinite;
}

@keyframes rotateGlow {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}

.result-label {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.3em;
    margin-bottom: 0.3rem;
}

.result-value {
    font-family: 'Orbitron', sans-serif;
    font-size: 2rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--neon-green), var(--neon-blue));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    position: relative;
    z-index: 1;
}

.confidence-ring {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1.1rem;
    color: var(--neon-blue);
    margin-top: 0.5rem;
    letter-spacing: 0.1em;
    position: relative;
    z-index: 1;
}

/* ── Prediction Rows ── */
.pred-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: var(--bg-card2);
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 10px;
    padding: 0.8rem 1.2rem;
    margin-bottom: 0.6rem;
    transition: all 0.2s ease;
}

.pred-row:hover {
    border-color: rgba(0,212,255,0.4);
    background: rgba(0,212,255,0.05);
}

.pred-rank {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.7rem;
    color: var(--neon-purple);
    min-width: 28px;
}

.pred-name {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
    flex: 1;
    text-transform: capitalize;
}

.pred-pct {
    font-family: 'Orbitron', sans-serif;
    font-size: 0.85rem;
    color: var(--neon-green);
    font-weight: 700;
    min-width: 60px;
    text-align: right;
}

/* ── Progress Bars ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--neon-green), var(--neon-blue)) !important;
    border-radius: 10px !important;
}
.stProgress > div > div > div {
    background: rgba(255,255,255,0.05) !important;
    border-radius: 10px !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card2) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
    border: 1px solid rgba(0,255,136,0.1) !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.1em !important;
    color: var(--text-muted) !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.2rem !important;
    background: transparent !important;
    border: none !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(0,255,136,0.2), rgba(0,212,255,0.15)) !important;
    color: var(--neon-green) !important;
    border: 1px solid rgba(0,255,136,0.3) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-card) !important;
    border-right: 1px solid rgba(0,255,136,0.1) !important;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--neon-green) !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.1em !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid rgba(0,255,136,0.2) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'Rajdhani', sans-serif !important;
    color: var(--text-muted) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Orbitron', sans-serif !important;
    color: var(--neon-green) !important;
}

/* ── Alerts ── */
.stSuccess {
    background: rgba(0,255,136,0.1) !important;
    border: 1px solid rgba(0,255,136,0.4) !important;
    color: var(--neon-green) !important;
    border-radius: 10px !important;
    font-family: 'Rajdhani', sans-serif !important;
}

.stInfo {
    background: rgba(0,212,255,0.08) !important;
    border: 1px solid rgba(0,212,255,0.3) !important;
    color: var(--neon-blue) !important;
    border-radius: 10px !important;
}

.stError {
    background: rgba(255,50,100,0.08) !important;
    border: 1px solid rgba(255,50,100,0.3) !important;
    border-radius: 10px !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(0,255,136,0.3), rgba(0,212,255,0.3), transparent) !important;
    margin: 1.5rem 0 !important;
}

/* ── Camera Input ── */
[data-testid="stCameraInput"] {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(0,255,136,0.2) !important;
}

/* ── Images ── */
[data-testid="stImage"] img {
    border-radius: 12px !important;
    border: 1px solid rgba(0,255,136,0.2) !important;
    box-shadow: 0 0 20px rgba(0,0,0,0.5) !important;
}

/* ── Footer ── */
.footer-text {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.75rem;
    color: var(--text-muted);
    text-align: center;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding: 1rem 0;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: var(--neon-green) !important;
}

/* ── Caption ── */
.stCaption {
    font-family: 'Rajdhani', sans-serif !important;
    color: var(--text-muted) !important;
    text-align: center !important;
}
</style>
""", unsafe_allow_html=True)

# ====================== MODEL LOAD ====================== #
@st.cache_resource
def load_my_model():
    try:
        class PatchedDense(tf.keras.layers.Dense):
            def __init__(self, *args, **kwargs):
                kwargs.pop('quantization_config', None)
                super().__init__(*args, **kwargs)

        with tf.keras.utils.custom_object_scope({'Dense': PatchedDense}):
            model = tf.keras.models.load_model(
                "clean_model.h5",
                compile=False
            )
        return model
    except Exception as e:
        st.error(f"❌ Model load failed: {e}")
        return None

# ====================== CLASS NAMES ====================== #
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

model = load_my_model()
class_names = load_class_names()

# ====================== SIDEBAR ====================== #
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
        <div style='font-family: Orbitron, sans-serif; font-size: 1.4rem; font-weight:900;
                    background: linear-gradient(135deg, #00ff88, #00d4ff);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    background-clip: text; letter-spacing: 0.1em;'>WASTE AI</div>
        <div style='font-family: Rajdhani, sans-serif; font-size: 0.65rem; color: #5a7a8a;
                    letter-spacing: 0.3em; margin-top: 0.2rem;'>SYSTEM v2.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    if model:
        st.success("⚡ Model Ready")
    else:
        st.error("❌ Model Error")

    st.info(f"🗂️ Total Classes: **{len(class_names)}**")
    st.markdown("---")

    st.markdown("""
    <div style='font-family: Orbitron, sans-serif; font-size: 0.7rem; color: #00d4ff;
                letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 0.8rem;'>
        Waste Categories
    </div>
    """, unsafe_allow_html=True)

    icons = {"plastic":"🧴","metal":"🔩","paper":"📄","cardboard":"📦",
             "glass":"🍶","battery":"🔋","clothes":"👕","trash":"🗑️"}

    for name in class_names:
        icon = icons.get(name.lower(), "♻️")
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:0.6rem; padding:0.4rem 0.6rem;
                    border-radius:8px; margin-bottom:0.3rem;
                    background: rgba(0,255,136,0.04); border: 1px solid rgba(0,255,136,0.08);'>
            <span style='font-size:1rem;'>{icon}</span>
            <span style='font-family: Rajdhani, sans-serif; font-size:0.85rem;
                         color:#c8dde8; text-transform:capitalize;'>{name}</span>
        </div>
        """, unsafe_allow_html=True)

# ====================== HERO HEADER ====================== #
st.markdown("""
<div style='padding: 2rem 0 1rem 0;'>
    <div class='hero-title'>♻ WASTE INTELLIGENCE</div>
    <div class='hero-subtitle'>AI-Powered Garbage Classification System</div>
</div>
""", unsafe_allow_html=True)

# ── Info Cards ──
st.markdown("""
<div class='info-grid'>
    <div class='info-card'>
        <div class='icon'>🧠</div>
        <div class='label'>Model</div>
        <div class='value'>MobileNetV2</div>
    </div>
    <div class='info-card'>
        <div class='icon'>🎯</div>
        <div class='label'>Method</div>
        <div class='value'>Transfer Learning</div>
    </div>
    <div class='info-card'>
        <div class='icon'>📐</div>
        <div class='label'>Input Size</div>
        <div class='value'>160 × 160 px</div>
    </div>
    <div class='info-card'>
        <div class='icon'>🗂️</div>
        <div class='label'>Categories</div>
        <div class='value'>8 Waste Types</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.divider()

# ====================== PREPROCESS ====================== #
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.resize((160, 160)).convert("RGB")
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

# ====================== PREDICT ====================== #
def predict(img: Image.Image):
    try:
        arr = preprocess_image(img)
        preds = model.predict(arr, verbose=0)[0]

        if len(preds) != len(class_names):
            st.error(f"⚠️ Class mismatch → Model: {len(preds)} vs Labels: {len(class_names)}")
            return "Mismatch Error", 0.0, []

        top3_idx = preds.argsort()[-3:][::-1]
        results = [(class_names[int(i)], float(preds[i])) for i in top3_idx]
        return results[0][0], results[0][1], results

    except Exception as e:
        st.error(f"⚠️ Prediction Error: {str(e)}")
        return "Error", 0.0, []

# ====================== SHOW RESULTS ====================== #
def show_results(image):
    if model is None:
        st.error("❌ Model not loaded.")
        return
    if not class_names:
        st.error("❌ class_names.json not found.")
        return

    with st.spinner("🔬 Scanning waste signature..."):
        label, confidence, top3 = predict(image)

    # Primary result card
    st.markdown(f"""
    <div class='result-box'>
        <div class='result-label'>Classification Result</div>
        <div class='result-value'>{label.upper()}</div>
        <div class='confidence-ring'>⬡ Confidence: {round(confidence * 100, 2)}%</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Top 3 Predictions</div>", unsafe_allow_html=True)

    rank_colors = ["#00ff88", "#00d4ff", "#b44fff"]
    rank_labels = ["#01", "#02", "#03"]

    for rank, (lbl, prob) in enumerate(top3):
        pct = round(prob * 100, 2)
        color = rank_colors[rank]
        st.markdown(f"""
        <div class='pred-row'>
            <span class='pred-rank'>{rank_labels[rank]}</span>
            <span class='pred-name'>{lbl}</span>
            <span class='pred-pct' style='color:{color};'>{pct}%</span>
        </div>
        """, unsafe_allow_html=True)
        st.progress(float(prob))

# ====================== TABS ====================== #
tab1, tab2 = st.tabs(["  📤  UPLOAD IMAGE  ", "  📷  LIVE WEBCAM  "])

with tab1:
    st.markdown("<div class='section-header'>Upload Waste Image</div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        uploaded_file = st.file_uploader(
            "Drop waste image here",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="📸 Uploaded Image", width=280)
            detect_btn = st.button("⚡ DETECT WASTE", use_container_width=True)
        else:
            detect_btn = False
            st.markdown("""
            <div style='text-align:center; padding: 3rem 1rem; color: #3a5a6a;
                        font-family: Rajdhani, sans-serif; font-size: 0.9rem;
                        letter-spacing: 0.15em;'>
                ↑ DROP IMAGE ABOVE ↑<br>
                <span style='font-size: 0.75rem; opacity: 0.7;'>JPG / JPEG / PNG supported</span>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        if uploaded_file and detect_btn:
            show_results(image)
        elif not uploaded_file:
            st.markdown("""
            <div style='display:flex; flex-direction:column; align-items:center;
                        justify-content:center; height: 300px; color: #3a5a6a;
                        font-family: Rajdhani, sans-serif; text-align:center;'>
                <div style='font-size: 3rem; margin-bottom: 1rem; opacity: 0.4;'>♻️</div>
                <div style='letter-spacing: 0.2em; font-size: 0.8rem;'>
                    AWAITING IMAGE INPUT
                </div>
                <div style='font-size: 0.7rem; opacity: 0.6; margin-top: 0.5rem;'>
                    Upload an image on the left to begin analysis
                </div>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.markdown("<div class='section-header'>Live Webcam Detection</div>", unsafe_allow_html=True)
    cam_image = st.camera_input("Point camera at waste item", label_visibility="collapsed")

    if cam_image:
        col1, col2 = st.columns([1, 2], gap="large")
        with col1:
            image = Image.open(cam_image).convert("RGB")
            st.image(image, caption="📸 Captured Frame", width=280)
        with col2:
            show_results(image)

# ====================== FOOTER ====================== #
st.divider()
st.markdown("""
<div class='footer-text'>
    ⬡ &nbsp; Powered by TensorFlow &amp; Streamlit &nbsp; ⬡ &nbsp; MobileNetV2 Transfer Learning &nbsp; ⬡ &nbsp; Waste AI v2.0 &nbsp; ⬡
</div>
""", unsafe_allow_html=True)

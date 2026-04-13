import streamlit as st
import numpy as np
from keras.models import load_model
from PIL import Image
import json
import plotly.express as px
import os

# ====================== PAGE CONFIG ====================== #
st.set_page_config(
    page_title="WasteAI — Smart Waste Classifier",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ====================== CUSTOM CSS ====================== #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

/* ---- Root & Background ---- */
:root {
    --bg: #0a0f0a;
    --surface: #111a11;
    --surface2: #162016;
    --border: #2a3d2a;
    --accent: #4ade80;
    --accent2: #86efac;
    --accent-dim: rgba(74,222,128,0.12);
    --text: #e8f5e8;
    --muted: #6b8f6b;
    --danger: #f87171;
    --warning: #fbbf24;
}

html, body, [data-testid="stApp"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border); }

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ---- Hero Banner ---- */
.hero-banner {
    background: linear-gradient(135deg, var(--surface) 0%, #0d1f0d 50%, var(--surface2) 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(74,222,128,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-banner::after {
    content: '♻';
    position: absolute;
    bottom: -20px; right: 32px;
    font-size: 120px;
    opacity: 0.05;
    line-height: 1;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.8rem;
    letter-spacing: -1px;
    color: var(--text);
    margin: 0 0 8px 0;
    line-height: 1.1;
}
.hero-title span { color: var(--accent); }
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0 0 16px 0;
}
.hero-desc {
    color: var(--muted);
    font-size: 0.95rem;
    max-width: 480px;
    line-height: 1.6;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--accent-dim);
    border: 1px solid rgba(74,222,128,0.3);
    border-radius: 20px;
    padding: 4px 14px;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--accent);
    letter-spacing: 1px;
    margin-top: 20px;
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--accent);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50% { opacity:0.4; transform:scale(0.8); }
}

/* ---- Section Cards ---- */
.section-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 28px;
    height: 100%;
}
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ---- Input Mode Toggle ---- */
.mode-btn-container {
    display: flex;
    gap: 8px;
    margin-bottom: 20px;
}

/* ---- Result Card ---- */
.result-card {
    background: linear-gradient(135deg, var(--surface2), var(--surface));
    border: 1px solid var(--accent);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.result-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
}
.result-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: 6px;
}
.result-class {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.2rem;
    color: var(--accent);
    letter-spacing: -1px;
    text-transform: uppercase;
    line-height: 1;
    margin-bottom: 4px;
}
.result-conf {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: var(--accent2);
}

/* ---- Progress Bars ---- */
.pred-row {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.78rem;
}
.pred-name {
    width: 90px;
    color: var(--text);
    text-overflow: ellipsis;
    overflow: hidden;
    white-space: nowrap;
    flex-shrink: 0;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.pred-bar-bg {
    flex: 1;
    height: 6px;
    background: var(--surface2);
    border-radius: 3px;
    overflow: hidden;
}
.pred-bar-fill {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    transition: width 0.8s ease;
}
.pred-pct {
    width: 44px;
    text-align: right;
    color: var(--muted);
    flex-shrink: 0;
}

/* ---- Waste Type Badges ---- */
.badge-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 8px;
}
.badge {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 10px;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--muted);
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* ---- Streamlit widget overrides ---- */
[data-testid="stFileUploader"] {
    background: var(--surface2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 10px !important;
    padding: 8px !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent) !important;
}
[data-testid="stCameraInput"] {
    background: var(--surface2) !important;
    border: 2px dashed var(--border) !important;
    border-radius: 10px !important;
}
[data-testid="stCameraInput"] video {
    border-radius: 8px !important;
}

button[kind="primary"] {
    background: var(--accent) !important;
    color: #000 !important;
    border: none !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border-radius: 8px !important;
}
button[kind="secondary"] {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    font-family: 'Syne', sans-serif !important;
    border-radius: 8px !important;
}
[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid var(--border);
}
[data-testid="stRadio"] label {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    color: var(--text) !important;
}
[data-testid="stRadio"] > div {
    gap: 8px !important;
}
[data-baseweb="radio"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 10px 16px !important;
}
div[data-testid="stAlert"] {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}
</style>
""", unsafe_allow_html=True)

# ====================== SESSION STATE ====================== #
if "model" not in st.session_state:
    st.session_state.model = None
if "class_names" not in st.session_state:
    st.session_state.class_names = []

# ====================== LOAD MODEL ====================== #
MODEL_PATH  = "final_garbage_model.keras"
CLASS_PATH  = "class_names.json"

@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return load_model(MODEL_PATH, compile=False)
    except Exception:
        return None

@st.cache_data
def load_class_names():
    if not os.path.exists(CLASS_PATH):
        return []
    try:
        with open(CLASS_PATH) as f:
            return json.load(f)
    except Exception:
        return []

model       = load_my_model()
class_names = load_class_names()

# ====================== HERO BANNER ====================== #
status_text = "MODEL ONLINE" if model else "MODEL OFFLINE"
status_color = "#4ade80" if model else "#f87171"

st.markdown(f"""
<div class="hero-banner">
    <p class="hero-sub">AI-Powered Waste Intelligence</p>
    <h1 class="hero-title">Waste<span>AI</span></h1>
    <p class="hero-desc">
        Real-time waste classification using deep learning.
        Upload an image or use your camera to instantly identify and categorize waste materials.
    </p>
    <div class="status-pill">
        <span class="status-dot" style="background:{status_color}"></span>
        {status_text} &nbsp;·&nbsp; MobileNetV2 &nbsp;·&nbsp; {len(class_names)} CLASSES
    </div>
</div>
""", unsafe_allow_html=True)

# ====================== HELPER FUNCS ====================== #
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.resize((160, 160))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

def predict(img: Image.Image):
    try:
        arr  = preprocess_image(img)
        pred = model.predict(arr, verbose=0)[0]
        top3 = pred.argsort()[-3:][::-1]
        results = [(class_names[i], float(pred[i])) for i in top3]
        return results[0][0], results[0][1], results
    except Exception as e:
        return "Error", 0.0, []

def get_waste_emoji(label: str) -> str:
    label = label.lower()
    mapping = {
        "plastic":   "🧴", "metal":    "🔩", "glass":    "🍶",
        "paper":     "📄", "trash":    "🗑️", "battery":  "🔋",
        "clothes":   "👕", "cardboard":"📦", "organic":  "🌿",
        "e-waste":   "💻", "medical":  "💊", "rubber":   "⚫",
    }
    for k, v in mapping.items():
        if k in label:
            return v
    return "♻️"

# ====================== MAIN LAYOUT ====================== #
left_col, right_col = st.columns([1.1, 1], gap="large")

# ---- LEFT: Input Panel ---- #
with left_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">01 &nbsp; Input Source</div>', unsafe_allow_html=True)

    # Mode selector
    mode = st.radio(
        "Choose input method",
        options=["📁  Upload Image", "📷  Camera Capture"],
        horizontal=True,
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    image = None

    if mode == "📁  Upload Image":
        st.markdown('<div class="section-label">02 &nbsp; Select File</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop your image here",
            type=["jpg", "jpeg", "png", "webp"],
            label_visibility="collapsed"
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, use_container_width=True, caption="")
            st.markdown(f"""
            <div style="font-family:'Space Mono',monospace;font-size:0.7rem;
                        color:var(--muted);margin-top:6px;letter-spacing:1px;">
                {uploaded.name.upper()} &nbsp;·&nbsp;
                {image.width}×{image.height}px &nbsp;·&nbsp;
                {round(uploaded.size/1024,1)} KB
            </div>
            """, unsafe_allow_html=True)

    else:  # Camera
        st.markdown('<div class="section-label">02 &nbsp; Camera</div>', unsafe_allow_html=True)
        cam_img = st.camera_input(
            "Point camera at waste item",
            label_visibility="collapsed"
        )
        if cam_img:
            image = Image.open(cam_img).convert("RGB")

    # Detect Button
    st.markdown("<br>", unsafe_allow_html=True)
    detect_clicked = st.button(
        "⚡  ANALYZE WASTE",
        type="primary",
        use_container_width=True,
        disabled=(image is None or model is None)
    )

    if model is None:
        st.warning("⚠️ Model not found — ensure `final_garbage_model.keras` is in repo root.")

    st.markdown('</div>', unsafe_allow_html=True)

# ---- RIGHT: Results Panel ---- #
with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">03 &nbsp; Detection Results</div>', unsafe_allow_html=True)

    if detect_clicked and image and model:
        with st.spinner(""):
            label, confidence, top3 = predict(image)

        emoji = get_waste_emoji(label)
        pct   = round(confidence * 100, 1)

        # Main result
        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">Detected Class</div>
            <div class="result-class">{emoji} &nbsp;{label}</div>
            <div class="result-conf">Confidence: {pct}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Top-3 bars
        st.markdown('<div class="section-label" style="margin-top:20px">Top Predictions</div>', unsafe_allow_html=True)
        for i, (cls, prob) in enumerate(top3):
            bar_w  = round(prob * 100, 1)
            opacity = 1.0 if i == 0 else 0.65 - i * 0.15
            st.markdown(f"""
            <div class="pred-row" style="opacity:{opacity}">
                <span class="pred-name">{get_waste_emoji(cls)} {cls}</span>
                <div class="pred-bar-bg">
                    <div class="pred-bar-fill" style="width:{bar_w}%"></div>
                </div>
                <span class="pred-pct">{bar_w}%</span>
            </div>
            """, unsafe_allow_html=True)

        # Plotly chart
        st.markdown('<div class="section-label" style="margin-top:20px">Probability Chart</div>', unsafe_allow_html=True)
        labels = [x[0] for x in top3]
        probs  = [round(x[1]*100, 2) for x in top3]
        fig = px.bar(
            x=probs, y=labels,
            orientation='h',
            text=[f"{p}%" for p in probs],
            color=probs,
            color_continuous_scale=["#162016", "#4ade80"],
        )
        fig.update_traces(textposition="outside", marker_line_width=0)
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Mono", color="#6b8f6b", size=11),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, autorange="reversed",
                       tickfont=dict(color="#e8f5e8", size=11)),
            coloraxis_showscale=False,
            margin=dict(l=0, r=40, t=0, b=0),
            height=140,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    elif detect_clicked and not model:
        st.error("❌ Model not loaded.")
    else:
        # Placeholder state
        st.markdown("""
        <div style="
            text-align:center;
            padding:60px 20px;
            color:var(--muted);
        ">
            <div style="font-size:3.5rem;margin-bottom:16px;opacity:0.3">♻️</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.72rem;
                        letter-spacing:2px;text-transform:uppercase;">
                Awaiting Input
            </div>
            <div style="font-size:0.85rem;margin-top:8px;opacity:0.7;">
                Upload or capture an image,<br>then click Analyze.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ====================== BOTTOM INFO ROW ====================== #
st.markdown("<br>", unsafe_allow_html=True)
info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown("""
    <div style="background:var(--surface);border:1px solid var(--border);
                border-radius:10px;padding:16px 20px;">
        <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
                    letter-spacing:2px;color:var(--muted);text-transform:uppercase;
                    margin-bottom:6px;">Model</div>
        <div style="font-family:'Syne',sans-serif;font-weight:700;color:var(--accent)">
            MobileNetV2
        </div>
        <div style="font-size:0.78rem;color:var(--muted);margin-top:2px;">
            Transfer Learning · Fine-tuned
        </div>
    </div>
    """, unsafe_allow_html=True)

with info_col2:
    st.markdown(f"""
    <div style="background:var(--surface);border:1px solid var(--border);
                border-radius:10px;padding:16px 20px;">
        <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
                    letter-spacing:2px;color:var(--muted);text-transform:uppercase;
                    margin-bottom:6px;">Input Size</div>
        <div style="font-family:'Syne',sans-serif;font-weight:700;color:var(--accent)">
            160 × 160
        </div>
        <div style="font-size:0.78rem;color:var(--muted);margin-top:2px;">
            RGB · Normalized
        </div>
    </div>
    """, unsafe_allow_html=True)

with info_col3:
    names_str = " · ".join(class_names[:6]) + (" · …" if len(class_names) > 6 else "")
    st.markdown(f"""
    <div style="background:var(--surface);border:1px solid var(--border);
                border-radius:10px;padding:16px 20px;">
        <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
                    letter-spacing:2px;color:var(--muted);text-transform:uppercase;
                    margin-bottom:6px;">Classes ({len(class_names)})</div>
        <div style="font-family:'Space Mono',monospace;font-size:0.68rem;
                    color:var(--muted);line-height:1.6;">
            {names_str if names_str else "Loading..."}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ====================== FOOTER ====================== #
st.markdown("""
<div style="text-align:center;padding:32px 0 8px;
            font-family:'Space Mono',monospace;font-size:0.65rem;
            letter-spacing:2px;color:#2a3d2a;text-transform:uppercase;">
    WasteAI &nbsp;·&nbsp; Powered by TensorFlow & Streamlit &nbsp;·&nbsp; ♻️
</div>
""", unsafe_allow_html=True)

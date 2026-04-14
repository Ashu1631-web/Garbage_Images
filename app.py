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

# ====================== SESSION STATE ====================== #
if "session_predictions" not in st.session_state:
    st.session_state.session_predictions = []   # list of (label, confidence)
if "session_class_count" not in st.session_state:
    st.session_state.session_class_count = {}   # {class: count}

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
st.sidebar.title("⚙️ Menu")
st.sidebar.success("Model Ready ✅" if model else "Model Error ❌")
st.sidebar.info(f"Total Classes: {len(class_names)}")
st.sidebar.markdown("---")
st.sidebar.markdown("**Supported Waste Types:**")
for name in class_names:
    st.sidebar.write(f"• {name}")

if st.sidebar.button("🔄 Clear Session Data"):
    st.session_state.session_predictions = []
    st.session_state.session_class_count = {}
    st.sidebar.success("Session cleared!")

# ====================== TITLE ====================== #
st.title("♻️ Waste Garbage Management System (AI)")
st.markdown("""
### 📌 About This Project
This AI-powered system classifies waste from images into categories:
- 🧴 Plastic | 🔩 Metal | 📄 Paper | 📦 Cardboard
- 🍶 Glass | 🔋 Battery | 👕 Clothes | 🗑️ Trash

Upload an image or use your **Live Webcam** to detect waste type instantly.

👉 Built using **MobileNetV2 + Transfer Learning**
""")
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
            return "Mismatch Error", 0.0, [], preds

        top3_idx = preds.argsort()[-3:][::-1]
        results = [(class_names[int(i)], float(preds[i])) for i in top3_idx]
        return results[0][0], results[0][1], results, preds

    except Exception as e:
        st.error(f"⚠️ Prediction Error: {str(e)}")
        return "Error", 0.0, [], np.zeros(len(class_names))

# ====================== GRAPH 5: RGB HISTOGRAM ====================== #
def graph_rgb_histogram(image: Image.Image):
    img_array = np.array(image.resize((160, 160)).convert("RGB"))
    fig = go.Figure()
    colors = ["red", "green", "blue"]
    channels = ["Red", "Green", "Blue"]
    for i, (color, channel) in enumerate(zip(colors, channels)):
        counts, bins = np.histogram(img_array[:, :, i].flatten(), bins=64, range=(0, 255))
        fig.add_trace(go.Scatter(
            x=bins[:-1], y=counts,
            mode="lines", fill="tozeroy",
            name=channel,
            line=dict(color=color, width=1.5),
            opacity=0.6
        ))
    fig.update_layout(
        title="📊 Graph 5 — Image RGB Channel Distribution",
        xaxis_title="Pixel Value (0–255)",
        yaxis_title="Frequency",
        template="plotly_dark",
        height=300,
        legend=dict(orientation="h")
    )
    st.plotly_chart(fig, use_container_width=True)

# ====================== GRAPH 6: PIXEL INTENSITY HEATMAP ====================== #
def graph_pixel_heatmap(image: Image.Image):
    img_gray = np.array(image.resize((80, 80)).convert("L"))
    fig = go.Figure(data=go.Heatmap(
        z=img_gray,
        colorscale="Viridis",
        showscale=True
    ))
    fig.update_layout(
        title="🌡️ Graph 6 — Image Pixel Intensity Heatmap",
        template="plotly_dark",
        height=300,
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig, use_container_width=True)

# ====================== SHOW RESULTS ====================== #
def show_results(image: Image.Image):
    if model is None:
        st.error("❌ Model not loaded.")
        return
    if not class_names:
        st.error("❌ class_names.json not found.")
        return

    with st.spinner("Analyzing..."):
        label, confidence, top3, all_preds = predict(image)

    # Update session state
    st.session_state.session_predictions.append((label, confidence))
    st.session_state.session_class_count[label] = \
        st.session_state.session_class_count.get(label, 0) + 1

    st.success(f"✅ Predicted: **{label.upper()}**")
    st.metric(label="Confidence", value=f"{round(confidence * 100, 2)}%")

    st.subheader("🔥 Top 3 Predictions (Text)")
    for rank, (lbl, prob) in enumerate(top3, start=1):
        st.write(f"**#{rank}** {lbl} → `{round(prob * 100, 2)}%`")
        st.progress(float(prob))

    st.divider()
    st.subheader("📈 Prediction Graphs")

    col_a, col_b = st.columns(2)

    # ── GRAPH 1: Top-3 Horizontal Bar ──────────────────────────── #
    with col_a:
        labels_top3 = [x[0] for x in top3]
        vals_top3   = [round(x[1]*100, 2) for x in top3]
        fig1 = go.Figure(go.Bar(
            x=vals_top3,
            y=labels_top3,
            orientation="h",
            marker=dict(
                color=vals_top3,
                colorscale="Teal",
                showscale=False
            ),
            text=[f"{v}%" for v in vals_top3],
            textposition="auto"
        ))
        fig1.update_layout(
            title="📊 Graph 1 — Top 3 Confidence (Bar)",
            xaxis_title="Confidence %",
            template="plotly_dark",
            height=300
        )
        st.plotly_chart(fig1, use_container_width=True)

    # ── GRAPH 2: All-Class Pie Chart ───────────────────────────── #
    with col_b:
        pie_labels = class_names
        pie_vals   = [round(float(p)*100, 2) for p in all_preds]
        fig2 = go.Figure(go.Pie(
            labels=pie_labels,
            values=pie_vals,
            hole=0.35,
            textinfo="label+percent",
            marker=dict(colors=px.colors.qualitative.Safe)
        ))
        fig2.update_layout(
            title="🥧 Graph 2 — All Classes Probability (Pie)",
            template="plotly_dark",
            height=300
        )
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    # ── GRAPH 3: Gauge Meter ───────────────────────────────────── #
    with col_c:
        fig3 = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=round(confidence * 100, 2),
            title={"text": f"🎯 Graph 3 — Confidence Gauge<br>{label.upper()}"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "#00cc96"},
                "steps": [
                    {"range": [0,  40], "color": "#ef553b"},
                    {"range": [40, 70], "color": "#ffa15a"},
                    {"range": [70,100], "color": "#00cc96"},
                ],
                "threshold": {
                    "line":  {"color": "white", "width": 3},
                    "thickness": 0.75,
                    "value": 70
                }
            }
        ))
        fig3.update_layout(template="plotly_dark", height=300)
        st.plotly_chart(fig3, use_container_width=True)

    # ── GRAPH 4: Radar Chart (Top-6 classes) ──────────────────── #
    with col_d:
        top6_idx = all_preds.argsort()[-6:][::-1]
        radar_labels = [class_names[i] for i in top6_idx]
        radar_vals   = [round(float(all_preds[i])*100, 2) for i in top6_idx]
        radar_labels_closed = radar_labels + [radar_labels[0]]
        radar_vals_closed   = radar_vals   + [radar_vals[0]]

        fig4 = go.Figure(go.Scatterpolar(
            r=radar_vals_closed,
            theta=radar_labels_closed,
            fill="toself",
            line=dict(color="#636efa", width=2)
        ))
        fig4.update_layout(
            title="🕸️ Graph 4 — Top-6 Radar Chart",
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            template="plotly_dark",
            height=300
        )
        st.plotly_chart(fig4, use_container_width=True)

    col_e, col_f = st.columns(2)

    # ── GRAPH 5 & 6: Image-based ──────────────────────────────── #
    with col_e:
        graph_rgb_histogram(image)

    with col_f:
        graph_pixel_heatmap(image)

    # ── GRAPH 10: Full Sorted Bar (all classes) ────────────────── #
    sorted_idx  = all_preds.argsort()[::-1]
    all_labels  = [class_names[i] for i in sorted_idx]
    all_vals    = [round(float(all_preds[i])*100, 2) for i in sorted_idx]
    bar_colors  = ["#00cc96" if l == label else "#636efa" for l in all_labels]

    fig10 = go.Figure(go.Bar(
        x=all_labels,
        y=all_vals,
        marker_color=bar_colors,
        text=[f"{v}%" for v in all_vals],
        textposition="outside"
    ))
    fig10.update_layout(
        title="📊 Graph 10 — All Classes Ranked by Probability (Green = Predicted)",
        xaxis_title="Waste Class",
        yaxis_title="Confidence %",
        template="plotly_dark",
        height=350
    )
    st.plotly_chart(fig10, use_container_width=True)

    # ── GRAPH 9: Treemap ──────────────────────────────────────── #
    tree_labels = class_names
    tree_vals   = [max(float(p)*100, 0.01) for p in all_preds]
    fig9 = go.Figure(go.Treemap(
        labels=tree_labels,
        parents=["" for _ in tree_labels],
        values=tree_vals,
        textinfo="label+value",
        marker=dict(colorscale="RdYlGn", colors=tree_vals, showscale=True)
    ))
    fig9.update_layout(
        title="🗂️ Graph 9 — Probability Treemap (Bigger = Higher Confidence)",
        template="plotly_dark",
        height=350
    )
    st.plotly_chart(fig9, use_container_width=True)


# ====================== TABS ====================== #
tab1, tab2, tab3 = st.tabs(["📤 Upload Image", "📷 Live Webcam", "📊 Session Analytics"])

with tab1:
    st.header("📤 Upload Image for Detection")
    col1, col2 = st.columns([1, 2])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload a waste image",
            type=["jpg", "jpeg", "png"]
        )
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", width=280)
            detect_btn = st.button("🔍 Detect Waste", use_container_width=True)
        else:
            detect_btn = False

    with col2:
        if uploaded_file and detect_btn:
            show_results(image)
        elif not uploaded_file:
            st.info("👈 Please upload an image to begin detection.")

with tab2:
    st.header("📷 Live Webcam Detection")
    cam_image = st.camera_input("Point camera at waste item")

    if cam_image:
        col1, col2 = st.columns([1, 2])
        with col1:
            image = Image.open(cam_image).convert("RGB")
            st.image(image, caption="Captured Image", width=280)
        with col2:
            show_results(image)

# ====================== TAB 3: SESSION ANALYTICS ====================== #
with tab3:
    st.header("📊 Session Analytics")

    if not st.session_state.session_predictions:
        st.info("🔍 No predictions yet. Upload images to see session analytics.")
    else:
        total = len(st.session_state.session_predictions)
        avg_conf = np.mean([c for _, c in st.session_state.session_predictions]) * 100

        m1, m2 = st.columns(2)
        m1.metric("Total Predictions This Session", total)
        m2.metric("Average Confidence", f"{round(avg_conf, 2)}%")

        st.divider()

        col_g, col_h = st.columns(2)

        # ── GRAPH 7: Session Class Frequency Bar ──────────────── #
        with col_g:
            sess_classes = list(st.session_state.session_class_count.keys())
            sess_counts  = list(st.session_state.session_class_count.values())

            fig7 = go.Figure(go.Bar(
                x=sess_classes,
                y=sess_counts,
                marker=dict(
                    color=sess_counts,
                    colorscale="Plasma",
                    showscale=False
                ),
                text=sess_counts,
                textposition="outside"
            ))
            fig7.update_layout(
                title="📦 Graph 7 — Class Detection Frequency (Session)",
                xaxis_title="Waste Class",
                yaxis_title="Times Detected",
                template="plotly_dark",
                height=350
            )
            st.plotly_chart(fig7, use_container_width=True)

        # ── GRAPH 8: Confidence Trend Line ────────────────────── #
        with col_h:
            pred_nums   = list(range(1, total + 1))
            conf_vals   = [round(c*100, 2) for _, c in st.session_state.session_predictions]
            pred_labels = [l for l, _ in st.session_state.session_predictions]

            fig8 = go.Figure()
            fig8.add_trace(go.Scatter(
                x=pred_nums,
                y=conf_vals,
                mode="lines+markers+text",
                text=pred_labels,
                textposition="top center",
                marker=dict(size=10, color=conf_vals, colorscale="Viridis", showscale=True),
                line=dict(color="#ab63fa", width=2)
            ))
            fig8.add_hline(
                y=70,
                line_dash="dash",
                line_color="orange",
                annotation_text="70% threshold"
            )
            fig8.update_layout(
                title="📈 Graph 8 — Confidence Trend Over Session",
                xaxis_title="Prediction #",
                yaxis_title="Confidence %",
                template="plotly_dark",
                height=350
            )
            st.plotly_chart(fig8, use_container_width=True)

        # ── Session History Table ─────────────────────────────── #
        st.subheader("📋 Session History")
        df = pd.DataFrame(
            [(i+1, lbl, f"{round(conf*100, 2)}%")
             for i, (lbl, conf) in enumerate(st.session_state.session_predictions)],
            columns=["#", "Predicted Class", "Confidence"]
        )
        st.dataframe(df, use_container_width=True)

# ====================== FOOTER ====================== #
st.divider()
st.caption("🤖 Powered by TensorFlow & Streamlit | Waste AI Classification System")

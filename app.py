"""
app.py — Streamlit multi-page dashboard for Soil & Land Use Classification
         using SE-MobileNetV2 on the EuroSAT dataset.

Run:
    cd streamlit_app
    streamlit run app.py
"""

# ── stdlib / third-party ──────────────────────────────────────────────────────
import os
import sys
import json
import warnings
import numpy as np

import matplotlib
matplotlib.use("Agg")          # must be before any other matplotlib import
import matplotlib.pyplot as plt

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

warnings.filterwarnings("ignore")

# ── local modules ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from model import (
    focal_loss,
    build_se_mobilenetv2,
    build_feature_extractor,
    agentic_router,
    make_gradcam,
    EUROSAT_CLASSES,
    NUM_CLASSES,
    IMG_SIZE,
    CONFIDENCE_THRESHOLD,
)
from data_utils import preprocess_single_image, CLASS_COLORS
from artefact_paths import (
    artefact_paths,
    ensure_artefacts,
    artefacts_json_path,
    bootstrap_artefacts_json,
)
from ui_components import NAV_CARDS, render_architecture_diagram

# ── paths (resolved after bootstrap so artefacts.json is always created) ────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _refresh_paths():
    p = ensure_artefacts()
    return (
        p["dir"],
        p["model"],
        p["json"],
        p["knn"],
        p["scaler"],
        p["x_test"],
        p["y_test"],
    )


ART_DIR, MODEL_PATH, ART_JSON, KNN_PATH, SCALER_PATH, XTEST_PATH, YTEST_PATH = _refresh_paths()


def fmt_pct(value):
    """Format accuracy: train/export store 0–100, older files may use 0–1."""
    if value is None:
        return 0.0
    v = float(value)
    return v if v > 1.0 else v * 100.0


def style_plot(fig, height=None):
    """Apply consistent dark-theme styling to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#151b28",
        font=dict(color="#e2e8f0", family="DM Sans, system-ui, sans-serif"),
        margin=dict(t=48, b=40, l=48, r=24),
    )
    fig.update_xaxes(gridcolor="#2a3548", zerolinecolor="#2a3548")
    fig.update_yaxes(gridcolor="#2a3548", zerolinecolor="#2a3548")
    if height is not None:
        fig.update_layout(height=height)
    return fig


def render_nav_cards():
    """Interactive navigation cards on the home page."""
    st.markdown('<div class="section-header">Explore the dashboard</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, card in enumerate(NAV_CARDS):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div class="nav-card" style="--accent:{card['accent']};">
                    <div class="nav-card__accent"></div>
                    <div class="nav-card__title">{card['title']}</div>
                    <div class="nav-card__subtitle">{card['subtitle']}</div>
                    <div class="nav-card__desc">{card['desc']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                f"Open {card['title']}",
                key=f"nav_btn_{card['page']}",
                use_container_width=True,
                type="primary" if i == 0 else "secondary",
            ):
                st.session_state.nav_page = card["page"]
                st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# Page config  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EuroSAT Land Use Classification",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', system-ui, sans-serif; }
.block-container { padding-top: 1.5rem; max-width: 1280px; }
.stApp { background: #0b0f19; }

/* ── metric cards ── */
.metric-card {
    background: linear-gradient(145deg, #151b28 0%, #1a2236 100%);
    border: 1px solid #2a3548;
    border-radius: 12px;
    padding: 22px 18px;
    text-align: left;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
    margin-bottom: 8px;
    border-top: 3px solid #38bdf8;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 32px rgba(56, 189, 248, 0.12);
}
.metric-card .value {
    font-size: 2rem; font-weight: 700; color: #f1f5f9;
    line-height: 1.1; font-variant-numeric: tabular-nums;
}
.metric-card .label { font-size: 0.8rem; color: #94a3b8; margin-top: 6px; font-weight: 500; }
.metric-card.accent-green { border-top-color: #34d399; }
.metric-card.accent-amber { border-top-color: #fbbf24; }
.metric-card.accent-violet { border-top-color: #a78bfa; }
.metric-card.accent-teal { border-top-color: #2dd4bf; }

/* ── badges ── */
.badge {
    display: inline-block; padding: 5px 12px; border-radius: 6px;
    font-size: 0.78rem; font-weight: 600; margin: 4px 4px 4px 0;
}
.badge-blue { background: rgba(56, 189, 248, 0.15); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.35); }
.badge-green { background: rgba(52, 211, 153, 0.15); color: #6ee7b7; border: 1px solid rgba(52, 211, 153, 0.35); }
.badge-orange { background: rgba(251, 146, 60, 0.15); color: #fdba74; border: 1px solid rgba(251, 146, 60, 0.35); }

.section-header {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 1.55rem; font-weight: 400; color: #f1f5f9;
    margin: 28px 0 14px 0; padding-bottom: 8px;
    border-bottom: 1px solid #2a3548;
}

.hero-box {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 40%, #0c4a6e 100%);
    border: 1px solid #334155; border-radius: 16px;
    padding: 48px 40px; color: #f8fafc; margin-bottom: 24px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45);
    position: relative; overflow: hidden;
}
.hero-box::before {
    content: ''; position: absolute; top: -40%; right: -10%;
    width: 50%; height: 140%;
    background: radial-gradient(circle, rgba(56, 189, 248, 0.18) 0%, transparent 70%);
    pointer-events: none;
}
.hero-box h1 {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 2.4rem; font-weight: 400; margin-bottom: 12px;
    letter-spacing: -0.02em; position: relative;
}
.hero-box p { font-size: 1.05rem; color: #cbd5e1; line-height: 1.65; max-width: 760px; position: relative; }

.class-card {
    background: #151b28; border: 1px solid #2a3548; border-radius: 10px;
    padding: 14px 10px; text-align: center; margin-bottom: 8px;
    transition: border-color 0.2s, transform 0.2s;
}
.class-card:hover { border-color: #38bdf8; transform: translateY(-1px); }
.class-card .swatch { width: 28px; height: 28px; border-radius: 50%; margin: 0 auto 8px; box-shadow: 0 0 12px rgba(0,0,0,0.4); }
.class-card .name { font-size: 0.72rem; font-weight: 600; color: #cbd5e1; line-height: 1.25; }

/* ── nav cards + buttons ── */
.nav-card {
    position: relative; background: #151b28; border: 1px solid #2a3548;
    border-radius: 12px; padding: 18px 16px 12px; margin-bottom: 6px;
    min-height: 130px; overflow: hidden;
}
.nav-card__accent {
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--accent); box-shadow: 0 0 16px var(--accent);
}
.nav-card__title { font-size: 1.15rem; font-weight: 700; color: #f1f5f9; margin-top: 4px; }
.nav-card__subtitle { font-size: 0.78rem; color: var(--accent); font-weight: 600; margin: 2px 0 8px; text-transform: uppercase; letter-spacing: 0.06em; }
.nav-card__desc { font-size: 0.82rem; color: #94a3b8; line-height: 1.45; }

div[data-testid="column"] .stButton > button {
    width: 100%; border-radius: 8px; font-weight: 600;
    border: 1px solid #334155; background: #1e293b; color: #e2e8f0;
    transition: all 0.2s ease;
}
div[data-testid="column"] .stButton > button:hover {
    border-color: #38bdf8; background: #0f2744; color: #f8fafc;
    box-shadow: 0 0 20px rgba(56, 189, 248, 0.25);
}
div[data-testid="column"] .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #0369a1, #0ea5e9);
    border-color: #38bdf8; color: #fff;
}

/* ── architecture diagram ── */
.arch-wrap {
    background: linear-gradient(180deg, #0f1419 0%, #151b28 100%);
    border: 1px solid #2a3548; border-radius: 16px;
    padding: 28px 24px 20px; margin: 8px 0 16px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}
.arch-pipeline { display: flex; flex-direction: column; align-items: center; gap: 0; }
.arch-stage { width: 100%; display: flex; justify-content: center; }
.arch-stage--grid {
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px;
    max-width: 920px; width: 100%;
}
.arch-stage--router { max-width: 720px; width: 100%; }

.arch-node {
    text-align: center; padding: 16px 24px; border-radius: 12px;
    border: 1px solid #334155; background: #1a2236; min-width: 200px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.arch-node strong { display: block; color: #f1f5f9; font-size: 1rem; margin: 4px 0; }
.arch-node small { display: block; color: #64748b; font-size: 0.72rem; line-height: 1.4; }
.arch-node__tag {
    font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #38bdf8; font-weight: 700;
}
.arch-node--input { border-color: #38bdf8; background: linear-gradient(180deg, #0c4a6e33, #1a2236); }
.arch-node--backbone { border-color: #a78bfa; min-width: 280px; }
.arch-node--backbone .arch-node__tag { color: #c4b5fd; }
.arch-node--fuse { border-color: #34d399; min-width: 300px; }
.arch-node--fuse .arch-node__tag { color: #6ee7b7; }
.arch-node--out { border-color: #fbbf24; }
.arch-node--out .arch-node__tag { color: #fcd34d; }

.arch-tap {
    text-align: center; padding: 12px 8px; border-radius: 10px;
    background: #1e293b; border: 1px solid #475569;
}
.arch-tap__layer { display: block; font-size: 0.75rem; font-weight: 700; color: #e2e8f0; }
.arch-tap__dim { display: block; font-size: 0.68rem; color: #94a3b8; margin: 2px 0 6px; }
.arch-mini {
    font-size: 0.65rem; color: #38bdf8; background: rgba(56,189,248,0.12);
    border-radius: 4px; padding: 3px 6px; display: inline-block;
}

.arch-connector { height: 28px; display: flex; justify-content: center; align-items: center; }
.arch-connector span {
    display: block; width: 2px; height: 100%;
    background: linear-gradient(180deg, #38bdf8, #334155);
}
.arch-connector--fan span { height: 16px; }
.arch-connector--split { height: 20px; }

.arch-router {
    width: 100%; padding: 18px; border-radius: 12px;
    border: 1px dashed #475569; background: #121820;
}
.arch-router__title {
    text-align: center; font-size: 0.8rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 14px;
}
.arch-router__paths { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.arch-route {
    padding: 14px; border-radius: 10px; text-align: center;
}
.arch-route strong { display: block; color: #f1f5f9; font-size: 0.95rem; margin: 4px 0; }
.arch-route small { color: #64748b; font-size: 0.7rem; }
.arch-route__cond {
    font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; padding: 3px 8px; border-radius: 4px; display: inline-block;
}
.arch-route--high {
    background: rgba(52, 211, 153, 0.1); border: 1px solid rgba(52, 211, 153, 0.35);
}
.arch-route--high .arch-route__cond { color: #6ee7b7; background: rgba(52, 211, 153, 0.2); }
.arch-route--low {
    background: rgba(251, 146, 60, 0.1); border: 1px solid rgba(251, 146, 60, 0.35);
}
.arch-route--low .arch-route__cond { color: #fdba74; background: rgba(251, 146, 60, 0.2); }

.arch-legend {
    display: flex; flex-wrap: wrap; gap: 16px; justify-content: center;
    margin-top: 18px; padding-top: 14px; border-top: 1px solid #2a3548;
    font-size: 0.78rem; color: #94a3b8;
}
.arch-legend .dot {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    margin-right: 6px; vertical-align: middle;
}
.dot--se { background: #38bdf8; }
.dot--curr { background: #a78bfa; }
.dot--route { background: #34d399; }

[data-testid="stSidebar"] {
    background: #0f1419 !important;
    border-right: 1px solid #2a3548;
}
[data-testid="stSidebar"] .stRadio label {
    padding: 8px 12px; border-radius: 8px; width: 100%;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: #1a2236;
}

@media (max-width: 768px) {
    .arch-stage--grid { grid-template-columns: repeat(2, 1fr); }
    .arch-router__paths { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Cached resource loaders
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading SE-MobileNetV2 model…")
def load_model():
    """Load the trained Keras model with custom focal loss."""
    import tensorflow as tf
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        gamma, alpha = 2.0, 0.25
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={"focal_loss_fn": focal_loss(gamma, alpha, NUM_CLASSES)},
        )
        return model
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None


@st.cache_resource(show_spinner="Loading KNN fallback & scaler…")
def load_knn_and_scaler():
    """Load pickled KNN fallback classifier and embedding scaler."""
    import pickle
    knn, scaler = None, None
    if os.path.exists(KNN_PATH):
        with open(KNN_PATH, "rb") as f:
            knn = pickle.load(f)
    if os.path.exists(SCALER_PATH):
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
    return knn, scaler


@st.cache_resource(show_spinner=False)
def load_feature_extractor(model):
    """Build feature extractor from loaded SE model."""
    if model is None:
        return None
    try:
        return build_feature_extractor(model)
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_artefacts(_json_path: str, _mtime: float):
    """Load artefacts.json; cache key includes path and file mtime."""
    if not _json_path or not os.path.isfile(_json_path):
        return None
    try:
        with open(_json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_artefacts():
    """Load artefacts with up-to-date path and cache invalidation on file change."""
    json_path = artefacts_json_path()
    if not os.path.isfile(json_path):
        bootstrap_artefacts_json()
        json_path = artefacts_json_path()
    mtime = os.path.getmtime(json_path) if os.path.isfile(json_path) else 0.0
    return load_artefacts(json_path, mtime)


def artefacts_missing_banner():
    """Show guidance when artefacts are not yet available."""
    st.warning(
        "**Training artefacts not found.** "
        "Export from Colab with `export_artefacts_colab.py`, or run training locally.",
    )
    st.markdown(
        "Expected files in `streamlit_app/artefacts/`: `artefacts.json`, "
        "`se_mobilenetv2_eurosat.h5`, `knn_fallback.pkl`, `emb_scaler.pkl`."
    )
    tab1, tab2 = st.tabs(["Colab export", "Local training"])
    with tab1:
        st.code(
            "# After training in Colab:\n"
            "%run export_artefacts_colab.py\n"
            "# Then download the artefacts/ folder into streamlit_app/",
            language="python",
        )
    with tab2:
        st.code("cd streamlit_app\npython train.py --data_dir ../data/EuroSAT", language="bash")


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────────────────
PAGES = [
    "Home",
    "Predict",
    "Model Comparison",
    "Training Progress",
    "Explainability",
    "Clustering",
    "About",
]

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "Home"

with st.sidebar:
    st.markdown("### EuroSAT Dashboard")
    st.caption("SE-MobileNetV2 · Curriculum Learning · Agentic Router")
    st.markdown("---")
    selected = st.radio(
        "Navigation",
        PAGES,
        index=PAGES.index(st.session_state.nav_page),
        label_visibility="collapsed",
    )
    if selected != st.session_state.nav_page:
        st.session_state.nav_page = selected
    page = st.session_state.nav_page
    st.markdown("---")
    art = get_artefacts()
    model_ok = os.path.exists(MODEL_PATH)
    if art:
        src = art.get("source", "artefacts.json")
        st.success("Metrics loaded")
        st.caption(f"Source: {src}")
        st.metric("SE-MobileNetV2", f"{fmt_pct(art.get('se_acc')):.1f}%")
        st.metric("Agentic router", f"{fmt_pct(art.get('routed_acc')):.1f}%")
        if not model_ok:
            st.caption("Model weights not found — charts work; live predict needs `.h5`.")
    else:
        st.error("Artefacts not found")
        if st.button("Load metrics from notebook", key="bootstrap_artefacts"):
            bootstrap_artefacts_json()
            load_artefacts.clear()
            st.rerun()
        st.caption("Run: `python extract_notebook_metrics.py` in `streamlit_app/`")
    st.markdown("---")
    json_path = artefacts_json_path()
    st.caption(f"Metrics: `{json_path}`")
    if not os.path.isfile(json_path):
        st.caption(f"Directory `{ART_DIR}` exists but `artefacts.json` is missing.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═════════════════════════════════════════════════════════════════════════════

def page_home():
    art = get_artefacts()

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-box">
        <h1>Soil &amp; Land Use Classification</h1>
        <p>Multi-class satellite land-use mapping with SE-MobileNetV2,
        three-phase curriculum learning, and an agentic confidence router on the
        <strong>EuroSAT RGB</strong> benchmark (27,000 Sentinel-2 patches, 10 classes).</p>
    </div>
    """, unsafe_allow_html=True)

    render_nav_cards()

    # ── Key metrics row ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Key Results</div>', unsafe_allow_html=True)

    if art:
        src = art.get("source", "")
        if src:
            st.caption(f"Results loaded from **{src}** (Kaggle notebook run).")
        se_acc      = fmt_pct(art.get("se_acc"))
        routed_acc  = fmt_pct(art.get("routed_acc"))
        van_acc     = fmt_pct(art.get("van_acc"))
        improvement = routed_acc - van_acc
        n_test      = art.get("n_test", 0)
        n_knn       = art.get("n_knn_routed", 0)
        knn_pct     = (n_knn / n_test * 100) if n_test > 0 else 0

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{se_acc:.1f}%</div>
                <div class="label">SE-MobileNetV2 Accuracy</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card accent-green">
                <div class="value">{routed_acc:.1f}%</div>
                <div class="label">Agentic Router Accuracy</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            sign = "+" if improvement >= 0 else ""
            st.markdown(f"""
            <div class="metric-card accent-amber">
                <div class="value">{sign}{improvement:.1f}%</div>
                <div class="label">Improvement vs Vanilla MobileNetV2</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card accent-violet">
                <div class="value">{knn_pct:.1f}%</div>
                <div class="label">Samples Routed to KNN Fallback</div>
            </div>""", unsafe_allow_html=True)
    else:
        artefacts_missing_banner()
        # Show placeholder cards
        c1, c2, c3, c4 = st.columns(4)
        for col, label in zip(
            [c1, c2, c3, c4],
            ["SE Accuracy", "Router Accuracy", "Improvement", "KNN Routed %"],
        ):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="value">—</div>
                    <div class="label">{label}</div>
                </div>""", unsafe_allow_html=True)

    # ── What We Did Differently ───────────────────────────────────────────────
    st.markdown('<div class="section-header">What We Did Differently</div>', unsafe_allow_html=True)

    comparison_data = {
        "Component": [
            "Backbone",
            "Attention",
            "Loss Function",
            "Training Strategy",
            "Inference",
            "Fallback",
        ],
        "Vanilla Baseline": [
            "MobileNetV2 (frozen)",
            "None",
            "Categorical Cross-Entropy",
            "Single-phase fine-tune",
            "Argmax of softmax",
            "None",
        ],
        "Our approach": [
            "MobileNetV2 (multi-scale, 4 taps)",
            "Squeeze-and-Excitation (SE) blocks",
            "Focal Loss (γ=2, α=0.25)",
            "3-phase Curriculum Learning",
            "Agentic Confidence Router",
            "KNN on SE embeddings",
        ],
    }

    import pandas as pd
    df_comp = pd.DataFrame(comparison_data)
    st.dataframe(df_comp, use_container_width=True, hide_index=True)

    # ── Class overview grid ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">EuroSAT Land-Use Classes</div>', unsafe_allow_html=True)

    cols = st.columns(5)
    for i, cls in enumerate(EUROSAT_CLASSES):
        color = CLASS_COLORS.get(cls, "#64748b")
        with cols[i % 5]:
            st.markdown(f"""
            <div class="class-card">
                <div class="swatch" style="background:{color};"></div>
                <div class="name">{cls}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Architecture Overview</div>', unsafe_allow_html=True)
    render_architecture_diagram()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — PREDICT
# ═════════════════════════════════════════════════════════════════════════════

def page_predict():
    st.markdown('<div class="section-header">Upload &amp; Classify</div>', unsafe_allow_html=True)
    st.markdown(
        "Upload an aerial or satellite image and the model will classify it into one of "
        "the 10 EuroSAT land-use categories using the **Agentic Confidence Router**."
    )

    # Load resources
    model          = load_model()
    knn, emb_scaler = load_knn_and_scaler()
    feat_extractor = load_feature_extractor(model)
    art            = get_artefacts()

    threshold = art.get("confidence_threshold", CONFIDENCE_THRESHOLD) if art else CONFIDENCE_THRESHOLD

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded = st.file_uploader(
        "Choose an image (JPG / PNG / TIF)",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="Upload a satellite or aerial image for classification.",
    )

    if uploaded is None:
        st.info("Upload a satellite or aerial image (JPG, PNG, or TIF) to run classification.")
        return

    pil_img = Image.open(uploaded).convert("RGB")

    # ── Preprocess ────────────────────────────────────────────────────────────
    img_array = preprocess_single_image(pil_img, IMG_SIZE)   # (64,64,3) float32

    # ── Inference ─────────────────────────────────────────────────────────────
    if model is None or knn is None or emb_scaler is None or feat_extractor is None:
        # ── Demo mode ─────────────────────────────────────────────────────────
        st.warning(
            "**Live inference unavailable** — place trained weights and pickles in "
            f"`{ART_DIR}` (see sidebar for export steps)."
        )
        col_img, _ = st.columns([1, 2])
        with col_img:
            st.image(pil_img, caption="Uploaded image", use_column_width=True)
        return

    with st.spinner("Running inference…"):
        batch = img_array[np.newaxis]   # (1,64,64,3)
        final_preds, sources, se_conf, se_proba = agentic_router(
            batch, model, knn, feat_extractor, emb_scaler, threshold
        )
        pred_idx   = int(final_preds[0])
        pred_class = EUROSAT_CLASSES[pred_idx]
        confidence = float(se_conf[0])
        source     = sources[0]
        proba_vec  = se_proba[0]

        # Grad-CAM
        gradcam_result = make_gradcam(model, img_array, IMG_SIZE)

    # ── Layout: image | grad-cam | overlay ───────────────────────────────────
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Original Image**")
        st.image(pil_img, use_column_width=True)

    if gradcam_result is not None:
        cam_resized, overlay, gc_pred_idx, gc_conf = gradcam_result
        with col2:
            st.markdown("**Grad-CAM Heatmap**")
            fig_cam, ax_cam = plt.subplots(figsize=(3, 3))
            ax_cam.imshow(cam_resized, cmap="jet")
            ax_cam.axis("off")
            st.pyplot(fig_cam)
            plt.close(fig_cam)
        with col3:
            st.markdown("**Grad-CAM Overlay**")
            fig_ov, ax_ov = plt.subplots(figsize=(3, 3))
            ax_ov.imshow(overlay)
            ax_ov.axis("off")
            st.pyplot(fig_ov)
            plt.close(fig_ov)
    else:
        with col2:
            st.info("Grad-CAM not available for this model configuration.")

    # ── Prediction result ─────────────────────────────────────────────────────
    st.markdown("---")
    res_col, conf_col = st.columns([2, 1])
    with res_col:
        swatch = CLASS_COLORS.get(pred_class, "#64748b")
        st.markdown(
            f'<p style="margin:0 0 4px;font-size:0.85rem;color:#64748b;">Predicted class</p>'
            f'<h3 style="margin:0;color:#0f172a;">'
            f'<span style="display:inline-block;width:12px;height:12px;border-radius:50%;'
            f'background:{swatch};margin-right:8px;vertical-align:middle;"></span>'
            f'{pred_class}</h3>',
            unsafe_allow_html=True,
        )

        if source == "se_direct":
            st.markdown(
                '<span class="badge badge-blue">SE direct</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span class="badge badge-orange">KNN fallback</span>',
                unsafe_allow_html=True,
            )

        st.markdown(f"SE Confidence: **{confidence*100:.1f}%**  |  Threshold: **{threshold*100:.0f}%**")

    with conf_col:
        # Confidence gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%", "font": {"size": 28}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#27ae60" if confidence >= threshold else "#e67e22"},
                "steps": [
                    {"range": [0, threshold * 100], "color": "#fdebd0"},
                    {"range": [threshold * 100, 100], "color": "#d5f5e3"},
                ],
                "threshold": {
                    "line": {"color": "#c0392b", "width": 3},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
            title={"text": "SE Confidence"},
        ))
        style_plot(fig_gauge, height=220)
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Top-5 bar chart ───────────────────────────────────────────────────────
    st.markdown("#### Top-5 Class Probabilities")
    top5_idx  = np.argsort(proba_vec)[::-1][:5]
    top5_cls  = [EUROSAT_CLASSES[i] for i in top5_idx]
    top5_prob = [float(proba_vec[i]) * 100 for i in top5_idx]
    colors    = ["#34d399" if i == pred_idx else "#38bdf8" for i in top5_idx]

    fig_bar = go.Figure(go.Bar(
        x=top5_prob,
        y=top5_cls,
        orientation="h",
        marker_color=colors,
        text=[f"{p:.1f}%" for p in top5_prob],
        textposition="outside",
    ))
    fig_bar.update_layout(
        xaxis_title="Probability (%)",
        xaxis_range=[0, 110],
        height=280,
        margin=dict(t=10, b=10, l=10, r=10),
        yaxis={"autorange": "reversed"},
    )
    style_plot(fig_bar, height=280)
    st.plotly_chart(fig_bar, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL COMPARISON
# ═════════════════════════════════════════════════════════════════════════════

def page_model_comparison():
    st.markdown('<div class="section-header">Model Comparison</div>', unsafe_allow_html=True)
    st.markdown(
        "Comparing classical ML baselines, Vanilla MobileNetV2, SE-MobileNetV2, "
        "and the full Agentic Router pipeline."
    )

    art = get_artefacts()
    if art is None:
        artefacts_missing_banner()
        return

    # ── Build unified results dict ────────────────────────────────────────────
    ml_results = art.get("ml_results", {})

    all_models = {}
    for name, vals in ml_results.items():
        all_models[name] = {"acc": vals.get("acc", 0), "f1": vals.get("f1", 0)}

    all_models["Vanilla MobileNetV2"] = {
        "acc": art.get("van_acc", 0),
        "f1":  art.get("van_f1",  0),
    }
    all_models["SE-MobileNetV2"] = {
        "acc": art.get("se_acc", 0),
        "f1":  art.get("se_f1",  0),
    }
    all_models["SE + Agentic Router"] = {
        "acc": art.get("routed_acc", 0),
        "f1":  art.get("routed_f1",  0),
    }

    model_names = list(all_models.keys())
    accs  = [fmt_pct(all_models[m]["acc"]) for m in model_names]
    f1s   = [fmt_pct(all_models[m]["f1"]) for m in model_names]

    # Color coding: our models in green shades, baselines in blue
    our_models = {"SE-MobileNetV2", "SE + Agentic Router", "Vanilla MobileNetV2"}
    colors_acc = []
    colors_f1  = []
    for m in model_names:
        if m == "SE + Agentic Router":
            colors_acc.append("#27ae60")
            colors_f1.append("#1e8449")
        elif m == "SE-MobileNetV2":
            colors_acc.append("#2ecc71")
            colors_f1.append("#27ae60")
        elif m == "Vanilla MobileNetV2":
            colors_acc.append("#f39c12")
            colors_f1.append("#d68910")
        else:
            colors_acc.append("#2980b9")
            colors_f1.append("#1a5276")

    # ── Side-by-side bar charts ───────────────────────────────────────────────
    col_acc, col_f1 = st.columns(2)

    with col_acc:
        fig_acc = go.Figure(go.Bar(
            x=model_names,
            y=accs,
            marker_color=colors_acc,
            text=[f"{v:.1f}%" for v in accs],
            textposition="outside",
        ))
        fig_acc.update_layout(title="Accuracy (%)", yaxis_range=[0, 110], xaxis_tickangle=-30)
        style_plot(fig_acc, height=420)
        st.plotly_chart(fig_acc, use_container_width=True)

    with col_f1:
        fig_f1 = go.Figure(go.Bar(
            x=model_names,
            y=f1s,
            marker_color=colors_f1,
            text=[f"{v:.1f}%" for v in f1s],
            textposition="outside",
        ))
        fig_f1.update_layout(title="Macro F1-Score (%)", yaxis_range=[0, 110], xaxis_tickangle=-30)
        style_plot(fig_f1, height=420)
        st.plotly_chart(fig_f1, use_container_width=True)

    # ── Legend ────────────────────────────────────────────────────────────────
    st.markdown("""
    <span class="badge badge-blue">Classical ML Baselines</span>
    <span class="badge badge-orange">Vanilla MobileNetV2</span>
    <span class="badge badge-green">Our Models (SE / Router)</span>
    """, unsafe_allow_html=True)

    # ── Summary table ─────────────────────────────────────────────────────────
    st.markdown("#### Summary Table")
    import pandas as pd
    rows = []
    for m in model_names:
        rows.append({
            "Model": m,
            "Accuracy (%)": f"{fmt_pct(all_models[m]['acc']):.2f}",
            "Macro F1 (%)": f"{fmt_pct(all_models[m]['f1']):.2f}",
            "Type": (
                "Best" if m == "SE + Agentic Router"
                else "Proposed" if m in {"SE-MobileNetV2"}
                else "Baseline" if m == "Vanilla MobileNetV2"
                else "Classical ML"
            ),
        })
    df_summary = pd.DataFrame(rows)
    st.dataframe(df_summary, use_container_width=True, hide_index=True)

    # ── Radar chart ───────────────────────────────────────────────────────────
    st.markdown("#### Accuracy vs F1 Scatter")
    fig_scatter = px.scatter(
        df_summary,
        x="Accuracy (%)",
        y="Macro F1 (%)",
        text="Model",
        color="Type",
        size_max=18,
        color_discrete_map={
            "Best": "#059669",
            "Proposed": "#10b981",
            "Baseline": "#d97706",
            "Classical ML": "#2563eb",
        },
    )
    fig_scatter.update_traces(textposition="top center", marker_size=12)
    style_plot(fig_scatter, height=400)
    st.plotly_chart(fig_scatter, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — TRAINING PROGRESS
# ═════════════════════════════════════════════════════════════════════════════

def page_training_progress():
    st.markdown('<div class="section-header">Training Progress</div>', unsafe_allow_html=True)
    st.markdown(
        "Validation accuracy across all three curriculum learning phases, "
        "plus the difficulty distribution used to schedule training."
    )

    art = get_artefacts()
    if art is None:
        artefacts_missing_banner()
        return

    history    = art.get("history", {})
    ph1        = history.get("phase1_val_acc", [])
    ph2        = history.get("phase2_val_acc", [])
    ph3        = history.get("phase3_val_acc", [])
    difficulty = art.get("difficulty", [])
    tier_easy  = art.get("tier_easy_max", 0.33)
    tier_med   = art.get("tier_med_max",  0.66)

    # ── Phase summary cards ───────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    phase_info = [
        ("Phase 1 — Easy Samples (40%)",  ph1, "#2980b9"),
        ("Phase 2 — Medium Samples (70%)", ph2, "#8e44ad"),
        ("Phase 3 — Full Dataset (100%)",  ph3, "#27ae60"),
    ]
    for col, (title, vals, color) in zip([c1, c2, c3], phase_info):
        best = max(vals) * 100 if vals else 0
        epochs = len(vals)
        with col:
            st.markdown(f"""
            <div style="
                background:{color};
                border-radius:10px;
                padding:16px;
                color:white;
                text-align:center;
                margin-bottom:8px;
            ">
                <div style="font-size:0.8rem;opacity:0.9;">{title}</div>
                <div style="font-size:1.8rem;font-weight:700;">{best:.1f}%</div>
                <div style="font-size:0.75rem;opacity:0.85;">Best val acc · {epochs} epochs</div>
            </div>""", unsafe_allow_html=True)

    # ── Validation accuracy line chart ────────────────────────────────────────
    st.markdown("#### Validation Accuracy — All Phases")

    if ph1 or ph2 or ph3:
        # Build continuous x-axis with phase offsets
        offset = 0
        traces = []
        phase_boundaries = []

        for ph_vals, ph_name, ph_color in [
            (ph1, "Phase 1 (Easy)", "#2980b9"),
            (ph2, "Phase 2 (Medium)", "#8e44ad"),
            (ph3, "Phase 3 (Full)", "#27ae60"),
        ]:
            if not ph_vals:
                continue
            x = list(range(offset + 1, offset + len(ph_vals) + 1))
            traces.append(go.Scatter(
                x=x,
                y=[v * 100 for v in ph_vals],
                mode="lines+markers",
                name=ph_name,
                line=dict(color=ph_color, width=2),
                marker=dict(size=5),
            ))
            if offset > 0:
                phase_boundaries.append(offset + 0.5)
            offset += len(ph_vals)

        fig_hist = go.Figure(traces)

        # Vertical phase separator lines
        for xb in phase_boundaries:
            fig_hist.add_vline(
                x=xb,
                line_dash="dash",
                line_color="gray",
                line_width=1.5,
                annotation_text="Phase →",
                annotation_position="top",
            )

        fig_hist.update_layout(
            xaxis_title="Epoch",
            yaxis_title="Validation Accuracy (%)",
            yaxis_range=[0, 105],
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        style_plot(fig_hist, height=400)
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No training history found in artefacts.json.")

    # ── Difficulty distribution ───────────────────────────────────────────────
    st.markdown("#### Sample Difficulty Distribution")
    st.markdown(
        "Difficulty is computed as `1 − max_confidence` from a Random Forest proxy. "
        "Curriculum learning starts with easy (low-difficulty) samples."
    )

    if difficulty:
        diff_arr = np.array(difficulty)
        fig_diff = go.Figure()
        fig_diff.add_trace(go.Histogram(
            x=diff_arr,
            nbinsx=50,
            marker_color="#2980b9",
            opacity=0.75,
            name="Difficulty",
        ))
        # Tier cutoff lines
        fig_diff.add_vline(
            x=tier_easy,
            line_dash="dash",
            line_color="#27ae60",
            line_width=2,
            annotation_text=f"Easy cutoff ({tier_easy:.2f})",
            annotation_position="top right",
        )
        fig_diff.add_vline(
            x=tier_med,
            line_dash="dash",
            line_color="#e67e22",
            line_width=2,
            annotation_text=f"Medium cutoff ({tier_med:.2f})",
            annotation_position="top right",
        )
        fig_diff.update_layout(
            xaxis_title="Difficulty Score (1 − max_conf)",
            yaxis_title="Number of Samples",
        )
        style_plot(fig_diff, height=350)
        st.plotly_chart(fig_diff, use_container_width=True)

        # Stats
        n = len(diff_arr)
        n_easy = int((diff_arr <= tier_easy).sum())
        n_med  = int(((diff_arr > tier_easy) & (diff_arr <= tier_med)).sum())
        n_hard = int((diff_arr > tier_med).sum())
        st.markdown(f"""
        | Tier | Difficulty Range | Count | % |
        |------|-----------------|-------|---|
        | Easy   | 0 – {tier_easy:.2f} | {n_easy} | {n_easy/n*100:.1f}% |
        | Medium | {tier_easy:.2f} – {tier_med:.2f} | {n_med} | {n_med/n*100:.1f}% |
        | Hard   | {tier_med:.2f} – 1.0 | {n_hard} | {n_hard/n*100:.1f}% |
        """)
    else:
        st.info("No difficulty data found in artefacts.json.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — EXPLAINABILITY (GRAD-CAM)
# ═════════════════════════════════════════════════════════════════════════════

def page_explainability():
    st.markdown('<div class="section-header">Explainability — Grad-CAM &amp; SE Attention</div>', unsafe_allow_html=True)

    with st.expander("What is Grad-CAM?", expanded=True):
        st.markdown("""
        **Gradient-weighted Class Activation Mapping (Grad-CAM)** produces a coarse localisation
        map highlighting the image regions most important for a prediction.

        **How it works:**
        1. Forward-pass the image through the network.
        2. Compute the gradient of the predicted class score w.r.t. the last convolutional
           feature map.
        3. Global-average-pool the gradients to get per-channel weights.
        4. Weighted sum of the feature map channels → ReLU → resize to input resolution.

        The resulting heatmap (red = high importance, blue = low) shows *where* the model
        is looking when it makes its decision.
        """)

    with st.expander("What are SE (Squeeze-and-Excitation) blocks?", expanded=False):
        st.markdown("""
        **SE blocks** add channel-wise attention to convolutional feature maps.

        ```
        Input feature map  (H × W × C)
              │
        GlobalAveragePool  → (1 × 1 × C)   ← "Squeeze"
              │
        FC(C/r, ReLU)      → (1 × 1 × C/r)
              │
        FC(C, Sigmoid)     → (1 × 1 × C)   ← "Excitation" (channel weights)
              │
        Multiply with input                 ← "Scale"
        ```

        The network learns *which channels* are most informative for each input,
        effectively re-calibrating feature responses adaptively.

        In our **multi-scale SE-MobileNetV2**, SE blocks are applied at 4 different
        spatial resolutions (16×16, 8×8, 4×4, 2×2), then fused via concatenation.
        This lets the model attend to both fine-grained textures and high-level semantics.
        """)

    # ── Live Grad-CAM on test samples ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Live Grad-CAM Visualisation")

    model = load_model()
    art   = get_artefacts()

    if model is None:
        artefacts_missing_banner()
        return

    # Try to load saved test set
    X_test, y_test = None, None
    if os.path.exists(XTEST_PATH) and os.path.exists(YTEST_PATH):
        try:
            X_test = np.load(XTEST_PATH)
            y_test = np.load(YTEST_PATH)
        except Exception:
            X_test, y_test = None, None

    if X_test is not None and len(X_test) > 0:
        n_samples = st.slider("Number of random samples to visualise", 2, 12, 6, step=2)
        seed_val  = st.number_input("Random seed", value=42, step=1)

        rng = np.random.RandomState(int(seed_val))
        idxs = rng.choice(len(X_test), min(n_samples, len(X_test)), replace=False)

        cols_per_row = 3
        for row_start in range(0, len(idxs), cols_per_row):
            row_idxs = idxs[row_start : row_start + cols_per_row]
            cols = st.columns(len(row_idxs))
            for col, idx in zip(cols, row_idxs):
                img_arr = X_test[idx]
                true_cls = EUROSAT_CLASSES[int(y_test[idx])]
                gc = make_gradcam(model, img_arr, IMG_SIZE)
                with col:
                    if gc is not None:
                        _, overlay, pred_idx, conf = gc
                        pred_cls = EUROSAT_CLASSES[pred_idx]
                        status = "OK" if pred_cls == true_cls else "Miss"
                        fig_ov, ax_ov = plt.subplots(figsize=(3, 3))
                        ax_ov.imshow(overlay)
                        ax_ov.axis("off")
                        ax_ov.set_title(
                            f"{status}: {pred_cls}\n"
                            f"True: {true_cls}\n"
                            f"Conf: {conf*100:.1f}%",
                            fontsize=7,
                        )
                        st.pyplot(fig_ov)
                        plt.close(fig_ov)
                    else:
                        st.image(img_arr, caption=f"True: {true_cls}", use_column_width=True)
    else:
        st.info(
            "No `X_test.npy` / `y_test.npy` found in `artefacts/`.  "
            "Run `python train.py` to generate them, or upload an image on the **Predict** page."
        )

    # ── Architecture diagram description ─────────────────────────────────────
    st.markdown("---")
    st.markdown("#### SE-MobileNetV2 Architecture Detail")
    st.markdown("""
    | Layer Group | Output Shape | SE Applied? |
    |-------------|-------------|-------------|
    | Input | 64 × 64 × 3 | — |
    | MobileNetV2 block_3_expand_relu | 16 × 16 × 96 | SE (ratio=16) |
    | MobileNetV2 block_6_expand_relu | 8 × 8 × 192 | SE (ratio=16) |
    | MobileNetV2 block_13_expand_relu | 4 × 4 × 576 | SE (ratio=16) |
    | MobileNetV2 out_relu | 2 × 2 × 1280 | SE (ratio=16) |
    | GlobalAveragePool × 4 → Concat | 2144-d | — |
    | Dense(512) + BN + Dropout(0.4) | 512-d | — |
    | Dense(256) + Dropout(0.3) | 256-d | — |
    | Dense(10, softmax) | 10 | — |

    **Total parameters:** ~3.5 M  |  **Trainable (phase 3):** ~3.5 M
    """)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — CLUSTERING
# ═════════════════════════════════════════════════════════════════════════════

def page_clustering():
    st.markdown('<div class="section-header">Clustering Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        "Unsupervised clustering of SE embeddings using K-Means and K-Medoids. "
        "Good clustering metrics indicate that the learned representations are "
        "semantically meaningful even without labels."
    )

    art = get_artefacts()
    if art is None:
        artefacts_missing_banner()
        return

    clustering = art.get("clustering", {})
    if not clustering:
        st.warning("No clustering results found in artefacts.json.")
        return

    km_ari  = clustering.get("km_ari",   0)
    km_nmi  = clustering.get("km_nmi",   0)
    km_sil  = clustering.get("km_sil",   0)
    kmed_ari = clustering.get("kmed_ari", 0)
    kmed_sil = clustering.get("kmed_sil", 0)

    # ── Metric cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, "K-Means ARI",      km_ari,   "blue"),
        (c2, "K-Means NMI",      km_nmi,   "blue"),
        (c3, "K-Means Silhouette", km_sil, "teal"),
        (c4, "K-Medoids ARI",    kmed_ari, "purple"),
        (c5, "K-Medoids Silhouette", kmed_sil, "purple"),
    ]
    for col, label, val, color in metrics:
        with col:
            accent = {"blue": "accent-teal", "teal": "accent-teal", "purple": "accent-violet"}.get(color, "")
            st.markdown(f"""
            <div class="metric-card {accent}">
                <div class="value">{val:.3f}</div>
                <div class="label">{label}</div>
            </div>""", unsafe_allow_html=True)

    # ── Bar chart ─────────────────────────────────────────────────────────────
    st.markdown("#### Clustering Metrics Comparison")

    metric_names = ["K-Means ARI", "K-Means NMI", "K-Means Silhouette",
                    "K-Medoids ARI", "K-Medoids Silhouette"]
    metric_vals  = [km_ari, km_nmi, km_sil, kmed_ari, kmed_sil]
    bar_colors   = ["#2980b9", "#2980b9", "#1abc9c", "#8e44ad", "#8e44ad"]

    fig_clust = go.Figure(go.Bar(
        x=metric_names,
        y=metric_vals,
        marker_color=bar_colors,
        text=[f"{v:.3f}" for v in metric_vals],
        textposition="outside",
    ))
    fig_clust.update_layout(
        yaxis_title="Score",
        yaxis_range=[0, max(metric_vals) * 1.25 + 0.05],
    )
    style_plot(fig_clust, height=380)
    st.plotly_chart(fig_clust, use_container_width=True)

    # ── Metric explanations ───────────────────────────────────────────────────
    st.markdown("#### What Do These Metrics Mean?")
    st.markdown("""
    | Metric | Range | Interpretation |
    |--------|-------|----------------|
    | **ARI** (Adjusted Rand Index) | −1 to 1 | Measures agreement between cluster assignments and true labels. 1 = perfect, 0 = random. |
    | **NMI** (Normalised Mutual Information) | 0 to 1 | Measures shared information between clusters and labels. 1 = perfect alignment. |
    | **Silhouette Score** | −1 to 1 | Measures how well-separated clusters are. Higher = more compact, well-separated clusters. |

    **K-Means** partitions embeddings by minimising within-cluster variance (Euclidean).

    **K-Medoids** uses actual data points as cluster centres, making it more robust to outliers.
    We use a custom implementation based on alternating assignment and medoid update.

    High ARI/NMI values confirm that the SE-MobileNetV2 embeddings capture semantically
    meaningful structure — the network has learned to group similar land-use types together
    in embedding space, even without explicit clustering supervision.
    """)

    # ── Radar chart of clustering metrics ─────────────────────────────────────
    st.markdown("#### Radar Comparison: K-Means vs K-Medoids")
    categories = ["ARI", "NMI / (NMI proxy)", "Silhouette"]
    km_vals_radar   = [km_ari,   km_nmi,  km_sil]
    kmed_vals_radar = [kmed_ari, km_nmi,  kmed_sil]   # NMI not computed for kmedoids, reuse km_nmi

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=km_vals_radar + [km_vals_radar[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="K-Means",
        line_color="#2980b9",
        fillcolor="rgba(41,128,185,0.2)",
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=kmed_vals_radar + [kmed_vals_radar[0]],
        theta=categories + [categories[0]],
        fill="toself",
        name="K-Medoids",
        line_color="#8e44ad",
        fillcolor="rgba(142,68,173,0.2)",
    ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    style_plot(fig_radar, height=380)
    st.plotly_chart(fig_radar, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 7 — ABOUT
# ═════════════════════════════════════════════════════════════════════════════

def page_about():
    st.markdown('<div class="section-header">About This Project</div>', unsafe_allow_html=True)

    st.markdown("""
    ## Soil & Land Use Classification with SE-MobileNetV2

    This project tackles **multi-class land-use classification** on the
    [EuroSAT RGB dataset](https://github.com/phelber/EuroSAT) — 27,000 labelled
    satellite images across 10 land-use categories captured by the Sentinel-2 satellite.

    ### Problem statement
    Accurate, automated land-use classification from satellite imagery supports:
    - Precision agriculture and crop monitoring
    - Deforestation and environmental change detection
    - Urban planning and infrastructure mapping
    - Water body and flood risk assessment

    ### Our Approach
    We go beyond a standard fine-tuned CNN by combining four complementary innovations:
    """)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **1. Multi-Scale SE-MobileNetV2**
        - MobileNetV2 backbone with 4 intermediate feature taps
        - Squeeze-and-Excitation (SE) channel attention at each scale
        - Features fused via concatenation → rich 2144-d representation

        **2. Focal Loss**
        - Addresses class imbalance by down-weighting easy examples
        - γ = 2.0, α = 0.25 (standard settings from RetinaNet paper)
        - Forces the model to focus on hard, misclassified samples
        """)
    with col_b:
        st.markdown("""
        **3. Curriculum Learning (3 phases)**
        - Phase 1: Easy 40% of training data (low difficulty)
        - Phase 2: Medium 70% of training data
        - Phase 3: Full 100% with backbone unfreezing
        - Difficulty scored by a Random Forest proxy on PCA features

        **4. Agentic Confidence Router**
        - SE model predicts with confidence score
        - High confidence (≥ 0.60) → SE direct prediction
        - Low confidence (< 0.60) → KNN fallback on SE embeddings
        - Combines strengths of deep learning + instance-based learning
        """)

    # ── Architecture table ────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Architecture Details")
    st.markdown("""
    | Component | Details |
    |-----------|---------|
    | **Backbone** | MobileNetV2 (ImageNet pre-trained, 64×64 input) |
    | **Feature taps** | block_3, block_6, block_13, out_relu |
    | **SE ratio** | 16 (channels / 16 bottleneck) |
    | **Head** | Dense(512) → BN → Dropout(0.4) → Dense(256) → Dropout(0.3) → Dense(10) |
    | **Loss** | Focal Loss (γ=2, α=0.25) |
    | **Optimiser** | Adam (lr=1e-3 → 1e-4 → 5e-5 across phases) |
    | **Augmentation** | Rotation ±20°, shift ±10%, flip, zoom ±15% |
    | **KNN fallback** | k=7, cosine distance, on scaled SE embeddings |
    | **Confidence threshold** | 0.60 (tuned on validation set) |
    | **Dataset** | EuroSAT RGB, 27,000 images, 10 classes, 64×64 px |
    | **Train/Test split** | 80% / 20% stratified |
    """)

    # ── Dataset overview ──────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### EuroSAT Dataset Classes")
    import pandas as pd
    class_df = pd.DataFrame([
        {
            "Class": cls,
            "Description": {
                "AnnualCrop":           "Fields of annual crops (wheat, corn, etc.)",
                "Forest":               "Dense forest and woodland areas",
                "HerbaceousVegetation": "Grasslands, meadows, and shrublands",
                "Highway":              "Major roads and motorways",
                "Industrial":           "Industrial zones, factories, warehouses",
                "Pasture":              "Grazing land for livestock",
                "PermanentCrop":        "Orchards, vineyards, permanent plantations",
                "Residential":          "Urban residential areas and suburbs",
                "River":                "Rivers, streams, and waterways",
                "SeaLake":              "Open water bodies — sea, lakes, reservoirs",
            }.get(cls, ""),
        }
        for cls in EUROSAT_CLASSES
    ])
    st.dataframe(class_df, use_container_width=True, hide_index=True)

    # ── Algorithmic contributions ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Algorithmic Contributions")
    st.markdown("""
    1. **Multi-scale SE fusion** — Unlike standard SE-Net which applies SE at a single scale,
       we tap 4 intermediate MobileNetV2 layers and apply independent SE blocks, then
       concatenate the globally-pooled outputs. This captures both texture (early layers)
       and semantics (deep layers).

    2. **Curriculum-aware data generator** — A custom `tf.keras.utils.Sequence` subclass
       (`CurriculumDataGenerator`) supports dynamic tier switching between phases without
       reloading data, enabling seamless curriculum transitions.

    3. **Agentic routing** — The router is "agentic" in the sense that it autonomously
       decides which specialist to invoke per sample, rather than using a fixed ensemble.
       This is inspired by mixture-of-experts and LLM tool-use patterns.

    4. **KNN on SE embeddings** — The KNN fallback operates in the learned embedding space
       (256-d, L2-normalised), not raw pixel space. This makes it far more discriminative
       than pixel-level KNN for satellite imagery.
    """)

    # ── Team info ─────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Team & Acknowledgements")
    st.markdown("""
    **Project:** Machine Learning / Deep Learning Coursework  
    **Dataset:** [EuroSAT](https://github.com/phelber/EuroSAT) by Helber et al. (2019)  
    **Framework:** TensorFlow / Keras, Scikit-learn, Streamlit, Plotly  

    > Helber, P., Bischke, B., Dengel, A., & Borth, D. (2019).
    > *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.*
    > IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.
    """)

    # ── Tech stack badges ─────────────────────────────────────────────────────
    st.markdown("""
    <span class="badge badge-blue">TensorFlow 2.x</span>
    <span class="badge badge-green">Scikit-learn</span>
    <span class="badge badge-blue">Streamlit</span>
    <span class="badge badge-orange">Plotly</span>
    <span class="badge badge-green">OpenCV</span>
    <span class="badge badge-blue">NumPy</span>
    <span class="badge badge-orange">Matplotlib</span>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# Router — dispatch to the selected page
# ═════════════════════════════════════════════════════════════════════════════

def main():
    routes = {
        "Home": page_home,
        "Predict": page_predict,
        "Model Comparison": page_model_comparison,
        "Training Progress": page_training_progress,
        "Explainability": page_explainability,
        "Clustering": page_clustering,
        "About": page_about,
    }
    handler = routes.get(page)
    if handler:
        handler()
    else:
        st.error(f"Unknown page: {page}")


if __name__ == "__main__":
    main()

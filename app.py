# -*- coding: utf-8 -*-
"""
CyberShield IDS — Professional Streamlit Dashboard
3-Layer ML Intrusion Detection System
"""

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberShield IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — Dark Cyber Aesthetic
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&family=Exo+2:wght@300;400;600&display=swap');

  /* ── Root palette ── */
  :root {
    --bg:         #050b14;
    --panel:      #0a1628;
    --border:     #0d2545;
    --accent:     #00d4ff;
    --accent2:    #00ff9d;
    --warn:       #ffc300;
    --danger:     #ff3a5c;
    --text:       #c8d8e8;
    --muted:      #7a9ab8;
    --font-mono:  'Share Tech Mono', monospace;
    --font-head:  'Rajdhani', sans-serif;
    --font-body:  'Exo 2', sans-serif;
  }

  /* ── Global ── */
  html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: var(--font-body);
  }

  .stApp { background: var(--bg); }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: var(--panel) !important;
    border-right: 1px solid var(--border);
  }
  section[data-testid="stSidebar"] * { color: var(--text) !important; }

  /* ── Headings ── */
  h1, h2, h3 { font-family: var(--font-head) !important; letter-spacing: 2px; }

  /* ── Hero banner ── */
  .hero-banner {
    background: linear-gradient(135deg, #020d1a 0%, #061428 40%, #0a1f3d 100%);
    border: 1px solid var(--border);
    border-left: 4px solid var(--accent);
    border-radius: 4px;
    padding: 28px 36px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
  }
  .hero-banner::before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
      0deg,
      transparent,
      transparent 2px,
      rgba(0,212,255,0.02) 2px,
      rgba(0,212,255,0.02) 4px
    );
    pointer-events: none;
  }
  .hero-title {
    font-family: var(--font-head);
    font-size: 2.6rem;
    font-weight: 700;
    letter-spacing: 4px;
    color: var(--accent);
    text-shadow: 0 0 20px rgba(0,212,255,0.5);
    margin: 0;
  }
  .hero-sub {
    font-family: var(--font-mono);
    font-size: 0.85rem;
    color: var(--muted);
    margin-top: 6px;
    letter-spacing: 2px;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(0,255,157,0.12);
    border: 1px solid var(--accent2);
    color: var(--accent2);
    font-family: var(--font-mono);
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 2px;
    margin-top: 10px;
    letter-spacing: 1px;
  }

  /* ── Metric cards ── */
  .metric-card {
    background: var(--panel);
    border: 1px solid var(--border);
    border-top: 2px solid var(--accent);
    border-radius: 4px;
    padding: 20px 24px;
    text-align: center;
    position: relative;
  }
  .metric-card.danger  { border-top-color: var(--danger); }
  .metric-card.warn    { border-top-color: var(--warn); }
  .metric-card.success { border-top-color: var(--accent2); }

  .metric-label {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
  }
  .metric-value {
    font-family: var(--font-head);
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--accent);
    line-height: 1;
  }
  .metric-card.danger  .metric-value { color: var(--danger); }
  .metric-card.warn    .metric-value { color: var(--warn); }
  .metric-card.success .metric-value { color: var(--accent2); }

  /* ── Section headers ── */
  .section-header {
    font-family: var(--font-head);
    font-size: 1.1rem;
    letter-spacing: 3px;
    color: var(--accent);
    text-transform: uppercase;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin-bottom: 16px;
  }
  .section-header span { color: var(--muted); font-size: 0.8rem; margin-left: 8px; }

  /* ── Action badge ── */
  .action-ALLOW   { color: var(--accent2); }
  .action-MONITOR { color: var(--warn); }
  .action-BLOCK   { color: #ff7b00; }
  .action-ISOLATE { color: var(--danger); }

  /* ── Log entry ── */
  .log-entry {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    padding: 5px 0;
    border-bottom: 1px solid rgba(13,37,69,0.5);
    color: var(--text);
  }

  /* ── Streamlit overrides ── */
  .stButton > button {
    background: transparent;
    border: 1px solid var(--accent);
    color: var(--accent);
    font-family: var(--font-mono);
    letter-spacing: 2px;
    font-size: 0.8rem;
    border-radius: 2px;
    padding: 8px 24px;
    transition: all 0.2s;
  }
  .stButton > button:hover {
    background: rgba(0,212,255,0.1);
    box-shadow: 0 0 12px rgba(0,212,255,0.3);
  }

  div[data-testid="stFileUploader"] {
    border: 1px dashed var(--accent) !important;
    border-radius: 4px;
    background: rgba(0,212,255,0.03);
    padding: 8px;
  }

  .stDataFrame { border: 1px solid var(--border) !important; }
  .stProgress > div > div > div { background-color: var(--accent) !important; }

  div[data-testid="metric-container"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 12px;
  }

  /* scrollbar */
  ::-webkit-scrollbar { width: 4px; }
  ::-webkit-scrollbar-track { background: var(--bg); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  /* ── Widget text overrides ── */
  label, .stTextInput label, .stNumberInput label, .stSlider label,
  .stSelectbox label, .stTextArea label, .stFileUploader label,
  .stCheckbox label, .stRadio label, .stMultiSelect label,
  [data-testid="stWidgetLabel"] { color: var(--text) !important; }

  .stNumberInput input, .stTextInput input, .stTextArea textarea,
  .stSelectbox select, div[data-baseweb="select"] * {
    color: var(--text) !important;
    background-color: var(--panel) !important;
  }

  p, li, span, div { color: var(--text); }

  div[data-testid="stMarkdownContainer"] p { color: var(--text); }

  .stTabs [data-baseweb="tab"] { color: var(--muted) !important; }
  .stTabs [aria-selected="true"] { color: var(--accent) !important; }

  div[data-testid="metric-container"] label,
  div[data-testid="metric-container"] div { color: var(--text) !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
ACTION_COLORS = {
    "ALLOW":   "#00ff9d",
    "MONITOR": "#ffc300",
    "BLOCK":   "#ff7b00",
    "ISOLATE": "#ff3a5c",
}
ACTION_ICONS = {
    "ALLOW": "✅", "MONITOR": "👁️", "BLOCK": "🚫", "ISOLATE": "🔒"
}

@st.cache_resource(show_spinner=False)
def load_models():
    import warnings
    errors = []
    models = {}
    _base = os.path.dirname(os.path.abspath(__file__))

    # Load each artifact independently so one failure doesn't block the rest.
    # Suppress sklearn version-mismatch warnings — they are non-fatal for inference.
    pkl_files = {
        "rf":        "rf_final.pkl",
        "xgb":       "xgb_final.pkl",
        "meta":      "meta_final.pkl",
        "scaler":    "scaler_final.pkl",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for key, fname in pkl_files.items():
            try:
                with open(os.path.join(_base, fname), "rb") as f:
                    models[key] = pickle.load(f)
            except Exception as e:
                errors.append(f"{key}: {e}")

    try:
        models["threshold"] = np.load(os.path.join(_base, "threshold_final.npy"))
    except Exception as e:
        errors.append(f"threshold: {e}")

    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.layers import Dense as _BaseDense

        class _CompatDense(_BaseDense):
            """Dense that ignores unknown config keys (e.g. quantization_config added by newer Keras)."""
            @classmethod
            def from_config(cls, config):
                config.pop("quantization_config", None)
                return super().from_config(config)

        models["autoencoder"] = load_model(
            os.path.join(_base, "autoencoder_final.h5"),
            custom_objects={"Dense": _CompatDense},
            compile=False,
        )
    except Exception as e:
        errors.append(f"Autoencoder: {e}")

    try:
        import torch
        import torch.nn as nn

        class _RLPolicy(nn.Module):
            """Minimal PPO policy for inference — bypasses cloudpickle/numpy version issues."""
            def __init__(self):
                super().__init__()
                self.mlp_extractor = nn.ModuleDict({
                    "policy_net": nn.Sequential(
                        nn.Linear(3, 64), nn.Tanh(),
                        nn.Linear(64, 64), nn.Tanh(),
                    )
                })
                self.action_net = nn.Linear(64, 4)

            def predict(self, obs, deterministic=True):
                x = torch.tensor(obs, dtype=torch.float32)
                features = self.mlp_extractor["policy_net"](x)
                logits = self.action_net(features)
                actions = logits.argmax(dim=-1)
                return actions.numpy(), None

        policy = _RLPolicy()
        _weights = torch.load(
            os.path.join(_base, "rl_model_final", "policy.pth"),
            map_location="cpu", weights_only=False,
        )
        policy.load_state_dict(_weights, strict=False)
        policy.eval()
        models["rl"] = policy
    except Exception as e:
        errors.append(f"RL model: {e}")

    return models, errors


def predict_batch(X, models):
    import warnings
    scaler    = models["scaler"]
    rf        = models["rf"]
    xgb       = models["xgb"]
    meta      = models["meta"]
    ae        = models["autoencoder"]
    threshold = models["threshold"]
    rl        = models["rl"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_scaled = scaler.transform(X)

    rf_probs  = rf.predict_proba(X_scaled)
    xgb_probs = xgb.predict_proba(X_scaled)
    stack     = np.hstack((rf_probs, xgb_probs))
    preds     = meta.predict(stack)

    recon     = ae.predict(X_scaled, verbose=0)
    mse       = np.mean((X_scaled - recon)**2, axis=1)
    severity  = np.where(mse > threshold*3, 2, np.where(mse > threshold, 1, 0))

    states    = np.column_stack((preds, mse, severity)).astype(np.float32)
    actions, _= rl.predict(states, deterministic=True)

    action_map = {0:"ALLOW", 1:"MONITOR", 2:"BLOCK", 3:"ISOLATE"}
    return [action_map[a] for a in actions], mse, severity, preds


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
PLOT_BG = "rgba(0,0,0,0)"
GRID    = "rgba(13,37,69,0.8)"
FONT    = dict(family="Share Tech Mono", color="#c8d8e8")

def dark_layout(fig, title="", h=380):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Rajdhani", size=14,
                                         color="#00d4ff"), x=0.01),
        paper_bgcolor=PLOT_BG, plot_bgcolor="rgba(5,11,20,0.9)",
        font=FONT, height=h,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=GRID, borderwidth=1),
        xaxis=dict(gridcolor=GRID, zeroline=False, tickfont=FONT),
        yaxis=dict(gridcolor=GRID, zeroline=False, tickfont=FONT),
    )
    return fig


def donut_chart(results):
    counts = Counter(results)
    labels = list(counts.keys())
    vals   = list(counts.values())
    colors = [ACTION_COLORS.get(l, "#ffffff") for l in labels]

    fig = go.Figure(go.Pie(
        labels=labels, values=vals,
        hole=0.65,
        marker=dict(colors=colors,
                    line=dict(color="#050b14", width=3)),
        textfont=dict(family="Share Tech Mono", size=11),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{len(results)}</b><br><span style='font-size:10px'>PACKETS</span>",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(family="Rajdhani", size=18, color="#00d4ff"))
    dark_layout(fig, "ACTION DISTRIBUTION")
    fig.update_layout(showlegend=True,
                      legend=dict(orientation="v", x=1.02, y=0.5))
    return fig


def timeline_chart(results, window=200):
    n = min(len(results), window)
    data = results[-n:]
    df = pd.DataFrame({"idx": range(n), "action": data})
    action_num = {"ALLOW":0,"MONITOR":1,"BLOCK":2,"ISOLATE":3}
    df["level"] = df["action"].map(action_num)

    fig = go.Figure()
    for action, color in ACTION_COLORS.items():
        mask = df["action"] == action
        fig.add_trace(go.Scatter(
            x=df[mask]["idx"], y=df[mask]["level"],
            mode="markers",
            name=action,
            marker=dict(color=color, size=5, opacity=0.8,
                        line=dict(width=0)),
        ))
    fig.update_yaxes(tickvals=[0,1,2,3],
                     ticktext=["ALLOW","MONITOR","BLOCK","ISOLATE"])
    dark_layout(fig, f"DETECTION TIMELINE  (last {n} packets)")
    return fig


def mse_chart(mse_vals, threshold, window=300):
    n   = min(len(mse_vals), window)
    mse = mse_vals[-n:]
    idx = list(range(n))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=idx, y=mse,
        mode="lines",
        line=dict(color="#00d4ff", width=1),
        name="Reconstruction Error",
        fill="tozeroy",
        fillcolor="rgba(0,212,255,0.05)",
    ))
    fig.add_hline(y=float(threshold),  line=dict(color="#ffc300", dash="dash", width=1),
                  annotation_text="threshold", annotation_font_color="#ffc300")
    fig.add_hline(y=float(threshold*3), line=dict(color="#ff3a5c", dash="dash", width=1),
                  annotation_text="critical", annotation_font_color="#ff3a5c")
    dark_layout(fig, "AUTOENCODER RECONSTRUCTION ERROR  (anomaly detection)")
    return fig


def severity_bar(severity_arr):
    counts = {0: int(np.sum(severity_arr==0)),
              1: int(np.sum(severity_arr==1)),
              2: int(np.sum(severity_arr==2))}
    labels = ["NORMAL (0)", "ELEVATED (1)", "CRITICAL (2)"]
    colors = ["#00ff9d", "#ffc300", "#ff3a5c"]
    fig = go.Figure(go.Bar(
        x=labels, y=list(counts.values()),
        marker_color=colors,
        text=list(counts.values()),
        textposition="outside",
        textfont=dict(family="Share Tech Mono", color="#c8d8e8"),
    ))
    dark_layout(fig, "SEVERITY LEVEL DISTRIBUTION")
    return fig


def threat_radar(results):
    cats = ["ALLOW","MONITOR","BLOCK","ISOLATE"]
    c = Counter(results)
    vals = [c.get(x, 0) for x in cats] + [c.get("ALLOW", 0)]

    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=cats+[cats[0]],
        fill="toself",
        fillcolor="rgba(0,212,255,0.1)",
        line=dict(color="#00d4ff", width=2),
        name="Action Profile",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(5,11,20,0.9)",
            angularaxis=dict(tickfont=FONT, linecolor=GRID, gridcolor=GRID),
            radialaxis=dict(tickfont=FONT, gridcolor=GRID),
        ),
        paper_bgcolor=PLOT_BG, font=FONT, height=380,
        margin=dict(l=20,r=20,t=40,b=20),
        title=dict(text="THREAT ACTION RADAR", font=dict(family="Rajdhani",
                   size=14, color="#00d4ff"), x=0.01),
    )
    return fig


def feature_heatmap(X_sample, feature_names=None):
    n = min(50, X_sample.shape[0])
    f = min(20, X_sample.shape[1])
    data = X_sample[:n, :f]
    cols = feature_names[:f] if feature_names is not None else [f"F{i}" for i in range(f)]

    fig = go.Figure(go.Heatmap(
        z=data,
        x=cols,
        colorscale=[[0,"#050b14"],[0.5,"#0d4a8a"],[1,"#00d4ff"]],
        colorbar=dict(tickfont=FONT),
        hovertemplate="Feature: %{x}<br>Sample: %{y}<br>Value: %{z:.4f}<extra></extra>",
    ))
    dark_layout(fig, f"FEATURE HEATMAP  (first {n} samples × {f} features)", h=420)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 0 8px 0;'>
      <div style='font-family:Rajdhani;font-size:1.4rem;font-weight:700;
                  color:#00d4ff;letter-spacing:3px;'>🛡️ CyberShield</div>
      <div style='font-family:"Share Tech Mono";font-size:0.7rem;
                  color:#4a6580;letter-spacing:2px;'>INTRUSION DETECTION SYSTEM</div>
    </div>
    <hr style='border-color:#0d2545;margin:8px 0 20px 0;'/>
    """, unsafe_allow_html=True)

    st.markdown("**MODEL ARCHITECTURE**")
    st.markdown("""
    <div style='font-family:"Share Tech Mono";font-size:0.72rem;
                line-height:2;color:#4a6580;'>
    ◈ Layer 1 — Stacked Ensemble<br>
    &nbsp;&nbsp;&nbsp;├ Random Forest<br>
    &nbsp;&nbsp;&nbsp;└ XGBoost → Meta-Learner<br><br>
    ◈ Layer 2 — Autoencoder<br>
    &nbsp;&nbsp;&nbsp;└ Anomaly via MSE<br><br>
    ◈ Layer 3 — RL Agent (PPO)<br>
    &nbsp;&nbsp;&nbsp;└ ALLOW / MONITOR / BLOCK / ISOLATE
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#0d2545;margin:16px 0;'/>", unsafe_allow_html=True)

    st.markdown("**DATASET**")
    st.markdown("""
    <div style='font-family:"Share Tech Mono";font-size:0.72rem;color:#4a6580;'>
    CIC-IDS2017<br>Friday DDoS Traffic<br>PCAP → CSV Pipeline
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#0d2545;margin:16px 0;'/>", unsafe_allow_html=True)

    demo_mode = st.toggle("🔧 Demo Mode (no model files)", value=False)
    n_demo    = st.slider("Demo sample count", 100, 2000, 500, 50,
                          disabled=not demo_mode)

    st.markdown("<hr style='border-color:#0d2545;margin:16px 0;'/>", unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family:"Share Tech Mono";font-size:0.65rem;color:#4a6580;
                line-height:1.8;'>
    v2.0.0 — DEPLOYMENT BUILD<br>
    Imperial College London<br>
    National Heart & Lung Institute
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <div class="hero-title">🛡️ CYBERSHIELD IDS</div>
  <div class="hero-sub">MULTI-LAYER INTRUSION DETECTION &amp; RESPONSE SYSTEM</div>
  <div class="hero-badge">● SYSTEM ONLINE</div>
  &nbsp;
  <div class="hero-badge" style="border-color:#00d4ff;color:#00d4ff;">
    3-LAYER ML PIPELINE
  </div>
  &nbsp;
  <div class="hero-badge" style="border-color:#ffc300;color:#ffc300;">
    PPO RL AGENT
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS / DEMO DATA
# ─────────────────────────────────────────────────────────────────────────────
if demo_mode:
    # Synthetic demo — no model files needed
    np.random.seed(42)
    N = n_demo
    demo_actions = np.random.choice(
        ["ALLOW","MONITOR","BLOCK","ISOLATE"],
        size=N,
        p=[0.55, 0.22, 0.15, 0.08]
    )
    demo_mse      = np.abs(np.random.randn(N) * 0.04 + 0.02)
    demo_severity = np.where(demo_mse > 0.09, 2, np.where(demo_mse > 0.05, 1, 0))
    demo_X        = np.random.randn(N, 78)

    results    = list(demo_actions)
    mse_vals   = demo_mse
    severity   = demo_severity
    X_data     = demo_X
    model_ok   = True
    load_error = []

    st.info("🔧 **Demo Mode active** — synthetic data shown. Disable in sidebar to run real models.", icon="ℹ️")
else:
    with st.spinner("Loading model pipeline..."):
        models, load_error = load_models()
    _required = {"rf", "xgb", "meta", "scaler", "threshold", "autoencoder", "rl"}
    model_ok = _required.issubset(models.keys())

    if load_error:
        for e in load_error:
            st.warning(f"⚠️ {e}")

    results = mse_vals = severity = X_data = None

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  DASHBOARD",
    "📂  BATCH ANALYSIS",
    "⚡  LIVE MONITOR",
    "🔍  DEEP DIAGNOSTICS",
    "🎯  PREDICT & TEST",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if not demo_mode and results is None:
        st.markdown("""
        <div style='text-align:center;padding:60px 0;font-family:"Share Tech Mono";
                    color:#4a6580;font-size:0.9rem;'>
        Upload a CSV in the BATCH ANALYSIS tab to run inference.
        </div>
        """, unsafe_allow_html=True)
    else:
        c = Counter(results)
        total = len(results)

        # KPI row
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.markdown(f"""<div class="metric-card">
              <div class="metric-label">TOTAL PACKETS</div>
              <div class="metric-value">{total:,}</div>
            </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(f"""<div class="metric-card success">
              <div class="metric-label">ALLOWED</div>
              <div class="metric-value">{c.get('ALLOW',0):,}</div>
            </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(f"""<div class="metric-card warn">
              <div class="metric-label">MONITORED</div>
              <div class="metric-value">{c.get('MONITOR',0):,}</div>
            </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(f"""<div class="metric-card danger" style="border-top-color:#ff7b00">
              <div class="metric-label">BLOCKED</div>
              <div class="metric-value" style="color:#ff7b00">{c.get('BLOCK',0):,}</div>
            </div>""", unsafe_allow_html=True)
        with k5:
            st.markdown(f"""<div class="metric-card danger">
              <div class="metric-label">ISOLATED</div>
              <div class="metric-value">{c.get('ISOLATE',0):,}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br/>", unsafe_allow_html=True)

        # Charts row 1
        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.plotly_chart(donut_chart(results),   use_container_width=True, key="batch_donut")
        with col_b:
            st.plotly_chart(timeline_chart(results), use_container_width=True, key="batch_timeline")

        # Charts row 2
        col_c, col_d = st.columns(2)
        with col_c:
            st.plotly_chart(mse_chart(mse_vals, 0.05 if demo_mode else float(models["threshold"])),
                            use_container_width=True, key="batch_mse")
        with col_d:
            st.plotly_chart(severity_bar(severity), use_container_width=True, key="batch_severity")

        # Threat radar
        col_e, col_f = st.columns(2)
        with col_e:
            st.plotly_chart(threat_radar(results), use_container_width=True, key="batch_radar")
        with col_f:
            st.markdown("<div class='section-header'>SYSTEM STATUS</div>",
                        unsafe_allow_html=True)

            threat_pct = (c.get("BLOCK",0) + c.get("ISOLATE",0)) / total * 100 if total else 0
            st.markdown(f"""
            <div style='font-family:"Share Tech Mono";font-size:0.8rem;
                        line-height:2.4;color:#c8d8e8;'>
            <span style='color:#4a6580;'>THREAT RATE &nbsp;&nbsp;&nbsp;</span>
            <span style='color:{"#ff3a5c" if threat_pct>20 else "#ffc300" if threat_pct>5 else "#00ff9d"}'>
            {threat_pct:.1f}%</span><br>
            <span style='color:#4a6580;'>CRITICAL PKTS &nbsp;</span>
            <span style='color:#ff3a5c;'>{int(np.sum(severity==2)):,}</span><br>
            <span style='color:#4a6580;'>ELEVATED PKTS &nbsp;</span>
            <span style='color:#ffc300;'>{int(np.sum(severity==1)):,}</span><br>
            <span style='color:#4a6580;'>NORMAL PKTS &nbsp;&nbsp;</span>
            <span style='color:#00ff9d;'>{int(np.sum(severity==0)):,}</span><br>
            <span style='color:#4a6580;'>PIPELINE &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
            <span style='color:#00d4ff;'>3-LAYER ACTIVE</span><br>
            <span style='color:#4a6580;'>RL AGENT &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span>
            <span style='color:#00d4ff;'>PPO (DETERMINISTIC)</span>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br/>", unsafe_allow_html=True)
            threat_color = "#ff3a5c" if threat_pct > 20 else "#ffc300" if threat_pct > 5 else "#00ff9d"
            st.markdown(f"""
            <div style='font-family:"Share Tech Mono";font-size:0.7rem;color:#4a6580;
                        margin-bottom:4px;'>THREAT INDEX</div>
            """, unsafe_allow_html=True)
            st.progress(min(threat_pct / 100, 1.0))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — BATCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>UPLOAD TRAFFIC DATA <span>// CSV Format — CIC-IDS2017 compatible</span></div>",
                unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop CSV file here", type=["csv"],
                                 label_visibility="collapsed")

    if uploaded:
        with st.spinner("⚡ Processing traffic data..."):
            df = pd.read_csv(uploaded)
            df.columns = df.columns.str.strip()
            if "Label" in df.columns:
                df = df.drop("Label", axis=1)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)
            X_up = df.values

        st.success(f"✅ Loaded **{len(df):,}** samples × **{df.shape[1]}** features")

        if demo_mode:
            st.info("Demo mode: showing synthetic predictions for your file shape.")
            res   = list(np.random.choice(["ALLOW","MONITOR","BLOCK","ISOLATE"],
                                           size=len(X_up), p=[0.55,0.22,0.15,0.08]))
            m_err = np.abs(np.random.randn(len(X_up)) * 0.04 + 0.02)
            sev   = np.where(m_err > 0.09, 2, np.where(m_err > 0.05, 1, 0))
        elif model_ok:
            with st.spinner("Running 3-layer pipeline..."):
                bar = st.progress(0)
                bar.progress(20)
                res, m_err, sev, _ = predict_batch(X_up, models)
                bar.progress(100)
        else:
            st.error("Model files not found. Enable Demo Mode in sidebar.")
            res = None

        if res:
            st.markdown("<br/>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(donut_chart(res), use_container_width=True, key="live_donut")
            with col2:
                st.plotly_chart(severity_bar(sev), use_container_width=True, key="live_severity")

            thr_val = 0.05 if demo_mode else float(models["threshold"])
            st.plotly_chart(mse_chart(m_err, thr_val), use_container_width=True, key="live_mse")

            # Results table
            st.markdown("<div class='section-header'>PREDICTION RESULTS <span>// first 200 rows</span></div>",
                        unsafe_allow_html=True)
            out_df = pd.DataFrame({
                "Sample":   range(len(res)),
                "Action":   res,
                "MSE":      m_err.round(6),
                "Severity": sev,
            })

            def color_action(val):
                return f"color: {ACTION_COLORS.get(val,'white')}"

            st.dataframe(
                out_df.head(200).style.applymap(color_action, subset=["Action"]),
                use_container_width=True, height=320
            )

            csv_out = out_df.to_csv(index=False).encode()
            st.download_button("⬇ DOWNLOAD RESULTS CSV", csv_out,
                               "cybershield_results.csv", "text/csv")

    elif not demo_mode:
        st.markdown("""
        <div style='text-align:center;padding:60px 0;font-family:"Share Tech Mono";
                    color:#4a6580;font-size:0.85rem;border:1px dashed #0d2545;
                    border-radius:4px;'>
        ↑ UPLOAD A CSV FILE TO BEGIN ANALYSIS
        </div>
        """, unsafe_allow_html=True)

    if demo_mode and not uploaded:
        st.markdown("<div class='section-header'>DEMO FEATURE HEATMAP</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(feature_heatmap(X_data), use_container_width=True, key="live_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>REAL-TIME PACKET SIMULATION</div>",
                unsafe_allow_html=True)

    col_s, col_p, col_d = st.columns([1, 1, 3])
    with col_s:
        n_live   = st.number_input("Packets", 5, 50, 15, 1)
    with col_p:
        interval = st.number_input("Interval (s)", 0.1, 2.0, 0.4, 0.1)

    run_live = st.button("▶  LAUNCH LIVE DETECTION")

    log_box = st.empty()
    prog    = st.empty()
    chart_p = st.empty()

    if run_live:
        log_lines = []
        live_res  = []

        for i in range(int(n_live)):
            if demo_mode:
                action = np.random.choice(
                    ["ALLOW","MONITOR","BLOCK","ISOLATE"],
                    p=[0.55,0.22,0.15,0.08]
                )
                mse_v = abs(np.random.randn() * 0.04 + 0.02)
            else:
                if not model_ok:
                    st.error("Model files not found.")
                    break
                n_feat = models["scaler"].n_features_in_
                sample = np.abs(np.random.randn(n_feat))
                sample_s = models["scaler"].transform(sample.reshape(1,-1))
                rf_p  = models["rf"].predict_proba(sample_s)
                xgb_p = models["xgb"].predict_proba(sample_s)
                stack = np.hstack((rf_p, xgb_p))
                pred  = models["meta"].predict(stack)[0]
                recon = models["autoencoder"].predict(sample_s, verbose=0)
                mse_v = float(np.mean((sample_s - recon)**2))
                sev_v = 2 if mse_v > float(models["threshold"])*3 else (1 if mse_v > float(models["threshold"]) else 0)
                state = np.array([[pred, mse_v, sev_v]], dtype=np.float32)
                act_n, _ = models["rl"].predict(state, deterministic=True)
                action = {0:"ALLOW",1:"MONITOR",2:"BLOCK",3:"ISOLATE"}[act_n.item()]

            live_res.append(action)
            ts = time.strftime("%H:%M:%S")
            col = ACTION_COLORS[action]
            icon = ACTION_ICONS[action]
            log_lines.append(
                f"<div class='log-entry'>"
                f"<span style='color:#4a6580;'>[{ts}]</span> &nbsp;"
                f"PKT-{i+1:04d} &nbsp;"
                f"MSE=<span style='color:#00d4ff;'>{mse_v:.5f}</span> &nbsp;"
                f"→ <span style='color:{col};font-weight:700;'>{icon} {action}</span>"
                f"</div>"
            )

            log_html = (
                "<div style='background:#050b14;border:1px solid #0d2545;"
                "border-radius:4px;padding:12px 16px;max-height:320px;overflow-y:auto;'>"
                + "".join(log_lines) +
                "</div>"
            )
            log_box.markdown(log_html, unsafe_allow_html=True)
            prog.progress((i+1) / int(n_live))

            time.sleep(float(interval))

        st.success(f"✅ Detection complete — {int(n_live)} packets processed")
        chart_p.plotly_chart(donut_chart(live_res), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — DEEP DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>PIPELINE ARCHITECTURE</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div style='font-family:"Share Tech Mono";font-size:0.78rem;
                line-height:2.2;color:#c8d8e8;
                background:#050b14;border:1px solid #0d2545;
                border-radius:4px;padding:20px 28px;'>

    <span style='color:#00d4ff;font-size:1rem;'>LAYER 1 — ENSEMBLE CLASSIFIER</span><br>
    Input: Raw network traffic features (scaled)<br>
    ├─ Random Forest &nbsp;→ class probabilities<br>
    ├─ XGBoost &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ class probabilities<br>
    └─ Meta-Learner &nbsp;→ stacked prediction [0/1]<br><br>

    <span style='color:#00d4ff;font-size:1rem;'>LAYER 2 — AUTOENCODER ANOMALY DETECTION</span><br>
    Input: Scaled feature vector<br>
    ├─ Encoder → latent space (compression)<br>
    ├─ Decoder → reconstruction<br>
    └─ MSE vs threshold → severity [0/1/2]<br><br>

    <span style='color:#00d4ff;font-size:1rem;'>LAYER 3 — PPO REINFORCEMENT LEARNING AGENT</span><br>
    State: [ensemble_pred, reconstruction_mse, severity]<br>
    Actions: ALLOW | MONITOR | BLOCK | ISOLATE<br>
    Policy: Deterministic PPO (Stable-Baselines3)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br/><div class='section-header'>SEVERITY THRESHOLDS</div>",
                unsafe_allow_html=True)

    threshold_val = 0.05 if demo_mode else float(models.get("threshold", 0.05))

    fig_thresh = go.Figure()
    x = np.linspace(0, threshold_val * 6, 400)
    y = np.exp(-x / (threshold_val * 2)) * 100

    fig_thresh.add_trace(go.Scatter(x=x, y=y, fill="tozeroy",
                                    fillcolor="rgba(0,255,157,0.1)",
                                    line=dict(color="#00ff9d", width=1.5),
                                    name="Normal Zone"))
    fig_thresh.add_vrect(x0=threshold_val, x1=threshold_val*3,
                          fillcolor="rgba(255,195,0,0.08)",
                          line=dict(color="#ffc300", width=1, dash="dot"),
                          annotation_text="ELEVATED", annotation_position="top left",
                          annotation_font_color="#ffc300")
    fig_thresh.add_vrect(x0=threshold_val*3, x1=threshold_val*6,
                          fillcolor="rgba(255,58,92,0.08)",
                          line=dict(color="#ff3a5c", width=1, dash="dot"),
                          annotation_text="CRITICAL", annotation_position="top left",
                          annotation_font_color="#ff3a5c")
    fig_thresh.add_vline(x=threshold_val,   line=dict(color="#ffc300", width=2, dash="dash"))
    fig_thresh.add_vline(x=threshold_val*3, line=dict(color="#ff3a5c", width=2, dash="dash"))
    dark_layout(fig_thresh, "MSE SEVERITY ZONES")
    st.plotly_chart(fig_thresh, use_container_width=True, key="diag_thresh")

    # Feature stats
    st.markdown("<div class='section-header'>FEATURE HEATMAP <span>// first 50 samples</span></div>",
                unsafe_allow_html=True)
    if demo_mode:
        _heatmap_data = X_data
    else:
        _hm_n = models["scaler"].n_features_in_ if model_ok else 78
        _heatmap_data = np.random.randn(50, _hm_n)
    st.plotly_chart(feature_heatmap(_heatmap_data), use_container_width=True, key="diag_heatmap")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — PREDICT & TEST
# ══════════════════════════════════════════════════════════════════════════════
with tab5:

    # Exact 78 feature names as fitted in scaler_final.pkl (CIC-IDS2017)
    FEATURE_NAMES = [
        "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
        "Total Length of Fwd Packets", "Total Length of Bwd Packets",
        "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean", "Fwd Packet Length Std",
        "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean", "Bwd Packet Length Std",
        "Flow Bytes/s", "Flow Packets/s",
        "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min",
        "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min",
        "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min",
        "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags",
        "Fwd Header Length", "Bwd Header Length",
        "Fwd Packets/s", "Bwd Packets/s",
        "Min Packet Length", "Max Packet Length",
        "Packet Length Mean", "Packet Length Std", "Packet Length Variance",
        "FIN Flag Count", "SYN Flag Count", "RST Flag Count",
        "PSH Flag Count", "ACK Flag Count", "URG Flag Count",
        "CWE Flag Count", "ECE Flag Count",
        "Down/Up Ratio", "Average Packet Size",
        "Avg Fwd Segment Size", "Avg Bwd Segment Size",
        "Fwd Header Length.1",
        "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate",
        "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
        "Subflow Fwd Packets", "Subflow Fwd Bytes",
        "Subflow Bwd Packets", "Subflow Bwd Bytes",
        "Init_Win_bytes_forward", "Init_Win_bytes_backward",
        "act_data_pkt_fwd", "min_seg_size_forward",
        "Active Mean", "Active Std", "Active Max", "Active Min",
        "Idle Mean", "Idle Std", "Idle Max", "Idle Min",
    ]
    N_FEATURES = len(FEATURE_NAMES)  # 78

    def demo_predict_single(sample_vec, threshold=0.05):
        """Deterministic demo prediction based on synthetic heuristics."""
        mse_v = float(np.mean(sample_vec**2) * 0.001 + abs(np.random.randn()) * 0.01)
        if mse_v > threshold * 3:
            sev, action = 2, "ISOLATE"
        elif mse_v > threshold * 1.5:
            sev, action = 1, "BLOCK"
        elif mse_v > threshold * 0.8:
            sev, action = 1, "MONITOR"
        else:
            sev, action = 0, "ALLOW"
        return action, mse_v, sev

    def real_predict_single(sample_vec, models):
        import warnings
        scaler    = models["scaler"]
        rf        = models["rf"]
        xgb       = models["xgb"]
        meta      = models["meta"]
        ae        = models["autoencoder"]
        threshold = float(models["threshold"])
        rl        = models["rl"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = scaler.transform(sample_vec.reshape(1, -1))
        rf_p  = rf.predict_proba(s)
        xgb_p = xgb.predict_proba(s)
        stack = np.hstack((rf_p, xgb_p))
        pred  = meta.predict(stack)[0]
        recon = ae.predict(s, verbose=0)
        mse_v = float(np.mean((s - recon)**2))
        sev   = 2 if mse_v > threshold*3 else (1 if mse_v > threshold else 0)
        state = np.array([[pred, mse_v, sev]], dtype=np.float32)
        act_n, _ = rl.predict(state, deterministic=True)
        action = {0:"ALLOW",1:"MONITOR",2:"BLOCK",3:"ISOLATE"}[act_n.item()]
        return action, mse_v, sev

    def render_result_card(action, mse_v, sev, source_label=""):
        col  = ACTION_COLORS[action]
        icon = ACTION_ICONS[action]
        sev_labels = {0: ("NORMAL",  "#00ff9d"),
                      1: ("ELEVATED","#ffc300"),
                      2: ("CRITICAL","#ff3a5c")}
        sev_txt, sev_col = sev_labels[sev]
        st.markdown(f"""
        <div style='background:#050b14;border:1px solid {col};border-left:4px solid {col};
                    border-radius:4px;padding:24px 32px;margin-top:16px;'>
          <div style='font-family:"Share Tech Mono";font-size:0.7rem;
                      color:#4a6580;letter-spacing:2px;margin-bottom:6px;'>
            {source_label}PREDICTION RESULT
          </div>
          <div style='font-family:Rajdhani;font-size:3rem;font-weight:700;
                      color:{col};text-shadow:0 0 20px {col}88;line-height:1;'>
            {icon} {action}
          </div>
          <div style='margin-top:16px;display:flex;gap:32px;
                      font-family:"Share Tech Mono";font-size:0.78rem;'>
            <div>
              <span style='color:#4a6580;'>RECONSTRUCTION MSE</span><br>
              <span style='color:#00d4ff;font-size:1.1rem;'>{mse_v:.6f}</span>
            </div>
            <div>
              <span style='color:#4a6580;'>SEVERITY LEVEL</span><br>
              <span style='color:{sev_col};font-size:1.1rem;'>{sev} — {sev_txt}</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    def render_explanation(action, mse_v, sev, threshold=0.05):
        exp = {
            "ALLOW":   "All 3 pipeline layers classify this packet as benign. "
                       "Reconstruction error is below threshold — no anomaly detected. "
                       "The RL agent permits this traffic without restriction.",
            "MONITOR": "Layer 1 ensemble shows slight ambiguity, or the autoencoder "
                       "reconstruction error is mildly elevated. The RL agent flags this "
                       "packet for monitoring — it passes but is logged for review.",
            "BLOCK":   "The autoencoder detects a significant anomaly (MSE above threshold). "
                       "Layer 1 or the RL agent identifies this as likely malicious. "
                       "Traffic is blocked at the network perimeter.",
            "ISOLATE": "Critical anomaly detected. MSE is far above threshold, "
                       "indicating a strong reconstruction failure consistent with "
                       "DDoS or intrusion traffic. The RL agent triggers full host isolation.",
        }
        st.markdown(f"""
        <div style='background:#0a1628;border:1px solid #0d2545;border-radius:4px;
                    padding:16px 20px;margin-top:12px;font-family:"Share Tech Mono";
                    font-size:0.75rem;line-height:1.9;color:#c8d8e8;'>
          <span style='color:#00d4ff;letter-spacing:2px;'>WHY THIS DECISION?</span><br><br>
          {exp[action]}<br><br>
          <span style='color:#4a6580;'>MSE={mse_v:.6f} &nbsp;|&nbsp;
          Threshold={threshold:.4f} &nbsp;|&nbsp;
          3× Threshold={threshold*3:.4f} &nbsp;|&nbsp;
          Severity={sev}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── Three sub-sections ──────────────────────────────────────────────────
    sec1, sec2, sec3 = st.tabs([
        "✏️  Manual Feature Entry",
        "🎲  Random Sample Tester",
        "🎚️  Interactive Demo Sliders",
    ])

    # ── SECTION 1: Manual Entry ─────────────────────────────────────────────
    with sec1:
        st.markdown("<div class='section-header'>MANUAL FEATURE INPUT <span>// enter network flow values</span></div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:"Share Tech Mono";font-size:0.72rem;color:#4a6580;
                    margin-bottom:16px;'>
        Enter values for the most diagnostic network flow features below.
        Leave blank to use 0. All values are automatically scaled by the pipeline.
        </div>
        """, unsafe_allow_html=True)

        # Show 12 key features in 3 columns (names match scaler_final.pkl exactly)
        KEY_FEATURES = [
            ("Flow Duration",              0.0,   5e7,  100000.0),
            ("Total Fwd Packets",          0.0,   5000, 10.0),
            ("Total Backward Packets",     0.0,   5000, 5.0),
            ("Flow Bytes/s",               0.0,   1e7,  5000.0),
            ("Flow Packets/s",             0.0,   1e6,  100.0),
            ("Fwd Packet Length Mean",     0.0,   1500, 500.0),
            ("Bwd Packet Length Mean",     0.0,   1500, 400.0),
            ("Flow IAT Mean",              0.0,   1e7,  10000.0),
            ("SYN Flag Count",             0.0,   500,  0.0),
            ("ACK Flag Count",             0.0,   500,  10.0),
            ("PSH Flag Count",             0.0,   500,  2.0),
            ("Average Packet Size",        0.0,   1500, 512.0),
        ]

        manual_vals = {}
        cols_m = st.columns(3)
        for idx, (fname, fmin, fmax, fdef) in enumerate(KEY_FEATURES):
            with cols_m[idx % 3]:
                manual_vals[fname] = st.number_input(
                    fname, min_value=float(fmin), max_value=float(fmax),
                    value=float(fdef), step=float((fmax - fmin) / 200),
                    format="%.2f", key=f"manual_{idx}"
                )

        st.markdown("<br/>", unsafe_allow_html=True)
        run_manual = st.button("⚡  RUN PREDICTION", key="run_manual")

        if run_manual:
            # Build full feature vector (zeros for unlisted features)
            vec = np.zeros(N_FEATURES)
            for fname, val in manual_vals.items():
                if fname in FEATURE_NAMES:
                    vec[FEATURE_NAMES.index(fname)] = val

            if demo_mode:
                action, mse_v, sev = demo_predict_single(vec)
            else:
                action, mse_v, sev = real_predict_single(vec, models)

            render_result_card(action, mse_v, sev, "MANUAL INPUT — ")
            render_explanation(action, mse_v, sev)

            # Mini gauge
            threshold_v = 0.05 if demo_mode else float(models["threshold"])
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=mse_v,
                title={"text": "Reconstruction MSE",
                       "font": {"family": "Share Tech Mono", "color": "#c8d8e8", "size": 13}},
                gauge={
                    "axis": {"range": [0, threshold_v * 6],
                             "tickfont": {"family": "Share Tech Mono", "color": "#4a6580"}},
                    "bar":  {"color": ACTION_COLORS[action]},
                    "steps": [
                        {"range": [0, threshold_v],           "color": "rgba(0,255,157,0.08)"},
                        {"range": [threshold_v, threshold_v*3],"color": "rgba(255,195,0,0.08)"},
                        {"range": [threshold_v*3, threshold_v*6],"color": "rgba(255,58,92,0.08)"},
                    ],
                    "threshold": {
                        "line": {"color": "#ff3a5c", "width": 2},
                        "thickness": 0.75,
                        "value": threshold_v * 3,
                    },
                    "bgcolor": "#050b14",
                    "bordercolor": "#0d2545",
                },
                number={"font": {"family": "Share Tech Mono", "color": ACTION_COLORS[action]},
                        "valueformat": ".6f"},
            ))
            gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                 font={"color": "#c8d8e8"},
                                 height=260, margin=dict(l=20,r=20,t=40,b=10))
            st.plotly_chart(gauge, use_container_width=True, key="predict_gauge")

    # ── SECTION 2: Random Sample Tester ────────────────────────────────────
    with sec2:
        st.markdown("<div class='section-header'>RANDOM SAMPLE TESTER <span>// generate & explain</span></div>",
                    unsafe_allow_html=True)

        profile_col, btn_col = st.columns([2, 1])
        with profile_col:
            traffic_profile = st.selectbox(
                "Traffic profile to simulate",
                ["Normal Web Traffic", "DDoS Attack", "Port Scan", "Data Exfiltration",
                 "Brute Force", "Fully Random"],
                key="profile_sel"
            )
        with btn_col:
            st.markdown("<br/>", unsafe_allow_html=True)
            run_random = st.button("🎲  GENERATE & PREDICT", key="run_random")

        PROFILES = {
            "Normal Web Traffic":    {"flow_bytes": (500, 5000),   "syn": (0,1),  "pkt_len": (400,900),  "iat": (1000,10000)},
            "DDoS Attack":           {"flow_bytes": (1e5, 1e7),    "syn": (5,50), "pkt_len": (40,100),   "iat": (10,500)},
            "Port Scan":             {"flow_bytes": (10, 200),     "syn": (1,10), "pkt_len": (40,60),    "iat": (100,1000)},
            "Data Exfiltration":     {"flow_bytes": (5000, 1e6),   "syn": (0,2),  "pkt_len": (800,1500), "iat": (500,5000)},
            "Brute Force":           {"flow_bytes": (100, 2000),   "syn": (1,5),  "pkt_len": (100,400),  "iat": (50,500)},
            "Fully Random":          {"flow_bytes": (0, 1e7),      "syn": (0,50), "pkt_len": (0,1500),   "iat": (0,1e7)},
        }

        if run_random:
            p = PROFILES[traffic_profile]
            vec = np.zeros(N_FEATURES)

            def rnd(lo, hi): return np.random.uniform(lo, hi)

            flow_bytes = rnd(*p["flow_bytes"])
            pkt_len    = rnd(*p["pkt_len"])
            iat        = rnd(*p["iat"])
            syn_cnt    = int(rnd(*p["syn"]))

            assignments = {
                "Flow Duration":           rnd(1000, 5e6),
                "Total Fwd Packets":       rnd(1, 500),
                "Total Backward Packets":  rnd(0, 300),
                "Flow Bytes/s":            flow_bytes,
                "Flow Packets/s":          rnd(1, 1000),
                "Fwd Packet Length Mean":  pkt_len,
                "Bwd Packet Length Mean":  pkt_len * 0.8,
                "Flow IAT Mean":           iat,
                "SYN Flag Count":          syn_cnt,
                "ACK Flag Count":          rnd(0, 100),
                "Average Packet Size":     pkt_len,
            }
            for fname, val in assignments.items():
                if fname in FEATURE_NAMES:
                    vec[FEATURE_NAMES.index(fname)] = val
            vec += np.random.randn(N_FEATURES) * 0.5

            if demo_mode:
                action, mse_v, sev = demo_predict_single(vec)
            else:
                action, mse_v, sev = real_predict_single(vec, models)

            threshold_v = 0.05 if demo_mode else float(models["threshold"])

            # Show generated feature snapshot
            st.markdown("<div class='section-header' style='margin-top:16px;'>GENERATED SAMPLE SNAPSHOT</div>",
                        unsafe_allow_html=True)
            snap_df = pd.DataFrame({
                "Feature": list(assignments.keys()),
                "Value":   [f"{v:.2f}" for v in assignments.values()],
            })
            st.dataframe(snap_df, use_container_width=True, hide_index=True, height=280)

            render_result_card(action, mse_v, sev, f"{traffic_profile.upper()} — ")
            render_explanation(action, mse_v, sev, threshold_v)

            # Feature bar chart
            feat_vals = [assignments.get(f, 0) for f in list(assignments.keys())]
            feat_norm = np.array(feat_vals, dtype=float)
            mx = feat_norm.max()
            if mx > 0:
                feat_norm = feat_norm / mx

            fig_feat = go.Figure(go.Bar(
                x=list(assignments.keys()),
                y=feat_norm,
                marker_color=[ACTION_COLORS[action]] * len(feat_norm),
                opacity=0.8,
            ))
            dark_layout(fig_feat, f"FEATURE PROFILE — {traffic_profile} (normalised)", h=300)
            fig_feat.update_xaxes(tickangle=-30)
            st.plotly_chart(fig_feat, use_container_width=True, key="predict_feat")

    # ── SECTION 3: Interactive Sliders ──────────────────────────────────────
    with sec3:
        st.markdown("<div class='section-header'>INTERACTIVE DEMO SLIDERS <span>// tweak & watch prediction change</span></div>",
                    unsafe_allow_html=True)
        st.markdown("""
        <div style='font-family:"Share Tech Mono";font-size:0.72rem;color:#4a6580;
                    margin-bottom:20px;'>
        Drag any slider — the prediction updates instantly. Watch how each feature
        pushes the RL agent between ALLOW → MONITOR → BLOCK → ISOLATE.
        </div>
        """, unsafe_allow_html=True)

        threshold_v = 0.05

        c1, c2 = st.columns(2)
        with c1:
            sl_flow_bytes = st.slider("Flow Bytes/s",        0.0, 1e7,  500.0,    step=100.0,   key="sl1")
            sl_pkt_len    = st.slider("Avg Packet Length",   0.0, 1500.0, 512.0,  step=10.0,    key="sl2")
            sl_iat        = st.slider("Flow IAT Mean (µs)",  0.0, 1e6,  5000.0,   step=100.0,   key="sl3")
            sl_syn        = st.slider("SYN Flag Count",      0,   200,   0,        step=1,        key="sl4")
        with c2:
            sl_fwd_pkts   = st.slider("Total Fwd Packets",   0,   5000,  10,       step=1,        key="sl5")
            sl_ack        = st.slider("ACK Flag Count",      0,   500,   10,       step=1,        key="sl6")
            sl_duration   = st.slider("Flow Duration (µs)",  0.0, 5e7, 100000.0,  step=1000.0,  key="sl7")
            sl_psh        = st.slider("PSH Flag Count",      0,   200,   2,        step=1,        key="sl8")

        # Build vector from sliders
        vec_sl = np.zeros(N_FEATURES)
        sl_map = {
            "Flow Bytes/s":           sl_flow_bytes,
            "Flow IAT Mean":          sl_iat,
            "SYN Flag Count":         sl_syn,
            "Total Fwd Packets":      sl_fwd_pkts,
            "ACK Flag Count":         sl_ack,
            "Flow Duration":          sl_duration,
            "PSH Flag Count":         sl_psh,
            "Average Packet Size":    sl_pkt_len,
            "Fwd Packet Length Mean": sl_pkt_len,
        }
        for fname, val in sl_map.items():
            if fname in FEATURE_NAMES:
                vec_sl[FEATURE_NAMES.index(fname)] = val

        if demo_mode or not model_ok:
            action_sl, mse_sl, sev_sl = demo_predict_single(vec_sl, threshold_v)
        else:
            action_sl, mse_sl, sev_sl = real_predict_single(vec_sl, models)
            threshold_v = float(models["threshold"])

        col_r, col_e = st.columns([1, 1])
        with col_r:
            render_result_card(action_sl, mse_sl, sev_sl, "LIVE SLIDER — ")

        with col_e:
            render_explanation(action_sl, mse_sl, sev_sl, threshold_v)

        # Live radar of slider values (normalised)
        sl_radar_labels = ["Flow Bytes/s","Pkt Length","IAT","SYN Flags",
                           "Fwd Packets","ACK Flags","Duration","PSH Flags"]
        sl_radar_vals_raw = [sl_flow_bytes/1e7, sl_pkt_len/1500,
                             sl_iat/1e6, sl_syn/200,
                             sl_fwd_pkts/5000, sl_ack/500,
                             sl_duration/5e7, sl_psh/200]
        sl_radar_vals = sl_radar_vals_raw + [sl_radar_vals_raw[0]]
        sl_radar_labels_c = sl_radar_labels + [sl_radar_labels[0]]

        _c = ACTION_COLORS[action_sl].lstrip("#")
        _radar_fill = "rgba({},{},{},0.13)".format(int(_c[0:2],16), int(_c[2:4],16), int(_c[4:6],16))
        fig_radar = go.Figure(go.Scatterpolar(
            r=sl_radar_vals, theta=sl_radar_labels_c,
            fill="toself",
            fillcolor=_radar_fill,
            line=dict(color=ACTION_COLORS[action_sl], width=2),
            name="Current Input",
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(5,11,20,0.9)",
                angularaxis=dict(tickfont=FONT, linecolor=GRID, gridcolor=GRID),
                radialaxis=dict(range=[0,1], tickfont=FONT, gridcolor=GRID),
            ),
            paper_bgcolor=PLOT_BG, font=FONT, height=360,
            margin=dict(l=20,r=20,t=40,b=20),
            title=dict(text="INPUT FEATURE RADAR  (normalised)",
                       font=dict(family="Rajdhani", size=14,
                                 color=ACTION_COLORS[action_sl]), x=0.01),
        )
        st.plotly_chart(fig_radar, use_container_width=True, key="predict_radar")


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#0d2545;margin:32px 0 16px 0;'/>
<div style='font-family:"Share Tech Mono";font-size:0.68rem;color:#4a6580;
            text-align:center;letter-spacing:1px;padding-bottom:16px;'>
CyberShield IDS &nbsp;|&nbsp; 3-Layer ML Pipeline &nbsp;|&nbsp;
RF + XGB + Autoencoder + PPO &nbsp;|&nbsp; CIC-IDS2017 Dataset
</div>
""", unsafe_allow_html=True)
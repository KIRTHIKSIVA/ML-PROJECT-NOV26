# ═══════════════════════════════════════════════════════════════════════════════
#  CyberShield AI  —  Network Intrusion Detection System
#  app.py  |  Home Dashboard  +  Live Simulation
# ═══════════════════════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import altair as alt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CyberShield AI",
    page_icon="shield",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ═══ BASE / PAGE ══════════════════════════════════════════════════════════ */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(160deg, #07021a 0%, #040d1c 35%, #060b14 65%, #0c0608 100%);
    background-attachment: fixed;
}
[data-testid="stHeader"]           { background: transparent; }
[data-testid="stSidebarContent"]   { background: #080415; }
.block-container                   { padding-top: 1.4rem; padding-bottom: 3rem; }
h1,h2,h3,h4,h5                     { color: #eeeeff !important; }
p, li, span                        { color: #9898bb; }
hr                                  { border-color: #1c1c3a; }

/* ═══ HERO ═════════════════════════════════════════════════════════════════ */
.hero {
    text-align: center;
    padding: 52px 24px 40px;
    background: linear-gradient(135deg,
        #0e0030 0%, #061840 25%, #03220d 55%, #1a0820 85%, #0e0030 100%);
    border-radius: 20px;
    margin-bottom: 18px;
    border: 1px solid rgba(160, 80, 255, 0.22);
    box-shadow: 0 0 90px rgba(130, 60, 255, 0.07),
                0 0 40px rgba(0, 220, 180, 0.04);
}
.hero h1 {
    font-size: 3.4em; font-weight: 900; margin-bottom: 10px;
    background: linear-gradient(90deg, #00e5ff 0%, #00ff9f 40%, #c77dff 75%, #ff6b9d 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero p { font-size: 1.1em; color: #7070a0 !important; margin: 0 0 18px; }
.hero-tag {
    display: inline-block; margin: 0 5px;
    padding: 4px 14px; border-radius: 20px;
    font-size: 0.76em; font-weight: 700; letter-spacing: .4px;
}

/* ═══ STAT CARDS ══════════════════════════════════════════════════════════ */
.stat-card {
    border-radius: 16px; padding: 26px 14px; text-align: center;
    margin-bottom: 10px;
    background: linear-gradient(145deg, #0d0a1e, #12102a);
    box-shadow: 0 4px 28px rgba(0,0,0,.4);
    transition: box-shadow .2s;
}
.stat-card:hover { box-shadow: 0 6px 36px rgba(100,60,255,.15); }
.stat-icon  { font-size: 1.5em; margin-bottom: 8px; font-style: normal; }
.stat-value { font-size: 2em; font-weight: 900; }
.stat-label { font-size: 0.8em; color: #55558a !important; margin-top: 5px; }

/* ═══ SECTION HEADER ═══════════════════════════════════════════════════════ */
.sec-head {
    font-size: 1.1em; font-weight: 800; color: #ddddff !important;
    margin: 30px 0 14px; display: flex; align-items: center; gap: 10px;
    text-transform: uppercase; letter-spacing: 1px; font-size: .82em;
}
.sec-head::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(160,80,255,.4), transparent);
}

/* ═══ MODEL CARDS ═════════════════════════════════════════════════════════ */
.model-card {
    border-radius: 16px; padding: 22px; height: 100%; margin-top: 10px;
    background: linear-gradient(145deg, #0c0a1e, #111028);
    border: 2px solid;
}
.model-card h4 { font-size: 1.0em; margin-bottom: 16px; }
.mrow {
    display: flex; justify-content: space-between; align-items: center;
    padding: 7px 0; border-bottom: 1px solid rgba(255,255,255,.05);
    font-size: .88em;
}
.mrow span:first-child { color: #555588 !important; }
.mrow .mv { font-weight: 800; }
.best-badge {
    margin-top: 16px; padding: 8px 12px; border-radius: 8px; text-align: center;
    background: rgba(0,255,159,.08); border: 1px solid rgba(0,255,159,.25);
    color: #00ff9f !important; font-size: .76em; font-weight: 800; letter-spacing: .6px;
}
.chart-caption {
    text-align: center; font-size: .78em; color: #55558a !important;
    margin-top: 6px; letter-spacing: .3px;
}

/* ═══ WHY CARD ════════════════════════════════════════════════════════════ */
.why-card {
    background: linear-gradient(135deg, #060616, #0b1222, #06120a);
    border: 1px solid rgba(0,255,159,.18); border-radius: 16px;
    padding: 28px; margin-top: 22px;
}
.why-card h3  { color: #00ff9f !important; margin-bottom: 12px; font-size: 1.0em;
                text-transform: uppercase; letter-spacing: 1px; }
.why-card p   { color: #8080b0 !important; line-height: 1.7; }
.why-card ul  { padding-left: 18px; }
.why-card li  { margin: 9px 0; color: #8080b0 !important; line-height: 1.6; }
.why-card strong { color: #d8d8f8 !important; }

/* ═══ BUTTONS ═════════════════════════════════════════════════════════════ */
div[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #00e5ff, #00aacc) !important;
    color: #000 !important; font-weight: 800; border: none !important;
    border-radius: 10px !important; font-size: 1.0em !important;
    box-shadow: 0 0 28px rgba(0,229,255,.28) !important;
}
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #0d0a28, #181540) !important;
    border: 1px solid rgba(0,229,255,.22) !important;
    color: #00e5ff !important; border-radius: 8px !important;
    font-weight: 600 !important; transition: all .18s ease;
}
div[data-testid="stButton"] > button:hover {
    border-color: rgba(0,229,255,.5) !important;
    box-shadow: 0 0 18px rgba(0,229,255,.2) !important;
}

/* ═══ SIMULATION ══════════════════════════════════════════════════════════ */
.sim-title {
    font-size: 1.9em; font-weight: 900;
    background: linear-gradient(90deg, #00e5ff, #c77dff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.start-banner {
    text-align: center; padding: 36px 20px;
    background: linear-gradient(135deg, #060616, #0b0b22);
    border: 1px dashed rgba(0,229,255,.22); border-radius: 16px; margin: 20px 0;
}
.start-banner h3 { color: #00e5ff !important; margin-bottom: 8px; }
.start-banner p  { color: #555588 !important; }

/* ═══ LIVE METRICS ════════════════════════════════════════════════════════ */
.lm {
    border-radius: 16px; padding: 24px 18px; text-align: center;
    margin-bottom: 10px; border: 2px solid;
    background: linear-gradient(145deg, #0d0a1e, #12102a);
    box-shadow: 0 4px 28px rgba(0,0,0,.35);
}
.lm-icon  { font-size: 1.3em; margin-bottom: 6px; }
.lm-value { font-size: 2.4em; font-weight: 900; line-height: 1.1; }
.lm-label { font-size: .76em; color: #44447a !important; margin-top: 5px; }

/* ═══ PROGRESS ════════════════════════════════════════════════════════════ */
[data-testid="stProgress"] > div { background: #18183a; border-radius: 8px; }
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #00e5ff, #c77dff) !important;
}

/* ═══ THREAT LOG ══════════════════════════════════════════════════════════ */
.log-col-head {
    font-size: .68em; font-weight: 800; color: #33336a !important;
    text-transform: uppercase; letter-spacing: 1.2px;
    padding: 0 4px 10px; border-bottom: 1px solid rgba(255,255,255,.06);
}
.log-cell   { font-size: .84em; color: #9090c0 !important; padding: 5px 4px; }
.log-attack { font-size: .84em; font-weight: 800; padding: 5px 4px; }
.log-conf   { font-size: .84em; font-weight: 600; color: #a0a0cc !important; padding: 5px 4px; }
.empty-box {
    text-align: center; padding: 44px 20px; color: #22225a !important;
    border: 1px dashed rgba(80,80,180,.18); border-radius: 12px; font-size: .95em;
}

/* ═══ INSPECT PANEL ═══════════════════════════════════════════════════════ */
.inspect-panel {
    border: 2px solid; border-radius: 18px; padding: 28px; margin-top: 22px;
    background: linear-gradient(145deg, #07061a, #0c0b28);
    box-shadow: 0 0 60px rgba(0,0,0,.5);
}
.inspect-panel h3 { margin-bottom: 20px; font-size: 1.1em; }
.igrid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 16px; margin-bottom: 20px;
}
.iitem       { display: flex; flex-direction: column; }
.iitem label {
    font-size: .68em; font-weight: 800; text-transform: uppercase;
    letter-spacing: 1.2px; color: #33336a !important; margin-bottom: 4px;
}
.iitem .iv { font-size: 1.1em; font-weight: 800; color: #e0e0ff !important; }
.explain-box { border-radius: 12px; padding: 16px; margin-top: 14px;
               background: rgba(0,0,0,.35); }
.explain-box strong { color: #d8d8f8 !important; }

/* ═══ PAUSED BANNER ══════════════════════════════════════════════════════ */
.paused-banner {
    text-align: center; padding: 10px 20px; border-radius: 10px;
    background: rgba(255,170,0,.08); border: 1px solid rgba(255,170,0,.3);
    color: #ffaa00 !important; font-weight: 700; font-size: .9em; margin: 8px 0;
}

/* ═══ DATAFRAME ══════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] th { background: #111130 !important; color: #7070a0 !important; }
[data-testid="stDataFrame"] td { color: #b0b0d0 !important; }

/* scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #07071a; }
::-webkit-scrollbar-thumb { background: #1a1a4a; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 4      # packets processed per tick
PACKET_INTERVAL = 0.6    # seconds per tick
SIM_DURATION    = 120.0  # 2 minutes

MODEL_METRICS = {
    "Logistic Regression": {"Accuracy":0.874,"Precision":0.861,"Recall":0.874,"F1-Score":0.862},
    "Random Forest":       {"Accuracy":0.972,"Precision":0.971,"Recall":0.972,"F1-Score":0.971},
    "XGBoost":             {"Accuracy":0.991,"Precision":0.991,"Recall":0.991,"F1-Score":0.991},
}
MODEL_COLORS = {"Logistic Regression":"#6b7fff","Random Forest":"#00e5ff","XGBoost":"#00ff9f"}

CLASS_COUNTS = {
    "DDoS":5000,"BENIGN":4970,"Port Scanning":4751,
    "DoS":4594,"Brute Force":4268,"Botnet":1948,"SQL Injection":21,
}
ATTACK_COLORS = {
    "BENIGN":"#00ff9f","Botnet":"#ff6b6b","Brute Force":"#ff9f43",
    "DDoS":"#ff1f5a","DoS":"#ff6090","Port Scanning":"#c77dff","SQL Injection":"#fbbf24",
}
STAT_COLORS = ["#00e5ff","#c77dff","#ff9f43","#00ff9f","#ff6b9d"]

# ── Load model artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    mdl   = joblib.load("models/best_model.pkl")
    enc   = joblib.load("models/label_encoder.pkl")
    imp   = joblib.load("models/imputer.pkl")
    feats = joblib.load("models/features.pkl")
    return mdl, enc, imp, feats

model, encoder, imputer, features = load_artifacts()

# ── Build curated simulation sequence from mini_cicids2017.csv ────────────────
@st.cache_data
def load_sim_data():
    df = pd.read_csv("mini_cicids2017.csv")

    # Drop original CICIDS " Label" column if present (it conflicts after stripping)
    # We want to keep "ProjectLabel" as our ground-truth label → rename to "Label"
    cols_to_drop = [c for c in df.columns if str(c).strip().lower() == "label"]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Normalise column names to match features.pkl
    new_cols = {}
    for c in df.columns:
        s = str(c).strip()
        new_cols[c] = "Label" if s == "ProjectLabel" else s.replace(" ", "_")
    df = df.rename(columns=new_cols)

    rng = np.random.RandomState(42)

    # --- sample from each class, always reset_index so iloc is safe ---
    benign_df  = df[df["Label"] == "BENIGN"].sample(n=550, random_state=1).reset_index(drop=True)
    attack_cls = ["DDoS","DoS","Port Scanning","Brute Force","Botnet","SQL Injection"]
    attack_parts = []
    for cls in attack_cls:
        sub = df[df["Label"] == cls].reset_index(drop=True)
        n   = min(55, len(sub))
        attack_parts.append(sub.sample(n=n, replace=(len(sub) < n), random_state=42).reset_index(drop=True))

    attacks_df = pd.concat(attack_parts, ignore_index=True).sample(frac=1, random_state=99).reset_index(drop=True)

    # --- build index-order list (avoid iloc on un-reset frames) ---
    order = []          # list of ('b', idx) or ('a', idx)
    b, a  = 0, 0
    gap   = int(rng.choice([5, 6, 7, 8]))
    ctr   = 0
    while b < len(benign_df) or a < len(attacks_df):
        if ctr >= gap and a < len(attacks_df):
            order.append(("a", a)); a += 1; ctr = 0
            gap = int(rng.choice([4, 5, 6, 7, 8]))
        elif b < len(benign_df):
            order.append(("b", b)); b += 1; ctr += 1
        else:
            while a < len(attacks_df):
                order.append(("a", a)); a += 1

    rows = []
    for kind, idx in order:
        rows.append((benign_df if kind == "b" else attacks_df).iloc[idx].to_dict())

    return pd.DataFrame(rows).reset_index(drop=True)

SIM_DATA = load_sim_data()
MAX_TICKS = int(SIM_DURATION / PACKET_INTERVAL)   # ~200 ticks

# ── Session state defaults ────────────────────────────────────────────────────
def init_state():
    defaults = {
        "page":           "home",
        "sim_running":    False,
        "sim_complete":   False,
        "sim_paused":     False,
        "sim_start_time": None,
        "sim_pause_start":None,
        "sim_threats":    [],
        "sim_packets":    0,
        "sim_seconds":    0.0,
        "last_tick":      -1,
        "inspected":      None,
        "inspect_paused": False,   # did inspect itself cause the pause?
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()
ss = st.session_state

# ── Helpers ───────────────────────────────────────────────────────────────────
def reset_simulation():
    ss.sim_running     = False
    ss.sim_complete    = False
    ss.sim_paused      = False
    ss.sim_start_time  = None
    ss.sim_pause_start = None
    ss.sim_threats     = []
    ss.sim_packets     = 0
    ss.sim_seconds     = 0.0
    ss.last_tick       = -1
    ss.inspected       = None
    ss.inspect_paused  = False


def effective_elapsed():
    """Wall-clock seconds adjusted for any ongoing pause."""
    if ss.sim_start_time is None:
        return 0.0
    el = time.time() - ss.sim_start_time
    return min(el, SIM_DURATION)


def process_tick(tick_idx: int):
    """Classify BATCH_SIZE consecutive packets starting at tick_idx*BATCH_SIZE."""
    start = tick_idx * BATCH_SIZE
    for i in range(BATCH_SIZE):
        idx = start + i
        if idx >= len(SIM_DATA):
            return
        row          = SIM_DATA.iloc[idx]
        actual_label = str(row.get("Label", "Unknown"))
        vals = []
        for f in features:
            raw = row.get(f, 0)
            try:    v = float(raw)
            except: v = 0.0
            vals.append(v)
        X = np.array([vals], dtype=np.float64)
        X = np.where(np.isinf(X), np.nan, X)
        X = imputer.transform(X)
        pred_code  = model.predict(X)[0]
        proba      = model.predict_proba(X)[0]
        prediction = encoder.inverse_transform([pred_code])[0]
        confidence = float(proba.max()) * 100.0
        if prediction != "BENIGN":
            ss.sim_threats.append({
                "packet_num":   idx + 1,
                "second":       round(tick_idx * PACKET_INTERVAL, 1),
                "prediction":   prediction,
                "confidence":   confidence,
                "actual_label": actual_label,
                "feat_vals":    {f: float(v) for f, v in zip(features, vals)},
            })
    ss.last_tick = tick_idx


def dark_chart(h=250):
    return dict(height=h, background="#0a0818")


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════════════════════
def show_home():

    # Hero
    st.markdown("""
    <div class="hero">
        <h1>CyberShield AI</h1>
        <p>AI-Powered Network Intrusion Detection &amp; Real-Time Threat Classification</p>
        <span class="hero-tag" style="background:rgba(0,229,255,.12);border:1px solid rgba(0,229,255,.3);color:#00e5ff;">
            CICIDS2017 Dataset
        </span>
        <span class="hero-tag" style="background:rgba(0,255,159,.10);border:1px solid rgba(0,255,159,.3);color:#00ff9f;">
            XGBoost &bull; 99.1% F1
        </span>
        <span class="hero-tag" style="background:rgba(199,125,255,.10);border:1px solid rgba(199,125,255,.3);color:#c77dff;">
            7-Class Detection
        </span>
        <span class="hero-tag" style="background:rgba(255,107,157,.10);border:1px solid rgba(255,107,157,.3);color:#ff6b9d;">
            78 Network Features
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Launch button — right below hero
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("▶  Launch Live Simulation", use_container_width=True, type="primary"):
            ss.page = "simulation"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Dataset Stats ────────────────────────────────────────
    st.markdown('<div class="sec-head">◆ Dataset Overview</div>', unsafe_allow_html=True)

    card_data = [
        ("25,552", "Total Records",      "#"),
        ("78",     "Network Features",   "#"),
        ("7",      "Attack Classes",     "#"),
        ("3",      "Models Compared",    "#"),
        ("99.1%",  "Best F1-Score",      "#"),
    ]
    cols = st.columns(5)
    icons = ["[DB]","[FEAT]","[CLASS]","[MDL]","[F1]"]
    for col, (val, lbl, _), icon, color in zip(cols, card_data, icons, STAT_COLORS):
        with col:
            st.markdown(f"""
            <div class="stat-card" style="border:1px solid {color}28;">
                <div class="stat-icon" style="color:{color};">{icon}</div>
                <div class="stat-value" style="color:{color};">{val}</div>
                <div class="stat-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # ── Class Distribution ────────────────────────────────────
    st.markdown('<div class="sec-head">◆ Attack Class Distribution in Dataset</div>',
                unsafe_allow_html=True)

    cdf    = pd.DataFrame(list(CLASS_COUNTS.items()), columns=["Class","Count"])\
               .sort_values("Count", ascending=False)
    domain = cdf["Class"].tolist()
    crange = [ATTACK_COLORS.get(c,"#aaa") for c in domain]

    dist = (
        alt.Chart(cdf)
        .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("Class:N", sort="-y",
                    axis=alt.Axis(labelAngle=0, labelColor="#7070a0",
                                  labelFontSize=10, domainColor="#222244",
                                  tickColor="#222244", title=None)),
            y=alt.Y("Count:Q",
                    axis=alt.Axis(labelColor="#7070a0", gridColor="#141430",
                                  domainColor="#222244", tickColor="#222244")),
            color=alt.Color("Class:N",
                            scale=alt.Scale(domain=domain, range=crange),
                            legend=None),
            tooltip=["Class","Count"],
        )
        .properties(**dark_chart(230))
    )
    st.altair_chart(dist, use_container_width=True)
    st.markdown('<div class="chart-caption">Sample count per traffic class — CICIDS2017 benchmark dataset</div>',
                unsafe_allow_html=True)

    # ── Model Comparison ──────────────────────────────────────
    st.markdown('<div class="sec-head">◆ Model Performance Comparison</div>',
                unsafe_allow_html=True)

    rows = []
    for mname, metrics in MODEL_METRICS.items():
        for metric, val in metrics.items():
            rows.append({"Model": mname, "Metric": metric, "Score": val})
    mdf = pd.DataFrame(rows)

    comp = (
        alt.Chart(mdf)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("Metric:N",
                    axis=alt.Axis(labelAngle=0, labelColor="#7070a0",
                                  domainColor="#222244", tickColor="#222244",
                                  title="Performance Metric")),
            y=alt.Y("Score:Q",
                    scale=alt.Scale(domain=[0.5, 1.03]),
                    axis=alt.Axis(labelColor="#7070a0", format=".0%",
                                  gridColor="#141430", domainColor="#222244",
                                  tickColor="#222244", title="Score")),
            color=alt.Color("Model:N",
                            scale=alt.Scale(
                                domain=list(MODEL_COLORS.keys()),
                                range=list(MODEL_COLORS.values()),
                            ),
                            legend=alt.Legend(
                                title="Model", orient="top",
                                labelColor="#b0b0d0", titleColor="#7070a0",
                                labelFontSize=11, titleFontSize=10,
                            )),
            xOffset="Model:N",
            tooltip=["Model","Metric", alt.Tooltip("Score:Q", format=".3f")],
        )
        .properties(**dark_chart(290))
    )
    st.altair_chart(comp, use_container_width=True)
    st.markdown(
        '<div class="chart-caption">'
        'Accuracy / Precision / Recall / F1-Score — evaluated on 20% stratified hold-out split from CICIDS2017'
        '</div>',
        unsafe_allow_html=True,
    )

    # Individual model cards
    card_cols = st.columns(3)
    for col, (mname, m) in zip(card_cols, MODEL_METRICS.items()):
        color   = MODEL_COLORS[mname]
        is_best = mname == "XGBoost"
        badge   = '<div class="best-badge">&#10022; SELECTED — BEST MODEL</div>' if is_best else ""
        with col:
            st.markdown(f"""
            <div class="model-card" style="border-color:{color}35;">
                <h4 style="color:{color};">{mname}</h4>
                <div class="mrow"><span>Accuracy</span><span class="mv" style="color:{color};">{m['Accuracy']:.1%}</span></div>
                <div class="mrow"><span>Precision</span><span class="mv" style="color:{color};">{m['Precision']:.1%}</span></div>
                <div class="mrow"><span>Recall</span><span class="mv" style="color:{color};">{m['Recall']:.1%}</span></div>
                <div class="mrow"><span>F1-Score</span><span class="mv" style="color:{color};">{m['F1-Score']:.1%}</span></div>
                {badge}
            </div>""", unsafe_allow_html=True)

    # Why XGBoost
    st.markdown("""
    <div class="why-card">
        <h3>&#9654; Why XGBoost was Selected</h3>
        <p>XGBoost achieved the highest F1-Score of <strong>99.1%</strong>, outperforming Logistic
        Regression (86.2%) by +12.9% and Random Forest (97.1%) by +2.0%.</p>
        <ul>
            <li><strong>Severe class imbalance handling</strong> — SQL Injection has only 21 samples vs 5,000 for DDoS;
                XGBoost's softprob objective still classifies it correctly</li>
            <li><strong>Histogram-based tree building</strong> (tree_method=hist) enables sub-second inference
                on 78-feature flow records, critical for real-time IDS</li>
            <li><strong>Built-in explainability</strong> — Feature importances allow security analysts to audit
                exactly which network metrics triggered each alert</li>
            <li><strong>Robustness to pcap artifacts</strong> — Handles Inf/NaN values common in flow-level
                features such as Flow Bytes/s and idle-time statistics</li>
            <li><strong>Calibrated confidence scores</strong> — softprob outputs probabilities across all 7 classes,
                enabling threat severity ranking</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — LIVE SIMULATION
# ═══════════════════════════════════════════════════════════════════════════════
def show_simulation():

    # ── Top bar ──────────────────────────────────────────────
    tc, bc, pc, rc = st.columns([7.5, 1.0, 0.8, 0.7])
    with tc:
        st.markdown('<h1 class="sim-title">&#9889; Live Packet Simulation</h1>',
                    unsafe_allow_html=True)
    with bc:
        if st.button("← Home", key="btn_home"):
            ss.page = "home"; st.rerun()
    with pc:
        # Pause / Resume toggle
        if ss.sim_running and not ss.sim_paused and not ss.sim_complete:
            if st.button("⏸", key="btn_pause", help="Pause simulation"):
                ss.sim_paused      = True
                ss.sim_pause_start = time.time()
                st.rerun()
        elif ss.sim_paused and not ss.inspect_paused:
            if st.button("▶", key="btn_resume", help="Resume simulation"):
                if ss.sim_pause_start:
                    ss.sim_start_time += (time.time() - ss.sim_pause_start)
                ss.sim_paused      = False
                ss.sim_pause_start = None
                st.rerun()
    with rc:
        if st.button("↺", key="btn_restart", help="Restart simulation"):
            reset_simulation(); st.rerun()

    # ── Start screen ─────────────────────────────────────────
    if not ss.sim_running and not ss.sim_complete:
        st.markdown("""
        <div class="start-banner">
            <h3>2-Minute Network Packet Capture Simulation</h3>
            <p>4 packets per 0.6 s tick &nbsp;&#183;&nbsp; ~800 total packets &nbsp;&#183;&nbsp;
            Realistic BENIGN-majority traffic &nbsp;&#183;&nbsp; All 7 attack classes appear</p>
        </div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        _, mid, _ = st.columns([1,2,1])
        with mid:
            if st.button("▶  Start Simulation", use_container_width=True, type="primary"):
                reset_simulation()
                ss.sim_running    = True
                ss.sim_start_time = time.time()
                st.rerun()
        return

    # ── Advance state if running ──────────────────────────────
    if ss.sim_running and not ss.sim_paused and not ss.inspected:
        elapsed = effective_elapsed()
        if elapsed >= SIM_DURATION:
            elapsed          = SIM_DURATION
            ss.sim_running   = False
            ss.sim_complete  = True
        target_tick = min(int(elapsed / PACKET_INTERVAL), MAX_TICKS - 1)
        for t in range(ss.last_tick + 1, target_tick + 1):
            process_tick(t)
        ss.sim_seconds = min(elapsed, SIM_DURATION)
        ss.sim_packets = (ss.last_tick + 1) * BATCH_SIZE

    # ── Paused banner ─────────────────────────────────────────
    if ss.sim_paused and not ss.sim_complete:
        reason = "Paused for inspection." if ss.inspect_paused else "Simulation paused."
        st.markdown(f'<div class="paused-banner">&#9646;&#9646; {reason}</div>',
                    unsafe_allow_html=True)

    # ── Live counters ─────────────────────────────────────────
    total_pkt  = ss.sim_packets
    elapsed_s  = ss.sim_seconds
    threat_cnt = len(ss.sim_threats)
    thr_pct    = (threat_cnt / total_pkt * 100) if total_pkt > 0 else 0.0

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f"""
        <div class="lm" style="border-color:#00e5ff;">
            <div class="lm-icon" style="color:#00e5ff;">[PKT]</div>
            <div class="lm-value" style="color:#00e5ff;">{total_pkt:,}</div>
            <div class="lm-label">Packets Processed</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        prog = min(elapsed_s / SIM_DURATION * 100, 100)
        st.markdown(f"""
        <div class="lm" style="border-color:#c77dff;">
            <div class="lm-icon" style="color:#c77dff;">[TIME]</div>
            <div class="lm-value" style="color:#c77dff;">{elapsed_s:.1f}s</div>
            <div class="lm-label">Elapsed — {prog:.0f}% of 120 s</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="lm" style="border-color:#ff4080;">
            <div class="lm-icon" style="color:#ff4080;">[ALERT]</div>
            <div class="lm-value" style="color:#ff4080;">{threat_cnt}</div>
            <div class="lm-label">Threats Detected ({thr_pct:.1f}%)</div>
        </div>""", unsafe_allow_html=True)

    if ss.sim_running and not ss.sim_paused:
        st.progress(min(elapsed_s / SIM_DURATION, 1.0),
                    text=f"Capturing traffic...  {int(elapsed_s)}s / 120s")
    elif ss.sim_complete:
        st.success(f"Simulation complete — {total_pkt:,} packets analysed, "
                   f"{threat_cnt} threats detected.  Press ↺ to restart.")

    st.divider()

    # ── Chart 1: Threat Timeline ──────────────────────────────
    st.markdown('<div class="sec-head">◆ Threat Timeline</div>', unsafe_allow_html=True)
    slot1 = st.empty()

    if ss.sim_threats:
        tdf             = pd.DataFrame(ss.sim_threats)
        tdf["bucket"]   = (tdf["second"] // 5).astype(int) * 5
        timeline        = tdf.groupby(["bucket","prediction"]).size().reset_index(name="Count")
        atk_dom         = list(ATTACK_COLORS.keys())
        atk_rng         = list(ATTACK_COLORS.values())

        t_chart = (
            alt.Chart(timeline)
            .mark_area(opacity=0.72, interpolate="monotone",
                       line={"strokeWidth": 1.5})
            .encode(
                x=alt.X("bucket:Q", title="Seconds Elapsed",
                        axis=alt.Axis(labelColor="#7070a0", gridColor="#141430",
                                      domainColor="#222244", tickColor="#222244")),
                y=alt.Y("Count:Q", stack="zero", title="Threat Count",
                        axis=alt.Axis(labelColor="#7070a0", gridColor="#141430",
                                      domainColor="#222244", tickColor="#222244")),
                color=alt.Color("prediction:N",
                                scale=alt.Scale(domain=atk_dom, range=atk_rng),
                                legend=alt.Legend(title="Attack Type", orient="right",
                                                  labelColor="#b0b0d0", titleColor="#7070a0")),
                tooltip=["bucket","prediction","Count"],
            )
            .properties(**dark_chart(240))
        )
        slot1.altair_chart(t_chart, use_container_width=True)
    else:
        slot1.markdown(
            '<div class="empty-box">Threat data will appear here as packets are analysed...</div>',
            unsafe_allow_html=True)

    # ── Chart 2: Attack Type Count ────────────────────────────
    st.markdown('<div class="sec-head">◆ Cumulative Attack Type Breakdown</div>',
                unsafe_allow_html=True)
    slot2 = st.empty()

    if ss.sim_threats:
        vc    = pd.DataFrame(ss.sim_threats)["prediction"].value_counts()
        tc_df = pd.DataFrame({"Attack Type": vc.index, "Count": vc.values})
        tc_df = tc_df.sort_values("Count", ascending=True)

        atk_dom = list(ATTACK_COLORS.keys())
        atk_rng = list(ATTACK_COLORS.values())

        type_chart = (
            alt.Chart(tc_df)
            .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
            .encode(
                x=alt.X("Count:Q", title="Total Detections",
                        axis=alt.Axis(labelColor="#7070a0", gridColor="#141430",
                                      domainColor="#222244", tickColor="#222244")),
                y=alt.Y("Attack Type:N", sort="-x", title=None,
                        axis=alt.Axis(labelColor="#c0c0e0", labelFontSize=11)),
                color=alt.Color("Attack Type:N",
                                scale=alt.Scale(domain=atk_dom, range=atk_rng),
                                legend=None),
                tooltip=["Attack Type","Count"],
            )
            .properties(**dark_chart(200))
        )
        slot2.altair_chart(type_chart, use_container_width=True)
    else:
        slot2.markdown(
            '<div class="empty-box">Attack breakdown will appear once threats are detected...</div>',
            unsafe_allow_html=True)

    st.divider()

    # ── Threat Log ────────────────────────────────────────────
    st.markdown('<div class="sec-head">◆ Threat Log</div>', unsafe_allow_html=True)

    if ss.sim_threats:
        hd = st.columns([0.5, 0.8, 1.9, 1.3, 1.0])
        for col, lbl in zip(hd, ["#","Packet","Attack Type","Confidence","Action"]):
            col.markdown(f'<div class="log-col-head">{lbl}</div>', unsafe_allow_html=True)

        display = list(reversed(ss.sim_threats))[:40]
        for i, t in enumerate(display):
            acolor = ATTACK_COLORS.get(t["prediction"], "#ff6b6b")
            rank   = len(ss.sim_threats) - i
            rw     = st.columns([0.5, 0.8, 1.9, 1.3, 1.0])
            with rw[0]: st.markdown(f'<div class="log-cell">{rank}</div>', unsafe_allow_html=True)
            with rw[1]: st.markdown(f'<div class="log-cell">#{t["packet_num"]}</div>', unsafe_allow_html=True)
            with rw[2]: st.markdown(f'<div class="log-attack" style="color:{acolor};">{t["prediction"]}</div>', unsafe_allow_html=True)
            with rw[3]: st.markdown(f'<div class="log-conf">{t["confidence"]:.1f}%</div>', unsafe_allow_html=True)
            with rw[4]:
                if st.button("Inspect", key=f"insp_{t['packet_num']}_{i}"):
                    ss.inspected = t
                    # Pause simulation on inspect
                    if ss.sim_running and not ss.sim_paused:
                        ss.sim_paused      = True
                        ss.sim_pause_start = time.time()
                        ss.inspect_paused  = True
    else:
        st.markdown('<div class="empty-box">No threats detected yet — monitoring in progress...</div>',
                    unsafe_allow_html=True)

    # ── Inspect Panel ─────────────────────────────────────────
    if ss.inspected:
        show_inspect_panel(ss.inspected)

    # ── Auto-rerun ────────────────────────────────────────────
    if ss.sim_running and not ss.sim_paused and not ss.inspected:
        elapsed_now = effective_elapsed()
        next_time   = (int(elapsed_now / PACKET_INTERVAL) + 1) * PACKET_INTERVAL
        sleep_secs  = max(0.05, min(next_time - elapsed_now, PACKET_INTERVAL))
        time.sleep(sleep_secs)
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  INSPECT PANEL
# ═══════════════════════════════════════════════════════════════════════════════
def show_inspect_panel(threat: dict):
    acolor = ATTACK_COLORS.get(threat["prediction"], "#ff6b6b")
    h = acolor.lstrip("#")
    r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)

    st.markdown(f"""
    <div class="inspect-panel" style="border-color:{acolor};">
        <h3 style="color:{acolor};">Threat Inspection — Packet #{threat['packet_num']}</h3>
        <div class="igrid">
            <div class="iitem"><label>Attack Type</label><span class="iv" style="color:{acolor};">{threat['prediction']}</span></div>
            <div class="iitem"><label>Confidence</label><span class="iv">{threat['confidence']:.2f}%</span></div>
            <div class="iitem"><label>Detected At</label><span class="iv">{threat['second']}s</span></div>
            <div class="iitem"><label>Packet No.</label><span class="iv">#{threat['packet_num']}</span></div>
            <div class="iitem"><label>Ground Truth</label><span class="iv">{threat['actual_label']}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature importances
    imp_df = (
        pd.DataFrame({
            "Feature":    features,
            "Importance": model.feature_importances_,
            "Value":      [threat["feat_vals"].get(f, 0.0) for f in features],
        })
        .sort_values("Importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    st.markdown("<br>", unsafe_allow_html=True)
    left, right = st.columns([1.6, 1])

    with left:
        st.markdown("**Top 15 Decisive Features (XGBoost Gain Importance)**")
        bar = (
            alt.Chart(imp_df)
            .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
            .encode(
                x=alt.X("Importance:Q",
                        axis=alt.Axis(labelColor="#7070a0", gridColor="#141430",
                                      domainColor="#222244", format=".3f")),
                y=alt.Y("Feature:N", sort="-x",
                        axis=alt.Axis(labelColor="#c0c0e0", labelFontSize=11)),
                color=alt.value(acolor),
                tooltip=["Feature",
                         alt.Tooltip("Importance:Q", format=".5f"),
                         alt.Tooltip("Value:Q",      format=".4f")],
            )
            .properties(**dark_chart(390))
        )
        st.altair_chart(bar, use_container_width=True)

    with right:
        st.markdown("**Packet Feature Values**")
        disp = imp_df[["Feature","Value","Importance"]].copy()
        disp["Value"]      = disp["Value"].map("{:.4f}".format)
        disp["Importance"] = disp["Importance"].map("{:.5f}".format)
        st.dataframe(disp, use_container_width=True, hide_index=True, height=410)

    top3 = imp_df.head(3)["Feature"].tolist()
    st.markdown(f"""
    <div class="explain-box" style="border:1px solid rgba({r},{g},{b},.2);">
        <strong style="color:{acolor};">Why was this classified as {threat['prediction']}?</strong><br><br>
        <span style="color:#9090c0;">
        The XGBoost model classified this packet as <strong style="color:{acolor};">{threat['prediction']}</strong>
        with <strong>{threat['confidence']:.1f}% confidence</strong>. The three most influential network flow
        metrics were <strong style="color:#d8d8f8;">{top3[0]}</strong>,
        <strong style="color:#d8d8f8;">{top3[1] if len(top3)>1 else '—'}</strong>, and
        <strong style="color:#d8d8f8;">{top3[2] if len(top3)>2 else '—'}</strong>.
        These features are the most discriminative signals for <strong style="color:{acolor};">
        {threat['prediction']}</strong>-type attacks in the CICIDS2017 benchmark.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("✕  Close Inspection", key="close_insp"):
        ss.inspected = None
        # Resume simulation if inspect caused the pause
        if ss.inspect_paused and ss.sim_running:
            if ss.sim_pause_start:
                ss.sim_start_time += (time.time() - ss.sim_pause_start)
            ss.sim_paused      = False
            ss.sim_pause_start = None
            ss.inspect_paused  = False
        st.rerun()


# ── Router ────────────────────────────────────────────────────────────────────
if ss.page == "home":
    show_home()
else:
    show_simulation()

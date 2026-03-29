import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import time

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Verity · Image Authenticity",
    page_icon="◈",
    layout="centered",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

/* ── Reset ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

/* ── Design tokens ── */
:root {
    --ink:       #0d0d0d;
    --paper:     #f5f0e8;
    --surface:   #fffdf7;
    --muted:     #8a8070;
    --border:    #d4cfc5;
    --accent:    #c8883a;
    --real:      #1a6b45;
    --real-bg:   #eaf5ee;
    --real-bd:   #a8d9be;
    --fake:      #8b1a1a;
    --fake-bg:   #faeaea;
    --fake-bd:   #e0aaaa;
    --radius-lg: 18px;
    --radius-md: 14px;
    --radius-sm: 10px;
    --shadow-sm: 0 2px 12px rgba(0,0,0,.06);
    --shadow-md: 0 4px 24px rgba(0,0,0,.10);
    --transition: .2s ease;
}

/* ── Base app ── */
.stApp {
    background-color: var(--paper);
    background-image:
        radial-gradient(ellipse 90% 55% at 50% -5%, rgba(200,136,58,.13) 0%, transparent 65%),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='60' height='60'%3E%3Ccircle cx='1' cy='1' r='.8' fill='%23c8883a' opacity='.07'/%3E%3C/svg%3E");
    font-family: 'DM Sans', sans-serif;
    color: var(--ink);
    -webkit-font-smoothing: antialiased;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Container: fluid on mobile, capped on desktop ── */
.block-container {
    padding: clamp(1.25rem, 5vw, 3rem) clamp(1rem, 5vw, 1.5rem) 4rem !important;
    max-width: min(720px, 100%) !important;
    width: 100% !important;
}

/* ═══════════════════════════════════
   HEADER
═══════════════════════════════════ */
.v-header {
    text-align: center;
    padding: clamp(1.5rem, 5vw, 2.5rem) 0 clamp(1.25rem, 4vw, 2rem);
}
.v-wordmark {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: clamp(2.2rem, 9vw, 3.8rem);
    letter-spacing: -.02em;
    color: var(--ink);
    line-height: 1;
}
.v-wordmark span { color: var(--accent); font-style: italic; }

.v-tagline {
    font-family: 'DM Mono', monospace;
    font-size: clamp(.6rem, 2.2vw, .72rem);
    font-weight: 400;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: .6rem;
}
.v-rule {
    width: 36px;
    height: 2px;
    background: var(--accent);
    margin: 1.1rem auto 0;
    border-radius: 2px;
}

/* ── Badge row ── */
.v-badges {
    display: flex;
    justify-content: center;
    gap: .5rem;
    flex-wrap: wrap;
    margin-top: 1rem;
}
.v-badge {
    font-family: 'DM Mono', monospace;
    font-size: .6rem;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--muted);
    border: 1px solid var(--border);
    border-radius: 99px;
    padding: .2rem .7rem;
    background: var(--surface);
}

/* ═══════════════════════════════════
   UPLOAD CARD
═══════════════════════════════════ */
.v-card {
    background: var(--surface);
    border: 1.5px solid var(--border);
    border-radius: var(--radius-lg);
    padding: clamp(1.25rem, 4vw, 2rem);
    box-shadow: var(--shadow-sm);
    margin-bottom: 1.4rem;
}
.v-card-label {
    font-family: 'DM Mono', monospace;
    font-size: .63rem;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .9rem;
    display: block;
}

/* ── Uploader zone ── */
[data-testid="stFileUploader"] { background: transparent !important; }
[data-testid="stFileUploader"] > div {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: rgba(213,207,197,.15) !important;
    padding: clamp(1rem, 4vw, 1.8rem) !important;
    transition: border-color var(--transition), background var(--transition);
}
[data-testid="stFileUploader"] > div:hover {
    border-color: var(--accent) !important;
    background: rgba(200,136,58,.06) !important;
}
[data-testid="stFileUploader"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: clamp(.8rem, 2.5vw, .95rem) !important;
    color: var(--muted) !important;
}

/* ── Supported formats hint ── */
.v-formats {
    font-family: 'DM Mono', monospace;
    font-size: .6rem;
    letter-spacing: .1em;
    color: var(--muted);
    text-align: center;
    margin-top: .7rem;
    text-transform: uppercase;
}

/* ═══════════════════════════════════
   IMAGE PREVIEW
═══════════════════════════════════ */
[data-testid="stImage"] { border-radius: var(--radius-md) !important; overflow: hidden !important; }
[data-testid="stImage"] img {
    border-radius: var(--radius-md) !important;
    border: 1.5px solid var(--border) !important;
    box-shadow: var(--shadow-md) !important;
    width: 100% !important;
    height: auto !important;
    display: block !important;
}

/* ── Image meta bar ── */
.v-img-meta {
    display: flex;
    align-items: center;
    gap: .6rem;
    flex-wrap: wrap;
    margin-top: .6rem;
    margin-bottom: .4rem;
}
.v-img-name {
    font-family: 'DM Mono', monospace;
    font-size: .65rem;
    letter-spacing: .1em;
    color: var(--muted);
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: .18rem .55rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 200px;
}
.v-img-size {
    font-family: 'DM Mono', monospace;
    font-size: .6rem;
    color: var(--muted);
}

/* ═══════════════════════════════════
   DIVIDER
═══════════════════════════════════ */
.v-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.4rem 0;
}

/* ═══════════════════════════════════
   RESULT BANNER
═══════════════════════════════════ */
.v-result {
    border-radius: var(--radius-lg);
    padding: clamp(1rem, 4vw, 1.6rem) clamp(1rem, 4vw, 2rem);
    display: flex;
    align-items: flex-start;
    gap: clamp(.8rem, 3vw, 1.2rem);
    margin-top: 1.2rem;
    animation: slideUp .4s cubic-bezier(.16,1,.3,1) both;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(14px); }
    to   { opacity: 1; transform: translateY(0); }
}
.v-result.real { background: var(--real-bg); border: 1.5px solid var(--real-bd); }
.v-result.fake { background: var(--fake-bg); border: 1.5px solid var(--fake-bd); }

/* Icon circle */
.v-result-icon-wrap {
    width: clamp(2.4rem, 8vw, 3rem);
    height: clamp(2.4rem, 8vw, 3rem);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-shrink: 0;
    font-size: clamp(1rem, 3.5vw, 1.3rem);
    font-weight: 700;
    font-family: 'DM Mono', monospace;
}
.real .v-result-icon-wrap { background: var(--real); color: #fff; }
.fake .v-result-icon-wrap { background: var(--fake); color: #fff; }

.v-result-body { flex: 1; min-width: 0; }

.v-result-verdict {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(1.2rem, 5vw, 1.55rem);
    line-height: 1.1;
    margin-bottom: .25rem;
}
.v-result.real .v-result-verdict { color: var(--real); }
.v-result.fake .v-result-verdict { color: var(--fake); }

.v-result-sub {
    font-family: 'DM Mono', monospace;
    font-size: clamp(.6rem, 2vw, .68rem);
    letter-spacing: .1em;
    color: var(--muted);
    text-transform: uppercase;
    margin-bottom: .9rem;
}

/* Confidence bar */
.v-conf-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: .4rem;
    gap: .5rem;
    flex-wrap: wrap;
}
.v-conf-label {
    font-family: 'DM Mono', monospace;
    font-size: clamp(.58rem, 1.8vw, .62rem);
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--muted);
}
.v-conf-pct {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(.95rem, 3.5vw, 1.1rem);
    color: var(--ink);
    white-space: nowrap;
}
.v-bar-wrap {
    height: 6px;
    background: rgba(0,0,0,.08);
    border-radius: 99px;
    overflow: hidden;
    width: 100%;
}
.v-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 1s cubic-bezier(.16,1,.3,1);
}
.real .v-bar-fill { background: var(--real); }
.fake .v-bar-fill { background: var(--fake); }

/* Stats grid */
.v-stats {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: .7rem;
    margin-top: 1rem;
}
.v-stat {
    background: rgba(255,255,255,.55);
    border: 1px solid rgba(0,0,0,.07);
    border-radius: var(--radius-sm);
    padding: .6rem .8rem;
    text-align: center;
}
.v-stat-val {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(.9rem, 3.5vw, 1.1rem);
    color: var(--ink);
    display: block;
}
.v-stat-key {
    font-family: 'DM Mono', monospace;
    font-size: .55rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--muted);
    display: block;
    margin-top: .15rem;
}

/* ═══════════════════════════════════
   FOOTER
═══════════════════════════════════ */
.v-footer {
    text-align: center;
    font-family: 'DM Mono', monospace;
    font-size: clamp(.55rem, 1.8vw, .62rem);
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
    line-height: 1.9;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    font-family: 'DM Mono', monospace !important;
    font-size: clamp(.68rem, 2.2vw, .75rem) !important;
    color: var(--muted) !important;
    letter-spacing: .1em !important;
}

/* ═══════════════════════════════════
   RESPONSIVE BREAKPOINTS
═══════════════════════════════════ */

/* Small phones */
@media (max-width: 480px) {
    .v-card    { padding: 1rem; }
    .v-result  { flex-direction: row; }
    .v-img-name { max-width: 130px; }
    .v-stats   { grid-template-columns: repeat(3, 1fr); gap: .45rem; }
    .v-stat    { padding: .5rem .4rem; }
}

/* Very small phones */
@media (max-width: 360px) {
    .v-badges  { gap: .35rem; }
    .v-badge   { font-size: .54rem; padding: .16rem .55rem; }
    .v-stats   { grid-template-columns: repeat(2, 1fr); }
    .v-result  { gap: .6rem; }
}

/* Tablets and up — slightly more breathing room */
@media (min-width: 600px) {
    .v-stats { grid-template-columns: repeat(3, 1fr); }
}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_keras_model():
    return load_model("model.keras")

model = load_keras_model()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="v-header">
    <div class="v-wordmark">Ver<span>ity</span></div>
    <div class="v-tagline">Image Authenticity Analysis</div>
    <div class="v-rule"></div>
    <div class="v-badges">
        <span class="v-badge">◈ Deep Learning</span>
        <span class="v-badge">◈ EfficientNet</span>
        <span class="v-badge">◈ Real-time</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Upload card ───────────────────────────────────────────────────────────────
st.markdown('<div class="v-card"><span class="v-card-label">◈ Upload Image</span>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    label="Drop an image or click to browse",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)

st.markdown("""
    <div class="v-formats">Supported · JPG · JPEG · PNG · WEBP</div>
</div>""", unsafe_allow_html=True)


# ── Analysis ──────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    w, h = img.size
    file_kb = round(uploaded_file.size / 1024, 1)

    # Image preview
    st.image(img, use_column_width=True)
    st.markdown(f"""
    <div class="v-img-meta">
        <span class="v-img-name">◈ {uploaded_file.name}</span>
        <span class="v-img-size">{w} × {h} px · {file_kb} KB</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="v-divider">', unsafe_allow_html=True)

    with st.spinner("Analysing authenticity…"):
        time.sleep(0.3)
        img_resized = img.resize((224, 224))
        img_array   = np.array(img_resized) / 255.0
        img_array   = np.expand_dims(img_array, axis=0)
        prediction  = model.predict(img_array)
        confidence  = float(prediction[0][0])

    is_fake   = confidence > 0.5
    verdict   = "Fabricated Image" if is_fake else "Authentic Image"
    css_cls   = "fake" if is_fake else "real"
    icon      = "✗" if is_fake else "✓"
    conf_pct  = confidence if is_fake else 1 - confidence
    bar_width = round(conf_pct * 100)
    risk      = "High" if conf_pct > 0.85 else "Medium" if conf_pct > 0.65 else "Low"

    st.markdown("""
    <div class="v-result {css_cls}">
        <div class="v-result-icon-wrap">{icon}</div>
        <div class="v-result-body">
            <div class="v-result-verdict">{verdict}</div>
            <div class="v-result-sub">Model confidence signal</div>

            <div class="v-conf-row">
                <span class="v-conf-label">Confidence</span>
                <span class="v-conf-pct">{conf_pct:.1%}</span>
            </div>
            <div class="v-bar-wrap">
                <div class="v-bar-fill" style="width:{bar_width}%"></div>
            </div>

            <div class="v-stats">
                <div class="v-stat">
                    <span class="v-stat-val">{conf_pct:.1%}</span>
                    <span class="v-stat-key">Score</span>
                </div>
                <div class="v-stat">
                    <span class="v-stat-val">{risk}</span>
                    <span class="v-stat-key">Risk</span>
                </div>
                <div class="v-stat">
                    <span class="v-stat-val">224px</span>
                    <span class="v-stat-key">Input</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="v-footer">
    Verity v1.0 · Powered by Deep Learning<br>
    Results are probabilistic — use as a guide, not a guarantee
</div>
""", unsafe_allow_html=True)
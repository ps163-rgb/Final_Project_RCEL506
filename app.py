"""
RCEL 506 — Legal Document Treatment Classifier
Streamlit App | Pavan Sastry
Run: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pickle, os, re, time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LegalSense | Document Classifier",
    page_icon="⚖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0D1B2A !important;
    color: #E8EDF2 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }

/* ── Left accent bar ── */
body::before {
    content: '';
    position: fixed;
    left: 0; top: 0;
    width: 5px; height: 100vh;
    background: #0E8C8C;
    z-index: 9999;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #0D1B2A; }
::-webkit-scrollbar-thumb { background: #0E8C8C; border-radius: 10px; }

/* ── Textarea ── */
textarea {
    background: #1B2A3B !important;
    color: #E8EDF2 !important;
    border: 1px solid #1E3A4A !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 13px !important;
    line-height: 1.7 !important;
    resize: vertical !important;
    transition: border-color 0.25s ease !important;
}
textarea:focus {
    border-color: #0E8C8C !important;
    box-shadow: 0 0 0 2px rgba(14,140,140,0.15) !important;
    outline: none !important;
}
textarea::placeholder { color: #4A6070 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #0E8C8C, #14B4B4) !important;
    color: #0D1B2A !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.65rem 2rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(14,140,140,0.35) !important;
    background: linear-gradient(135deg, #14B4B4, #1ACECE) !important;
}

/* ── Select box ── */
[data-testid="stSelectbox"] > div > div {
    background: #1B2A3B !important;
    color: #E8EDF2 !important;
    border: 1px solid #1E3A4A !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #0E8C8C !important; }

/* ── Hide Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Label metadata ─────────────────────────────────────────────────────────────
LABEL_META = {
    "cited": {
        "color": "#4A90D9",
        "icon": "📌",
        "desc": "The case is referenced as legal authority but the ruling is not directly applied to the current facts.",
        "risk": "LOW",
        "risk_color": "#27AE60",
    },
    "applied": {
        "color": "#E8A020",
        "icon": "⚖",
        "desc": "The legal test or principle from the cited case is directly applied to resolve the current dispute.",
        "risk": "HIGH",
        "risk_color": "#C0392B",
    },
    "followed": {
        "color": "#0E8C8C",
        "icon": "→",
        "desc": "The court adopts and follows the reasoning of the prior case as binding or persuasive precedent.",
        "risk": "MEDIUM",
        "risk_color": "#E8A020",
    },
    "referred to": {
        "color": "#8AA0B4",
        "icon": "🔗",
        "desc": "The case is mentioned in passing or for contextual background without substantive reliance.",
        "risk": "LOW",
        "risk_color": "#27AE60",
    },
    "considered": {
        "color": "#9B59B6",
        "icon": "🔍",
        "desc": "The court examines the case carefully but may not fully adopt its reasoning or outcome.",
        "risk": "MEDIUM",
        "risk_color": "#E8A020",
    },
}

# ── Model loading ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    """Load pre-trained model from pickle, or train a demo model if not found."""
    model_path = "rf_model.pkl"
    tfidf_path = "tfidf.pkl"

    if os.path.exists(model_path) and os.path.exists(tfidf_path):
        with open(tfidf_path, "rb") as f:
            tfidf = pickle.load(f)
        with open(model_path, "rb") as f:
            rf = pickle.load(f)
        return tfidf, rf

    # ── Demo model: train on small representative samples ──
    samples = [
        # cited
        ("The principle in Donoghue v Stevenson was cited as foundational authority.", "cited"),
        ("Smith v Jones was cited by counsel in support of the proposition.", "cited"),
        ("The court cited Harrison v Lee for the general duty of care framework.", "cited"),
        ("As cited in Brown v Board, equal protection applies universally.", "cited"),
        ("Counsel cited multiple authorities including White v Black.", "cited"),
        ("The case was cited simply as an example of the broader principle.", "cited"),
        # applied
        ("The duty of care test was applied directly to the facts of this case.", "applied"),
        ("Applying the test from Caparo, the court found the defendant liable.", "applied"),
        ("The reasonable person standard was applied to assess the defendant's conduct.", "applied"),
        ("The principle applied here requires proof of causation and damage.", "applied"),
        ("The Bolam test was applied and the defendant met the required standard.", "applied"),
        # followed
        ("The court followed the reasoning in Hedley Byrne without deviation.", "followed"),
        ("Following established precedent, the tribunal held in favour of the claimant.", "followed"),
        ("The approach followed was that set out in the earlier Court of Appeal decision.", "followed"),
        ("This court followed the position adopted in the landmark ruling.", "followed"),
        # referred to
        ("The matter was briefly referred to in the judgment without further analysis.", "referred to"),
        ("Counsel referred to several cases in passing during oral submissions.", "referred to"),
        ("The decision in Taylor v Lane was referred to as background context.", "referred to"),
        ("The judge referred to the legislative history to clarify intent.", "referred to"),
        # considered
        ("The court considered but ultimately distinguished the earlier authority.", "considered"),
        ("After carefully considered the submissions, the tribunal reached its conclusion.", "considered"),
        ("The competing interests were considered in light of proportionality.", "considered"),
        ("The court considered the applicability of the precedent to these novel facts.", "considered"),
    ]

    texts, labels = zip(*samples)
    tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words="english")
    X = tfidf.fit_transform(texts)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, labels)
    return tfidf, rf

# ── Example texts ─────────────────────────────────────────────────────────────
EXAMPLES = {
    "Applied — Donoghue v Stevenson": """The principle established in Donoghue v Stevenson was applied directly to the present matter. The court found that the duty of care test, as applied in that case, was directly applicable to the facts before it.""",
    "Cited — Authority Reference": """The court cited Caparo Industries plc v Dickman [1990] as foundational authority for the three-stage test of duty of care, without substantively applying it to the present dispute.""",
    "Followed — Binding Precedent": """Following the reasoning in Hedley Byrne & Co Ltd v Heller & Partners Ltd, the tribunal held that a duty of care in negligent misstatement arises where there is a special relationship of reliance.""",
    "Referred To — Background": """Counsel briefly referred to White v Jones in passing, noting it as contextual background to the broader discussion of solicitor liability, without placing material reliance upon it.""",
    "Considered — Careful Examination": """The court carefully considered the applicability of Bolam v Friern Hospital Management Committee to the present facts, ultimately finding that while the principle was relevant, the circumstances warranted a modified approach.""",
}

# ── Classify function ─────────────────────────────────────────────────────────
def classify(text, tfidf, rf):
    vec   = tfidf.transform([text])
    label = rf.predict(vec)[0]
    probs = rf.predict_proba(vec)[0]
    prob_dict = dict(zip(rf.classes_, probs))
    return label, prob_dict

# ── Render confidence bars ─────────────────────────────────────────────────────
def render_bars(prob_dict, predicted):
    sorted_items = sorted(prob_dict.items(), key=lambda x: -x[1])
    bars_html = ""
    for label, prob in sorted_items:
        meta   = LABEL_META.get(label, {})
        color  = meta.get("color", "#8AA0B4")
        icon   = meta.get("icon", "•")
        pct    = prob * 100
        is_top = label == predicted
        border = f"border: 1px solid {color}44;" if is_top else "border: 1px solid #1E3A4A;"
        glow   = f"box-shadow: 0 0 12px {color}22;" if is_top else ""
        badge  = f'<span style="background:{color}22;color:{color};font-size:10px;padding:2px 8px;border-radius:20px;font-weight:600;margin-left:8px;">PREDICTED</span>' if is_top else ""
        bars_html += f"""
        <div style="background:#1B2A3B;border-radius:10px;padding:14px 16px;margin-bottom:10px;{border}{glow}transition:all 0.3s;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">
                <span style="font-family:'DM Sans',sans-serif;font-weight:{'600' if is_top else '400'};color:{'#E8EDF2' if is_top else '#8AA0B4'};font-size:13px;">
                    {icon}&nbsp;&nbsp;{label.upper()}{badge}
                </span>
                <span style="font-family:'DM Mono',monospace;font-size:13px;color:{color};font-weight:600;">{pct:.1f}%</span>
            </div>
            <div style="background:#0D1B2A;border-radius:4px;height:6px;overflow:hidden;">
                <div style="width:{pct}%;height:100%;background:linear-gradient(90deg,{color},{color}aa);border-radius:4px;transition:width 0.8s ease;"></div>
            </div>
        </div>
        """
    return bars_html

# ── Load model ────────────────────────────────────────────────────────────────
with st.spinner("Loading Random Forest model…"):
    tfidf, rf = load_model()

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(180deg,#0D1B2A 0%,#111E2C 100%);padding:40px 48px 28px;border-bottom:1px solid #1E3A4A;">
    <div style="display:flex;align-items:flex-end;gap:16px;margin-bottom:6px;">
        <span style="font-family:'Playfair Display',serif;font-size:36px;font-weight:700;color:#E8EDF2;letter-spacing:-0.5px;">LegalSense</span>
        <span style="font-family:'DM Mono',monospace;font-size:11px;color:#0E8C8C;margin-bottom:8px;letter-spacing:0.12em;font-weight:500;">v1.0 · RANDOM FOREST</span>
    </div>
    <p style="font-family:'DM Sans',sans-serif;color:#8AA0B4;font-size:14px;max-width:620px;line-height:1.6;margin:0;">
        NLP-powered legal document treatment classifier. Paste any legal or regulatory text to detect 
        how a prior case is being treated — <em>cited, applied, followed, referred to,</em> or <em>considered.</em>
    </p>
    <div style="display:flex;gap:24px;margin-top:18px;">
        <div style="display:flex;align-items:center;gap:7px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#27AE60;box-shadow:0 0 6px #27AE60;"></div>
            <span style="font-family:'DM Mono',monospace;font-size:11px;color:#8AA0B4;">MODEL ACTIVE</span>
        </div>
        <div style="display:flex;align-items:center;gap:7px;">
            <span style="font-family:'DM Mono',monospace;font-size:11px;color:#8AA0B4;">63.0% TEST ACCURACY</span>
        </div>
        <div style="display:flex;align-items:center;gap:7px;">
            <span style="font-family:'DM Mono',monospace;font-size:11px;color:#8AA0B4;">+10pp OVER BASELINE</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Main layout ───────────────────────────────────────────────────────────────
st.markdown('<div style="padding:32px 48px;">', unsafe_allow_html=True)

left_col, right_col = st.columns([1.05, 0.95], gap="large")

with left_col:
    st.markdown("""
    <p style="font-family:'DM Sans',sans-serif;font-size:11px;color:#0E8C8C;font-weight:600;letter-spacing:0.12em;margin-bottom:8px;">
        INPUT DOCUMENT
    </p>
    """, unsafe_allow_html=True)

    # Example selector
    example_choice = st.selectbox(
        "Load an example",
        ["— type or paste your own text —"] + list(EXAMPLES.keys()),
        label_visibility="collapsed",
    )

    default_text = EXAMPLES.get(example_choice, "")

    user_text = st.text_area(
        "Document text",
        value=default_text,
        height=240,
        placeholder="Paste legal or regulatory text here…\n\nExample: 'The duty of care test established in Donoghue v Stevenson was applied directly to the present matter.'",
        label_visibility="collapsed",
    )

    word_count = len(user_text.split()) if user_text.strip() else 0
    st.markdown(f"""
    <p style="font-family:'DM Mono',monospace;font-size:11px;color:#4A6070;text-align:right;margin-top:4px;">
        {word_count} words
    </p>
    """, unsafe_allow_html=True)

    run_btn = st.button("⚖  Classify Document", use_container_width=True)

    # ── How it works expander ──
    st.markdown('<div style="margin-top:20px;">', unsafe_allow_html=True)
    with st.expander("How does this work?"):
        st.markdown("""
        <div style="font-family:'DM Sans',sans-serif;font-size:13px;color:#8AA0B4;line-height:1.8;">
        <b style="color:#0E8C8C;">Pipeline:</b><br>
        1. Raw text is <b style="color:#E8EDF2;">tokenized</b> — split into individual words/n-grams<br>
        2. Stop words (the, and, of…) are removed as noise<br>
        3. <b style="color:#E8EDF2;">TF-IDF</b> converts tokens into weighted numeric features<br>
        4. A <b style="color:#E8EDF2;">Random Forest</b> (63% accuracy, +10pp over naive baseline) predicts the treatment class<br><br>
        <b style="color:#0E8C8C;">Dataset:</b> 24,985 Australian legal cases · 5 treatment classes<br>
        <b style="color:#0E8C8C;">Validation:</b> 10-fold cross-validation · Weighted F1: 0.43
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown("""
    <p style="font-family:'DM Sans',sans-serif;font-size:11px;color:#0E8C8C;font-weight:600;letter-spacing:0.12em;margin-bottom:8px;">
        CLASSIFICATION RESULT
    </p>
    """, unsafe_allow_html=True)

    result_placeholder = st.empty()

    if not run_btn and not st.session_state.get("has_result"):
        result_placeholder.markdown("""
        <div style="background:#1B2A3B;border-radius:12px;border:1px dashed #1E3A4A;padding:60px 24px;text-align:center;">
            <div style="font-size:32px;margin-bottom:12px;">⚖</div>
            <p style="font-family:'DM Sans',sans-serif;color:#4A6070;font-size:13px;line-height:1.6;">
                Paste a legal document excerpt and<br>click <strong style="color:#8AA0B4;">Classify Document</strong> to see results.
            </p>
        </div>
        """, unsafe_allow_html=True)

    if run_btn:
        if not user_text.strip():
            result_placeholder.markdown("""
            <div style="background:#1A0A0A;border-radius:12px;border:1px solid #C0392B44;padding:24px;text-align:center;">
                <p style="font-family:'DM Sans',sans-serif;color:#C0392B;font-size:13px;">
                    ⚠ Please enter some text before classifying.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Analysing…"):
                time.sleep(0.4)
                label, prob_dict = classify(user_text, tfidf, rf)

            st.session_state["has_result"] = True
            meta       = LABEL_META.get(label, {})
            color      = meta.get("color", "#8AA0B4")
            icon       = meta.get("icon", "•")
            desc       = meta.get("desc", "")
            risk       = meta.get("risk", "—")
            risk_color = meta.get("risk_color", "#8AA0B4")
            confidence = prob_dict.get(label, 0) * 100
            bars_html  = render_bars(prob_dict, label)

            result_placeholder.markdown(f"""
            <div>
                <!-- Primary result card -->
                <div style="background:#1B2A3B;border-radius:12px;border:1px solid {color}55;
                            padding:24px;margin-bottom:16px;
                            box-shadow:0 0 24px {color}18;">
                    <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">
                        <div>
                            <p style="font-family:'DM Mono',monospace;font-size:10px;color:#4A6070;letter-spacing:0.12em;margin-bottom:4px;">TREATMENT CLASS</p>
                            <p style="font-family:'Playfair Display',serif;font-size:28px;color:{color};font-weight:700;line-height:1.1;">
                                {icon}&nbsp;{label.title()}
                            </p>
                        </div>
                        <div style="text-align:right;">
                            <p style="font-family:'DM Mono',monospace;font-size:10px;color:#4A6070;letter-spacing:0.12em;margin-bottom:4px;">CONFIDENCE</p>
                            <p style="font-family:'Playfair Display',serif;font-size:28px;color:{color};font-weight:700;">{confidence:.1f}%</p>
                        </div>
                    </div>
                    <div style="background:#0D1B2A;border-radius:8px;padding:12px 14px;margin-bottom:14px;">
                        <p style="font-family:'DM Sans',sans-serif;font-size:12.5px;color:#8AA0B4;line-height:1.65;margin:0;">{desc}</p>
                    </div>
                    <div style="display:flex;gap:10px;">
                        <div style="background:{risk_color}18;border:1px solid {risk_color}44;border-radius:6px;padding:6px 14px;">
                            <span style="font-family:'DM Mono',monospace;font-size:10px;color:{risk_color};letter-spacing:0.1em;font-weight:600;">
                                {risk} OPERATIONAL RISK
                            </span>
                        </div>
                        <div style="background:#0E8C8C18;border:1px solid #0E8C8C44;border-radius:6px;padding:6px 14px;">
                            <span style="font-family:'DM Mono',monospace;font-size:10px;color:#0E8C8C;letter-spacing:0.1em;font-weight:600;">
                                RANDOM FOREST · RF
                            </span>
                        </div>
                    </div>
                </div>

                <!-- Confidence breakdown -->
                <p style="font-family:'DM Sans',sans-serif;font-size:11px;color:#0E8C8C;font-weight:600;letter-spacing:0.12em;margin-bottom:10px;">
                    CONFIDENCE BREAKDOWN
                </p>
                {bars_html}
            </div>
            """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="border-top:1px solid #1E3A4A;padding:18px 48px;margin-top:16px;display:flex;justify-content:space-between;align-items:center;">
    <span style="font-family:'DM Mono',monospace;font-size:10px;color:#2A3A4A;">
        RCEL 506 · NLP Legal Document Classifier · Pavan Sastry · Random Forest · TF-IDF
    </span>
    <span style="font-family:'DM Mono',monospace;font-size:10px;color:#2A3A4A;">
        Dataset: Legal Text Classification — Kaggle (amohankumar)
    </span>
</div>
</div>
""", unsafe_allow_html=True)

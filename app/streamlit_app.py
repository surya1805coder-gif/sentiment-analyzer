"""Streamlit UI — DistilBERT sentiment with general-purpose fallback, batch mode, and active learning."""

import sys, csv, io, json, re
from pathlib import Path
import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.express as px

_APP_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _APP_DIR.parent
sys.path.insert(0, str(_PROJECT_DIR))

from src.data.loader import clean_text
from src.model.architecture import ID2LABEL

MODEL_DIR = _PROJECT_DIR / "models" / "sentiment_model"
FEEDBACK_FILE = _PROJECT_DIR / "feedback.jsonl"

EMOJI_MAP = {0: "\U0001f621", 1: "\U0001f610", 2: "\U0001f60a"}
COLOR_MAP = {0: "#dc3545", 1: "#fd7e14", 2: "#28a745"}
LABEL_NAMES = ["Negative", "Neutral", "Positive"]
THEME_KEYWORDS = {
    "Baggage": ["bag", "luggage", "lost bag", "suitcase", "checked bag"],
    "Delay": ["delay", "late", "waiting", "stuck", "cancel"],
    "Refund": ["refund", "money back", "compensation", "voucher", "credit"],
    "Customer Service": ["agent", "support", "help", "complaint", "rude", "unhelpful"],
    "Food": ["food", "meal", "snack", "hungry", "drink"],
    "Seating": ["seat", "legroom", "recline", "aisle", "window seat"],
}

CONF_THRESHOLD = 0.60


# ── Model Loading ────────────────────────────────────────────────────────

@st.cache_resource
def load_airline_model():
    if not MODEL_DIR.exists():
        st.error(f"Model not found at `{MODEL_DIR}`. Run `python main.py` first.")
        st.stop()
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_DIR))
    model.to(device)
    model.eval()
    return model, tokenizer, device


@st.cache_resource
def load_general_pipeline():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


_general_pipe = load_general_pipeline()


# ── Airline Prediction ───────────────────────────────────────────────────

def predict_airline(text, model, tokenizer, device):
    cleaned = clean_text(text)
    if not cleaned:
        return None, None, None, ""
    inputs = tokenizer(cleaned, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred = int(np.argmax(probs))
    return pred, float(probs[pred]), probs, cleaned


@st.cache_data(show_spinner=False)
def predict_general(text):
    result = _general_pipe(text[:512])[0]
    label = result["label"].lower()
    conf = result["score"]
    if label == "positive":
        return 2, conf, np.array([1 - conf, 0.0, conf]), text
    return 0, conf, np.array([conf, 0.0, 1 - conf]), text


# ── Word Highlighting ────────────────────────────────────────────────────

def highlight_words(text, sentiment_label):
    positive_words = {"good", "great", "amazing", "love", "best", "awesome",
                      "fantastic", "excellent", "wonderful", "happy", "thanks",
                      "perfect", "nice", "helpful", "comfortable", "smooth",
                      "easy", "fast", "friendly", "beautiful", "kind"}
    negative_words = {"bad", "terrible", "awful", "hate", "worst", "horrible",
                      "poor", "rude", "ugly", "slow", "late", "delay",
                      "cancel", "lost", "broken", "uncomfortable", "disgusting",
                      "pathetic", "useless", "annoying", "stupid", "dirty",
                      "miss", "refund", "complaint", "damage", "never"}
    words = text.split()
    result_words = []
    for w in words:
        clean_w = re.sub(r"[^a-zA-Z]", "", w).lower()
        if sentiment_label == "Positive" and clean_w in positive_words:
            result_words.append(f'<span style="background:#b7e4c7;padding:0 4px;border-radius:4px;font-weight:600;">{w}</span>')
        elif sentiment_label == "Negative" and clean_w in negative_words:
            result_words.append(f'<span style="background:#f5c6cb;padding:0 4px;border-radius:4px;font-weight:600;">{w}</span>')
        else:
            result_words.append(w)
    return " ".join(result_words)


# ── Theme Extraction ─────────────────────────────────────────────────────

def extract_themes(text):
    text_lower = text.lower()
    found = []
    for theme, keywords in THEME_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            found.append(theme)
    return found if found else ["General"]


# ── Session State Init ───────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history = []
if "model_mode" not in st.session_state:
    st.session_state.model_mode = "Airline Model"


# ── Page Config ──────────────────────────────────────────────────────────

st.set_page_config(page_title="Sentiment Analyzer Pro", page_icon="\U0001f4ac", layout="wide")

# ── Sidebar Controls ─────────────────────────────────────────────────────

with st.sidebar:
    st.header("\u2699\ufe0f Settings")
    mode = st.radio("Model Mode", ["Airline Model", "General Model"],
                    index=0, help="Airline = trained on tweets. General = works on any text.")
    st.session_state.model_mode = mode
    st.divider()
    show_history = st.checkbox("Show Prediction History", value=True)
    show_dashboard = st.checkbox("Show Dashboard", value=True)
    st.divider()
    st.caption(f"Running on {'CPU' if not torch.cuda.is_available() else 'GPU'}")

# ── Main Title ───────────────────────────────────────────────────────────

st.title("\U0001f4ac Sentiment Analyzer Pro")
st.markdown(
    "**Airline Model**: fine-tuned on Twitter airline tweets  |  "
    "**General Model**: zero-shot on any text"
)
st.divider()

# ── Batch Upload ─────────────────────────────────────────────────────────

with st.expander("\U0001f4c1 Batch Upload CSV", expanded=False):
    uploaded = st.file_uploader("Upload CSV with a 'text' column", type="csv")
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        if "text" not in batch_df.columns:
            st.error("CSV must have a 'text' column")
        else:
            texts = batch_df["text"].dropna().tolist()
            if texts:
                st.info(f"Analyzing {len(texts)} tweets...")
                airline_model = None
                if mode == "Airline Model":
                    airline_model, tok, dev = load_airline_model()
                batch_results = []
                for t in texts:
                    if mode == "Airline Model":
                        p, c, probs, _ = predict_airline(t, *airline_model)
                    else:
                        p, c, probs, _ = predict_general(t)
                    themes = extract_themes(t)
                    batch_results.append({
                        "text": t[:80] + ("..." if len(t) > 80 else ""),
                        "sentiment": LABEL_NAMES[p] if p is not None else "?",
                        "confidence": f"{c:.1%}" if c else "?",
                        "themes": ", ".join(themes),
                    })
                st.dataframe(pd.DataFrame(batch_results), use_container_width=True)

st.divider()

# ── Single Prediction ────────────────────────────────────────────────────

col_input, col_preview = st.columns([3, 1])

with col_input:
    tweet = st.text_area("Enter text to analyze:", height=130,
                         placeholder="e.g., @United your customer service is terrible...")

with col_preview:
    st.markdown("**Word Highlight Preview**")
    if tweet.strip():
        st.markdown(highlight_words(tweet, "Negative"), unsafe_allow_html=True)

analyze = st.button("\U0001f50d Analyze", type="primary", use_container_width=True)

if analyze:
    if not tweet.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            if mode == "Airline Model":
                model, tokenizer, device = load_airline_model()
                pred, conf, probs, cleaned = predict_airline(tweet, model, tokenizer, device)
            else:
                pred, conf, probs, cleaned = predict_general(tweet)

        if pred is None:
            st.warning("Could not analyze. Try different text.")
        else:
            label = LABEL_NAMES[pred]
            emoji = EMOJI_MAP[pred]

            # Feature 6: Confidence warning
            if conf < CONF_THRESHOLD:
                st.warning(f"\u26a0\ufe0f Low confidence ({conf:.1%}) — model is unsure. Consider trying the General model.")

            st.divider()
            row1 = st.columns([1, 2, 1])

            with row1[0]:
                st.markdown(f"# {emoji}")
                st.markdown(f"<h2 style='color:{COLOR_MAP[pred]};margin:0;'>{label}</h2>", unsafe_allow_html=True)

            with row1[1]:
                st.markdown("**Confidence**")
                bar_color = COLOR_MAP[pred]
                st.markdown(f"""
                    <div style='height:22px;width:100%;background:#e9ecef;border-radius:11px;overflow:hidden;'>
                    <div style='height:100%;width:{conf*100:.0f}%;background:{bar_color};
                    border-radius:11px;text-align:center;color:white;font-size:12px;line-height:22px;'>{conf:.1%}</div></div>
                """, unsafe_allow_html=True)

            with row1[2]:
                themes = extract_themes(tweet)
                st.markdown("**Detected Themes**")
                st.markdown(", ".join(f"`{t}`" for t in themes))

            # Feature 3: Word highlighting
            st.divider()
            st.markdown("#### Highlighted Text")
            st.markdown(highlight_words(tweet, label), unsafe_allow_html=True)

            # Probability breakdown
            st.divider()
            st.markdown("#### Probability Breakdown")
            for i, name in enumerate(LABEL_NAMES):
                pct = float(probs[i])
                bc = COLOR_MAP[i]
                st.markdown(f"""
                    <div style='display:flex;align-items:center;gap:8px;margin:4px 0;'>
                    <span style='width:80px;text-align:right;'>{name}</span>
                    <div style='flex:1;height:22px;background:#e9ecef;border-radius:11px;overflow:hidden;'>
                    <div style='height:100%;width:{pct*100:.0f}%;background:{bc};
                    border-radius:11px;text-align:center;color:white;font-size:12px;line-height:22px;'>{pct:.1%}</div></div></div>
                """, unsafe_allow_html=True)

            # Feature 8: Active learning feedback
            st.divider()
            st.markdown("#### Was this prediction correct?")
            col_fb1, col_fb2, _ = st.columns([1, 1, 4])
            with col_fb1:
                if st.button("\u2705 Yes", key="fb_yes"):
                    entry = {"text": tweet, "predicted": label, "correct": True, "mode": mode}
                    with open(FEEDBACK_FILE, "a") as f:
                        f.write(json.dumps(entry) + "\n")
                    st.success("Thanks! Feedback saved.")
            with col_fb2:
                if st.button("\u274c No", key="fb_no"):
                    entry = {"text": tweet, "predicted": label, "correct": False, "mode": mode}
                    with open(FEEDBACK_FILE, "a") as f:
                        f.write(json.dumps(entry) + "\n")
                    st.error("Thanks! We'll improve.")

            # Save to history
            st.session_state.history.insert(0, {
                "text": tweet[:60] + ("..." if len(tweet) > 60 else ""),
                "label": label,
                "confidence": f"{conf:.1%}",
                "emoji": emoji,
                "themes": themes,
            })

# ── Prediction History ───────────────────────────────────────────────────

if show_history and st.session_state.history:
    st.divider()
    st.markdown(f"#### Prediction History ({len(st.session_state.history)} items)")
    hist_df = pd.DataFrame(st.session_state.history)
    if not hist_df.empty:
        st.dataframe(hist_df, use_container_width=True)

# ── Live Dashboard ───────────────────────────────────────────────────────

if show_dashboard and st.session_state.history:
    st.divider()
    st.markdown("#### Dashboard")
    hist = st.session_state.history
    label_counts = pd.Series([h["label"] for h in hist]).value_counts().reset_index()
    label_counts.columns = ["Sentiment", "Count"]

    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        fig = px.pie(label_counts, values="Count", names="Sentiment",
                     color="Sentiment", color_discrete_map={
                         "Negative": "#dc3545", "Neutral": "#fd7e14", "Positive": "#28a745"},
                     title="Sentiment Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col_chart2:
        theme_flat = []
        for h in hist:
            for t in h.get("themes", []):
                theme_flat.append(t)
        if theme_flat:
            theme_counts = pd.Series(theme_flat).value_counts().reset_index()
            theme_counts.columns = ["Theme", "Count"]
            fig2 = px.bar(theme_counts, x="Theme", y="Count", color="Theme",
                          title="Theme Breakdown")
            st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.caption("Sentiment Analyzer Pro  |  Airline Model + General Model  |  Built with DistilBERT & Streamlit")

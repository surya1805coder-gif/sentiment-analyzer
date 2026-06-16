<div align="center">
  <h1>Sentiment Analyzer Pro</h1>
  <p>Fine-tuned DistilBERT for Twitter airline sentiment · General-purpose fallback · Active learning</p>
  <p>
    <img src="https://img.shields.io/badge/Python-3.11%2B-blue" />
    <img src="https://img.shields.io/badge/Framework-Streamlit-red" />
    <img src="https://img.shields.io/badge/Model-DistilBERT-orange" />
    <img src="https://img.shields.io/badge/License-MIT-green" />
  </p>
</div>

---

## Features

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Airline Model** | DistilBERT fine-tuned on 14,640 airline tweets — Negative / Neutral / Positive |
| 2 | **General Model** | Zero-shot sentiment pipeline for any text (toggle in sidebar) |
| 3 | **Word Highlighting** | Sentiment-driving words highlighted green (positive) or red (negative) |
| 4 | **Theme Extraction** | Auto-tags tweets: Baggage, Delay, Refund, Customer Service, Food, Seating |
| 5 | **Batch CSV Upload** | Upload a CSV with a `text` column, get all predictions at once |
| 6 | **Confidence Warnings** | Flags predictions below 60% confidence |
| 7 | **Live Dashboard** | Plotly pie chart + theme breakdown (updates in real-time) |
| 8 | **Active Learning** | "Was this correct?" feedback saved to `feedback.jsonl` for future retraining |
| 9 | **Prediction History** | Session-based log of all predictions |

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

> The fine-tuned model is **not included in git** (267MB). Train your own using Colab or download from [Releases](https://github.com/surya1805coder-gif/sentiment-analyzer/releases).

---

## Training

### Option A: Colab (recommended — free GPU, 15 min)

Open and run: [`notebooks/colab_training.ipynb`](notebooks/colab_training.ipynb)

Download the model zip → extract to `models/sentiment_model/`

### Option B: Local (CPU, 7.5 hours)

```bash
python main.py
python main.py --epochs 5 --batch-size 16
```

### Option C: Override config

```bash
SENTIMENT_BATCH_SIZE=16 SENTIMENT_EPOCHS=5 python main.py
```

---

## App

```bash
streamlit run app/streamlit_app.py
```
Then open **http://localhost:8501** (or :8502 if port changed).

---

## Project Structure

```
├── main.py                     # Training entry point with argparse
├── configs/config.yaml         # All hyperparams (env-overridable)
├── requirements.txt
├── Dockerfile
├── Makefile
├── .gitignore
│
├── src/
│   ├── config.py               # YAML loader + device detection + env overrides
│   ├── utils.py                # Logging, seed setting, metadata persistence
│   ├── data/
│   │   ├── loader.py           # Data loading, cleaning, train/test split
│   │   └── preprocessor.py     # Tokenization & Hugging Face Dataset construction
│   ├── model/
│   │   └── architecture.py     # DistilBERT loading with label mappings
│   └── training/
│       ├── trainer.py          # HF Trainer setup, weighted metrics, early stopping
│       └── evaluator.py        # Classification report, confusion matrix
│
├── app/
│   └── streamlit_app.py        # Streamlit UI (all 9 features)
│
├── tests/
│   └── test_data.py            # Unit tests for data pipeline
│
├── notebooks/
│   └── colab_training.ipynb    # One-click Colab training notebook
│
└── scripts/
    ├── train.ps1               # Windows PowerShell helper
    └── app.ps1                 # Windows app launcher
```

---

## Tech Stack

| Component | Tool |
|-----------|------|
| **Fine-tuning** | Hugging Face Transformers + Trainer API |
| **Inference** | PyTorch + Transformers pipeline |
| **UI** | Streamlit |
| **Visualization** | Plotly |
| **Dataset** | Twitter US Airline Sentiment (Hugging Face) |
| **ML Platform** | Google Colab (T4 GPU) |
| **Container** | Docker |

---

## Model Details

- **Architecture:** DistilBERT (66M params, 97% BERT performance, 60% smaller)
- **Training data:** 14,640 tweets (11,712 train / 2,928 validation)
- **Labels:** Negative, Neutral, Positive
- **Metrics tracked:** Accuracy, weighted F1, precision, recall

### Why DistilBERT?

| Metric | BERT-Base | DistilBERT |
|--------|-----------|------------|
| Parameters | 110M | **66M** |
| Inference speed | 1x | **1.6x** |
| Language understanding | 100% | **97%** |

---

## Weaknesses

- **Airline-only domain** — general text (philosophy, food, etc.) gets low-confidence predictions. Use "General Model" toggle for those.
- **2015 dataset** — trained on tweets from 2015; may miss modern slang.
- **CPU training** — 7.5 hours locally; Colab GPU does it in 15 minutes.

---

## License

MIT

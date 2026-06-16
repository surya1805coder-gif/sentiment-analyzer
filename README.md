<p align="center">
  <h1 align="center">Sentiment Analyzer</h1>
  <p align="center">
    Fine-tune DistilBERT on airline tweets &middot; Serve predictions via Streamlit &middot; Active learning feedback
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.11%2B-blue" alt="Python 3.11+" />
    <img src="https://img.shields.io/badge/Model-DistilBERT-orange" alt="DistilBERT" />
    <img src="https://img.shields.io/badge/UI-Streamlit-red" alt="Streamlit" />
    <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT License" />
  </p>
</p>

---

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Open `http://localhost:8501`. The app works immediately with the general-purpose model.  
For the airline-specific model, train on Colab (see [Training](#training)).

---

## Features

- **Airline model** — DistilBERT fine-tuned on 14,640 tweets (Negative / Neutral / Positive)
- **General model** — Switch to a zero-shot pipeline for any non-airline text
- **Word highlighting** — Positive words in green, negative words in red
- **Theme extraction** — Auto-tags tweets: Baggage, Delay, Refund, Customer Service, Food, Seating
- **Batch CSV upload** — Upload a CSV with a `text` column and get bulk predictions
- **Confidence warnings** — Flags predictions below 60%
- **Live dashboard** — Plotly pie chart and theme breakdown (updates in real-time)
- **Active learning** — "Was this correct?" Yes/No feedback saved to `feedback.jsonl`
- **Prediction history** — Session-based log of all predictions

---

## Usage

### Single prediction

Type or paste text and click **Analyze**. View sentiment, confidence score, highlighted words, and detected theme.

### Batch upload

Upload a CSV with a `text` column. Results appear in a table with download option.

### Model toggle

In the sidebar, switch between **Airline Model** (fine-tuned) and **General Model** (zero-shot). Use general for non-airline text.

---

## Training

### Option A: Colab (recommended, free GPU, ~15 min)

Run [`notebooks/colab_training.ipynb`](notebooks/colab_training.ipynb) on Google Colab with a T4 GPU.  
Download the resulting `sentiment_model.zip` and extract to `sentiment_model/`.

### Option B: Local (CPU only, ~7.5 hours)

```bash
python main.py
python main.py --epochs 5 --batch-size 16
```

### Option C: Environment overrides

```bash
SENTIMENT_BATCH_SIZE=32 SENTIMENT_EPOCHS=3 python main.py
```

---

## Project Structure

```
├── main.py                     Training entry point (argparse)
├── configs/
│   └── config.yaml             Hyperparameters and paths
├── requirements.txt
├── Dockerfile
├── Makefile
├── .gitignore
│
├── src/
│   ├── config.py               YAML loader + device detection + env overrides
│   ├── utils.py                Logging, seed setting, metadata persistence
│   ├── data/
│   │   ├── loader.py           Data loading, cleaning, train/test split
│   │   └── preprocessor.py     Tokenization and dataset construction
│   ├── model/
│   │   └── architecture.py     DistilBERT loading with label mappings
│   └── training/
│       ├── trainer.py          Trainer setup, metrics, early stopping
│       └── evaluator.py        Classification report, confusion matrix
│
├── app/
│   └── streamlit_app.py        Streamlit UI
│
├── tests/
│   └── test_data.py            Unit tests for data pipeline
│
├── notebooks/
│   └── colab_training.ipynb    One-click Colab training notebook
│
└── scripts/
    ├── train.ps1               Windows PowerShell helper
    └── app.ps1                 Windows app launcher
```

---

## Configuration

Edit `configs/config.yaml` or set environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `SENTIMENT_BATCH_SIZE` | `16` | Training batch size |
| `SENTIMENT_EPOCHS` | `3` | Number of training epochs |
| `SENTIMENT_LEARNING_RATE` | `2e-5` | Learning rate |
| `SENTIMENT_MAX_LENGTH` | `128` | Max token length |
| `SENTIMENT_MODEL_NAME` | `distilbert-base-uncased` | Base model |

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Fine-tuning | Hugging Face Transformers + Trainer API |
| Inference | PyTorch + Transformers pipeline |
| UI | Streamlit |
| Visualization | Plotly |
| Dataset | Twitter US Airline Sentiment (Hugging Face) |
| Training platform | Google Colab (T4 GPU) |
| Container | Docker |

---

## FAQ

**The airline model gives low confidence on non-airline text.**  
Use the **General Model** toggle in the sidebar for general text.

**The fine-tuned model isn't included in the repo.**  
It's 267MB — exceeds GitHub's limit. Train on Colab or download from Releases.

**How do I contribute?**  
Open an issue or submit a PR. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## License

MIT

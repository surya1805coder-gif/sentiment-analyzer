# Real-Time Sentiment Analyzer

Fine-tune **DistilBERT** on Twitter US Airline Sentiment and serve predictions via a Streamlit UI.

## Project Structure

```
├── main.py                     # Entry point — training pipeline
├── configs/
│   └── config.yaml             # Hyperparameters and paths
├── src/
│   ├── config.py               # YAML loader with env overrides
│   ├── utils.py                # Logging, seeds, metadata
│   ├── data/
│   │   ├── loader.py           # Data loading, cleaning, splitting
│   │   └── preprocessor.py     # Tokenization and dataset construction
│   ├── model/
│   │   └── architecture.py     # DistilBERT loading with label mappings
│   └── training/
│       ├── trainer.py          # Trainer setup, metrics, early stopping
│       └── evaluator.py        # Classification report, confusion matrix
├── app/
│   └── streamlit_app.py        # Streamlit UI
├── tests/
│   └── test_data.py            # Unit tests for data pipeline
├── scripts/
│   ├── train.ps1               # Windows training helper
│   └── app.ps1                 # Windows app launcher
├── configs/
│   └── config.yaml
├── Makefile
├── Dockerfile
├── requirements.txt
└── .gitignore
```

## Quick Start

```bash
pip install -r requirements.txt
python main.py                              # train model
streamlit run app/streamlit_app.py          # launch UI
```

Override training params:

```bash
python main.py --epochs 5 --batch-size 16
```

## Tests

```bash
python -m pytest tests/ -v
```

## Docker

```bash
docker build -t sentiment-analyzer .
docker run -p 8501:8501 sentiment-analyzer
```

## Model

- **Architecture:** DistilBERT (66M params, 97% of BERT performance, 60% smaller)
- **Dataset:** Twitter US Airline Sentiment (14,640 tweets)
- **Classes:** Negative, Neutral, Positive
- **Metrics:** Accuracy, weighted F1, precision, recall

## Why DistilBERT?

| Aspect | BERT-Base | DistilBERT |
|---|---|---|
| Parameters | 110M | 66M |
| Speed | 1x | 1.6x |
| Performance | 100% | 97% |

DistilBERT gives near-BERT quality at significantly lower latency — ideal for real-time inference.

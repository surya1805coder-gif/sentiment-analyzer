"""
Data loading, cleaning, and splitting pipeline for Twitter US Airline Sentiment.
"""

import re
import logging
from typing import Tuple, List, Dict
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

LABEL_MAP: Dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL: Dict[int, str] = {v: k for k, v in LABEL_MAP.items()}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def load_data(source: str) -> pd.DataFrame:
    logger.info("Loading dataset from %s", source)

    if source.endswith(".csv") or source.startswith("http"):
        df = pd.read_csv(source, usecols=["text", "airline_sentiment"])
    else:
        from datasets import load_dataset
        ds = load_dataset(source, split="train")
        df = ds.to_pandas()[["text", "airline_sentiment"]]

    logger.info("Loaded %d rows", len(df))
    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["text", "airline_sentiment"])
    df["clean_text"] = df["text"].astype(str).apply(clean_text)
    df["label"] = df["airline_sentiment"].map(LABEL_MAP)
    df = df.dropna(subset=["label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[str], List[str], List[int], List[int]]:
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df["clean_text"].tolist(),
        df["label"].tolist(),
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    logger.info(
        "Split: train=%d, val=%d, distribution=%s",
        len(train_texts),
        len(val_texts),
        df["airline_sentiment"].value_counts().to_dict(),
    )
    return train_texts, val_texts, train_labels, val_labels


def load_and_split(
    source: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[str], List[str], List[int], List[int]]:
    df = load_data(source)
    df = prepare_data(df)
    return split_data(df, test_size, random_state)

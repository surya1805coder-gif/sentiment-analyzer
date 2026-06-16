"""
Tokenization wrapper for Hugging Face tokenizers with configurable truncation/padding.
"""

import logging
from typing import Dict, Any
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from datasets import Dataset

logger = logging.getLogger(__name__)

Tokenizer = PreTrainedTokenizer | PreTrainedTokenizerFast


def tokenize_texts(
    tokenizer: Tokenizer,
    texts: list[str],
    max_length: int = 128,
) -> Dict[str, list]:
    return tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )


def to_dataset(
    encodings: Dict[str, list],
    labels: list[int],
) -> Dataset:
    return Dataset.from_dict({
        "input_ids": encodings["input_ids"],
        "attention_mask": encodings["attention_mask"],
        "labels": labels,
    })

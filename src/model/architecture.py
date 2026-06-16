"""
Model architecture — DistilBERT for sequence classification with label mappings.
"""

import logging
from typing import Dict
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizer,
    PreTrainedModel,
)

logger = logging.getLogger(__name__)

ID2LABEL: Dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID: Dict[str, int] = {v: k for k, v in ID2LABEL.items()}


def build_model(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 3,
    device: str = "cpu",
    cache_dir: str | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    logger.info("Loading model: %s on %s", model_name, device)
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        cache_dir=cache_dir,
    )
    model.to(torch.device(device))
    model.eval()
    logger.info("Model loaded: %s parameters", sum(p.numel() for p in model.parameters()))
    return model, tokenizer

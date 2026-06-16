"""
Training orchestration — Trainer setup, metrics, and early stopping.
"""

import logging
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    PreTrainedModel,
)
from datasets import Dataset

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


def build_training_args(
    output_dir: str = "./models",
    batch_size: int = 8,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    fp16: bool = False,
    logging_steps: int = 50,
    save_strategy: str = "epoch",
    eval_strategy: str = "epoch",
    load_best: bool = True,
    metric_for_best: str = "f1",
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=f"{output_dir}/logs",
        logging_steps=logging_steps,
        eval_strategy=eval_strategy,
        save_strategy=save_strategy,
        load_best_model_at_end=load_best,
        metric_for_best_model=metric_for_best,
        greater_is_better=True,
        fp16=fp16,
        report_to="none",
        save_total_limit=2,
    )


def create_trainer(
    model: PreTrainedModel,
    train_dataset: Dataset,
    val_dataset: Dataset,
    args: TrainingArguments,
    early_stopping_patience: int = 2,
) -> Trainer:
    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

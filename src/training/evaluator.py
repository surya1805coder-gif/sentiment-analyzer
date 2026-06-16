"""
Post-training evaluation and classification report generation.
"""

import logging
import json
from pathlib import Path
from typing import Dict
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from transformers import Trainer

logger = logging.getLogger(__name__)


def evaluate(trainer: Trainer, output_dir: str) -> Dict[str, float]:
    logger.info("Running evaluation...")
    results = trainer.evaluate()
    logger.info("Evaluation results:")
    for k, v in results.items():
        logger.info("  %s: %.4f", k, v)
    return results


def generate_classification_report(
    trainer: Trainer,
    dataset,
    label_names: list[str],
    output_dir: str,
) -> str:
    predictions = trainer.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    report = classification_report(
        predictions.label_ids,
        preds,
        target_names=label_names,
        digits=4,
    )
    logger.info("Classification report:\n%s", report)
    out_path = Path(output_dir) / "classification_report.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    logger.info("Report saved to %s", out_path)
    return report


def save_confusion_matrix(
    trainer: Trainer,
    dataset,
    label_names: list[str],
    output_dir: str,
) -> np.ndarray:
    predictions = trainer.predict(dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    cm = confusion_matrix(predictions.label_ids, preds)
    report = {
        "confusion_matrix": cm.tolist(),
        "labels": label_names,
    }
    out_path = Path(output_dir) / "confusion_matrix.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    logger.info("Confusion matrix saved to %s", out_path)
    logger.info("Confusion matrix:\n%s", np.array2string(cm))
    return cm

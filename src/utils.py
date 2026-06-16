"""
Shared utilities — logging, seed setting, and model persistence.
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from typing import Any
import numpy as np
import torch


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(name)-28s | %(levelname)-6s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_metadata(
    model_dir: str,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    path = Path(model_dir) / "metadata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": {k: float(v) if isinstance(v, (np.floating,)) else v for k, v in metrics.items()},
        "config": config,
    }
    path.write_text(json.dumps(payload, indent=2))
    logging.getLogger(__name__).info("Metadata saved to %s", path)

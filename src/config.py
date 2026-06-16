"""
Project-wide configuration loaded from YAML with typed access and env override.
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml
import torch


ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "configs" / "config.yaml"


def load_config(path: Path = CONFIG_PATH) -> Dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cfg["model"]["cache_dir"] = str(ROOT_DIR / "models" / "cache")
    cfg["training"]["output_dir"] = str(ROOT_DIR / cfg["training"]["output_dir"])
    return cfg


def get_config() -> Dict[str, Any]:
    cfg = load_config()
    cfg = resolve_paths(cfg)
    cfg["device"] = get_device()
    if os.environ.get("SENTIMENT_BATCH_SIZE"):
        cfg["training"]["batch_size"] = int(os.environ["SENTIMENT_BATCH_SIZE"])
    if os.environ.get("SENTIMENT_EPOCHS"):
        cfg["training"]["epochs"] = int(os.environ["SENTIMENT_EPOCHS"])
    return cfg

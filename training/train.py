"""Acoustic model training and fine-tuning runner.

Supports full baseline training, fine-tuning pruned models, and distillation.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from training.utils import set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("train")


def train_asr(config_path: Path) -> None:
    """Execute acoustic model training loop."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("experiment", {}).get("seed", 42)
    set_seed(seed)

    logger.info(f"Loaded training config from: {config_path}")
    logger.info(f"Experiment: {cfg.get('experiment', {}).get('name', 'unnamed')}")
    logger.info(f"Dataset: {cfg.get('dataset', {}).get('name', 'common_voice_swahili')}")

    # TODO: Initialize acoustic model architecture, optimizer, DataLoader, and training loop
    logger.info("Training pipeline initialized. Connect Swahili dataset to commence gradient updates.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train streaming speech recognition model.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_asr(Path(args.config))

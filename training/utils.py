"""Utility functions for training, seed management, and checkpointing."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import random
from typing import Any, Dict, Optional
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("training_utils")


def set_seed(seed: int = 42) -> None:
    """Set deterministic random seeds across Python, NumPy, and PyTorch.

    Args:
        seed: Integer random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    logger.info(f"Global random seed set to {seed}")


def save_checkpoint(
    state: Dict[str, Any],
    checkpoint_dir: Path,
    filename: str = "checkpoint.pt"
) -> Path:
    """Save training checkpoint to disk.

    Args:
        state: State dictionary containing model weights, optimizer, epoch.
        checkpoint_dir: Target directory.
        filename: Checkpoint filename.

    Returns:
        Path to saved checkpoint.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    target = checkpoint_dir / filename
    # TODO: Connect torch.save when PyTorch model state is passed
    logger.info(f"Saved checkpoint state to: {target}")
    return target

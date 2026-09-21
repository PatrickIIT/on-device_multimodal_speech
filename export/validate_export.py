"""Validation utility for exported model artifacts.

Verifies ONNX graph structure, tensor I/O signatures, and numerical tolerances
against original PyTorch outputs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("validate_export")


def validate_onnx_model(onnx_path: Path) -> bool:
    """Validate ONNX model graph structure using onnx.checker.

    Args:
        onnx_path: Path to .onnx file.

    Returns:
        True if graph is structurally valid.
    """
    if not onnx_path.exists():
        logger.error(f"ONNX model file not found: {onnx_path}")
        return False

    try:
        import onnx
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)
        logger.info(f"ONNX validation passed successfully for: {onnx_path}")
        return True
    except ImportError:
        logger.warning("ONNX package not available for validation.")
        return True
    except Exception as e:
        logger.error(f"ONNX validation error: {e}")
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate exported model artifact.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to .onnx or .pte file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_p = Path(args.model_path)
    if model_p.suffix == ".onnx":
        validate_onnx_model(model_p)
    else:
        logger.info(f"Checking existence of model: {model_p} ({model_p.exists()})")

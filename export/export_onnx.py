"""ONNX export pipeline for streaming speech models.

Converts PyTorch acoustic encoders/decoders to ONNX graphs with dynamic batching
and streaming acoustic chunk input shapes.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("export_onnx")


def export_to_onnx(
    model_checkpoint: Path,
    output_onnx_path: Path,
    opset_version: int = 17,
    chunk_size_ms: int = 400,
    sample_rate: int = 16000,
) -> Path:
    """Export PyTorch acoustic model to ONNX.

    Args:
        model_checkpoint: Path to trained PyTorch weights.
        output_onnx_path: Target .onnx path.
        opset_version: Target ONNX operator set version (default: 17).
        chunk_size_ms: Expected streaming chunk size in ms.
        sample_rate: Audio sampling rate in Hz.

    Returns:
        Path to exported ONNX model artifact.
    """
    output_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Exporting model from {model_checkpoint} -> {output_onnx_path} (Opset {opset_version})")

    chunk_samples = int((chunk_size_ms / 1000.0) * sample_rate)
    logger.info(f"Configured input chunk shape: [batch_size=1, samples={chunk_samples}]")

    # TODO: Connect torch.onnx.export(model, dummy_input, output_onnx_path, opset_version=opset_version)
    return output_onnx_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export streaming ASR model to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to input PyTorch checkpoint.")
    parser.add_argument("--output", type=str, default="models/exported/streaming_model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--chunk-ms", type=int, default=400)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_to_onnx(
        model_checkpoint=Path(args.checkpoint),
        output_onnx_path=Path(args.output),
        opset_version=args.opset,
        chunk_size_ms=args.chunk_ms,
    )

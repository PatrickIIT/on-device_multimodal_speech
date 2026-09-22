"""ASR Recognition Quality Benchmarking (WER / CER).

Evaluates speech recognition accuracy over manifest files using
Word Error Rate (WER) and Character Error Rate (CER).
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.metrics import cer, wer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("benchmark_asr")


def load_manifest(manifest_path: Union[str, Path]) -> List[Dict[str, str]]:
    """Load an evaluation manifest (TSV format).

    Expected manifest columns: audio_filepath, duration, text

    Args:
        manifest_path: Path to the TSV manifest.

    Returns:
        List of sample records.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found at {path}")

    records: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            records.append(row)
    return records


def evaluate_predictions(
    references: List[str],
    hypotheses: List[str]
) -> Dict[str, float]:
    """Compute aggregate WER and CER across reference-hypothesis pairs.

    Args:
        references: List of reference transcript strings.
        hypotheses: List of model prediction strings.

    Returns:
        Dictionary with 'wer', 'cer', and 'sample_count'.
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"Reference count ({len(references)}) does not match hypothesis count ({len(hypotheses)})."
        )

    calculated_wer = wer(references, hypotheses)
    calculated_cer = cer(references, hypotheses)

    return {
        "wer": round(calculated_wer, 6),
        "cer": round(calculated_cer, 6),
        "sample_count": len(references),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ASR Word & Character Error Rates.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to evaluation manifest TSV.")
    parser.add_argument("--model-checkpoint", type=str, default=None, help="Path to model checkpoint / ONNX.")
    parser.add_argument("--predictions-file", type=str, default=None, help="Pre-computed predictions TSV.")
    parser.add_argument("--output", type=str, default=None, help="Path to save evaluation summary JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Loading manifest from {args.manifest}...")
    try:
        manifest_data = load_manifest(args.manifest)
        logger.info(f"Loaded {len(manifest_data)} samples from manifest.")
        # TODO: Connect model inference pipeline when checkpoint is provided.
        logger.info("Ready for model evaluation.")
    except Exception as e:
        logger.warning(f"Note: {e}")

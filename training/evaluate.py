"""Acoustic model offline evaluation script.

Evaluates an acoustic model on test split manifests and reports WER and CER.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from benchmarks.benchmark_asr import evaluate_predictions, load_manifest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("evaluate")


def run_evaluation(manifest_path: str, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
    """Evaluate speech model on manifest test set."""
    logger.info(f"Evaluating manifest: {manifest_path}")
    manifest = Path(manifest_path)
    if not manifest.exists():
        logger.warning(f"Manifest {manifest_path} does not exist yet. Returning empty metrics.")
        return {"wer": None, "cer": None, "sample_count": 0}

    samples = load_manifest(manifest)
    logger.info(f"Loaded {len(samples)} evaluation samples.")

    # TODO: Connect model checkpoint inference to generate hypotheses
    return {"wer": None, "cer": None, "sample_count": len(samples)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ASR model checkpoint.")
    parser.add_argument("--manifest", type=str, default="data/manifests/cv_sw_test.tsv")
    parser.add_argument("--checkpoint", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    results = run_evaluation(args.manifest, args.checkpoint)
    logger.info(f"Evaluation summary: {results}")

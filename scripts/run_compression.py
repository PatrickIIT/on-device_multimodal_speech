"""Compression runner script.

Executes quantization, structured pruning, or distillation workflows
based on configuration YAML specifications.
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

from compression.compression_pipeline import CompressionPipeline
from compression.distillation import DistillationConfig
from compression.pruning import PruningConfig
from compression.quantization import QuantizationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("run_compression")


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute model compression pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to compression config YAML.")
    parser.add_argument("--model-name", type=str, default="streaming_conformer_swahili")
    parser.add_argument("--input-checkpoint", type=str, default="models/checkpoints/baseline_fp32.pt")
    parser.add_argument("--output-dir", type=str, default="models/exported")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_yaml(cfg_path)
    logger.info(f"Loaded compression configuration from {cfg_path}")

    # Build stage configurations if present in YAML
    quant_cfg = None
    if "quantization" in cfg:
        q = cfg["quantization"]
        quant_cfg = QuantizationConfig(
            precision=q.get("precision", "int8"),
            method=q.get("method", "ptq"),
            calibration_samples=q.get("calibration", {}).get("num_samples", 256),
        )

    prune_cfg = None
    if "pruning" in cfg:
        p = cfg["pruning"]
        prune_cfg = PruningConfig(
            granularity=p.get("granularity", "channel"),
            target_sparsity=p.get("sparsity_targets", [0.40])[0] if isinstance(p.get("sparsity_targets"), list) else 0.40,
        )

    distill_cfg = None
    if "distillation" in cfg:
        d = cfg["distillation"]
        distill_cfg = DistillationConfig(
            temperature=d.get("loss_weights", {}).get("temperature", 2.0),
            alpha_ce=d.get("loss_weights", {}).get("alpha_ce", 0.5),
            alpha_kd=d.get("loss_weights", {}).get("alpha_kd", 0.5),
        )

    pipeline = CompressionPipeline(
        quant_config=quant_cfg,
        prune_config=prune_cfg,
        distill_config=distill_cfg,
    )

    profile = pipeline.run_pipeline(
        model_name=args.model_name,
        input_checkpoint=args.input_checkpoint,
        output_dir=args.output_dir,
    )

    logger.info(f"Compression pipeline successfully configured model profile: {profile}")


if __name__ == "__main__":
    main()

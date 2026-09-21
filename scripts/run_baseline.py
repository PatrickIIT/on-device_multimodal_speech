"""Baseline execution script for uncompressed FP32 streaming Swahili ASR.

Loads `configs/baseline.yaml`, initializes model harness, and profiles baseline metrics.
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

from benchmarks.benchmark import create_empty_benchmark_record, save_benchmark_record

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("run_baseline")


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_baseline_pipeline(config_path: str, output_path: str, dry_run: bool = False) -> None:
    """Execute baseline evaluation pipeline."""
    cfg = load_yaml_config(Path(config_path))
    exp_name = cfg.get("experiment", {}).get("name", "E0_baseline")
    device = cfg.get("hardware", {}).get("device", "samsung_a13")

    logger.info(f"=== Running Baseline Experiment: {exp_name} on {device} ===")

    record = create_empty_benchmark_record(
        experiment=exp_name,
        model=cfg.get("model", {}).get("name", "TBD"),
        dataset=cfg.get("dataset", {}).get("name", "common_voice_swahili"),
        device=device,
        precision="fp32",
        chunk_ms=cfg.get("streaming", {}).get("chunk_ms", 400),
    )

    if dry_run:
        logger.info("Dry-run execution completed. Baseline template created.")
    else:
        logger.info(
            "Ready for baseline execution. "
            "To capture empirical measurements, connect the trained Swahili checkpoint and target hardware."
        )

    out = save_benchmark_record(record, output_path)
    logger.info(f"Baseline result record saved to {out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run uncompressed FP32 baseline experiment.")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to config YAML.")
    parser.add_argument("--output", type=str, default="results/raw/E0_baseline_samsung_a13.json")
    parser.add_argument("--dry-run", action="store_true", help="Generate template record without running inference.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_baseline_pipeline(args.config, args.output, dry_run=args.dry_run)

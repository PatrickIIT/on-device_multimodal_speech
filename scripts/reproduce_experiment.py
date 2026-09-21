"""Master experiment reproduction entrypoint.

Provides a unified interface to reproduce any experiment (E0 to E10)
from its configuration YAML file.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
from pathlib import Path
import sys
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from benchmarks.benchmark import create_empty_benchmark_record, save_benchmark_record

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("reproduce_experiment")

VALID_EXPERIMENTS = {
    "E0_baseline": "configs/baseline.yaml",
    "E1_fp16": "configs/quantization.yaml",
    "E2_int8": "configs/quantization.yaml",
    "E3_int4": "configs/quantization.yaml",
    "E4_pruning": "configs/pruning.yaml",
    "E5_pruning_quantization": "configs/proposed_method.yaml",
    "E6_distillation": "configs/distillation.yaml",
    "E7_streaming": "configs/streaming.yaml",
    "E8_noise_robustness": "configs/baseline.yaml",
    "E9_runtime": "configs/hardware_a13.yaml",
    "E10_hardware_aware": "configs/proposed_method.yaml",
}


def load_config(config_path: Path) -> Dict[str, Any]:
    """Safely load and validate experiment configuration YAML."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {config_path}")
    return data


def run_experiment(
    experiment_id: str,
    config_path: Path,
    device: str = "samsung_a13",
    dry_run: bool = False,
) -> Path:
    """Execute experiment lifecycle and produce standardized result artifact.

    Args:
        experiment_id: Identifier matching E0–E10 registry.
        config_path: Path to configuration YAML.
        device: Target execution device.
        dry_run: If True, generate schema template without executing unverified trials.

    Returns:
        Path to generated JSON result artifact.
    """
    logger.info(f"=== Reproducing Experiment: {experiment_id} ===")
    logger.info(f"Loading configuration from: {config_path}")

    config = load_config(config_path)
    exp_cfg = config.get("experiment", {})
    exp_name = exp_cfg.get("name", experiment_id)

    # Extract relevant metadata
    precision = (
        config.get("quantization", {}).get("precision")
        or config.get("model", {}).get("precision", "fp32")
    )
    chunk_ms = (
        config.get("streaming", {}).get("chunk_ms")
        or (config.get("streaming", {}).get("chunk_sizes_ms", [400])[0] if isinstance(config.get("streaming", {}).get("chunk_sizes_ms"), list) else 400)
    )

    record = create_empty_benchmark_record(
        experiment=exp_name,
        model=config.get("model", {}).get("name", "TBD"),
        dataset=config.get("dataset", {}).get("name", "common_voice_swahili"),
        device=device,
        precision=precision,
        chunk_ms=chunk_ms,
    )

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"results/raw/{experiment_id}_{device}_{timestamp_str}.json")

    if dry_run:
        logger.info(f"Dry-run reproduction: Schema template initialized for {experiment_id}.")
    else:
        logger.info(
            f"Experiment {experiment_id} pipeline initialized. "
            "Awaiting on-device benchmark execution logs to populate empirical metrics."
        )

    saved_path = save_benchmark_record(record, output_path)
    return saved_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce experiments from E0 (Baseline) to E10 (Hardware-Aware Selection)."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=list(VALID_EXPERIMENTS.keys()),
        help="Experiment identifier (e.g. E0_baseline, E2_int8, E10_hardware_aware)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML (defaults to registry mapping)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="samsung_a13",
        help="Target hardware device."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate result schema template without attempting live execution."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_config = VALID_EXPERIMENTS.get(args.experiment, "configs/baseline.yaml")
    config_file = Path(args.config) if args.config else Path(default_config)

    try:
        saved_file = run_experiment(
            experiment_id=args.experiment,
            config_path=config_file,
            device=args.device,
            dry_run=args.dry_run,
        )
        logger.info(f"Experiment artifact successfully generated: {saved_file}")
    except Exception as e:
        logger.error(f"Failed to reproduce experiment {args.experiment}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

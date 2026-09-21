"""Main benchmark entrypoint and schema manager.

This script manages end-to-end benchmark runs, recording structured
experimental metrics into standardized JSON schemas.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("benchmark")


def create_empty_benchmark_record(
    experiment: str = "",
    model: str = "",
    dataset: str = "",
    device: str = "",
    precision: str = "",
    sparsity: Optional[float] = None,
    chunk_ms: Optional[int] = None,
) -> Dict[str, Any]:
    """Create a standardized benchmark record initialized with null measurements.

    Args:
        experiment: Experiment identifier (e.g., 'E0_baseline').
        model: Model identifier or architecture name.
        dataset: Dataset identifier or split name.
        device: Target execution device.
        precision: Arithmetic precision (e.g., 'fp32', 'int8').
        sparsity: Sparsity fraction if pruned.
        chunk_ms: Streaming chunk size in milliseconds.

    Returns:
        Dictionary following the standard research benchmark schema.
    """
    return {
        "experiment": experiment,
        "model": model,
        "dataset": dataset,
        "device": device,
        "precision": precision,
        "sparsity": sparsity,
        "chunk_ms": chunk_ms,
        "wer": None,
        "cer": None,
        "model_size_mb": None,
        "parameters": None,
        "flops": None,
        "latency_mean_ms": None,
        "latency_p50_ms": None,
        "latency_p90_ms": None,
        "latency_p95_ms": None,
        "ram_peak_mb": None,
        "cpu_percent": None,
        "rtf": None,
        "energy_wh": None,
        "thermal_state": "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def save_benchmark_record(record: Dict[str, Any], output_path: Union[str, Path]) -> Path:
    """Save a benchmark record to disk in JSON format.

    Args:
        record: Benchmark record dictionary.
        output_path: Target path for the JSON artifact.

    Returns:
        Path to the saved JSON file.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)
    logger.info(f"Saved benchmark schema record to: {out_path}")
    return out_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for benchmark execution."""
    parser = argparse.ArgumentParser(
        description="Unified benchmarking framework for on-device streaming speech recognition."
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="E0_baseline",
        help="Experiment identifier (e.g. E0_baseline, E2_int8)."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TBD",
        help="Model path or architecture name."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="common_voice_swahili",
        help="Dataset name or path to evaluation manifest."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="samsung_a13",
        help="Target execution hardware (e.g. samsung_a13, desktop_cpu)."
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "int8", "int4", "mixed"],
        help="Model precision."
    )
    parser.add_argument(
        "--sparsity",
        type=float,
        default=None,
        help="Model structured sparsity fraction (e.g. 0.20, 0.40)."
    )
    parser.add_argument(
        "--chunk-ms",
        type=int,
        default=None,
        help="Streaming chunk duration in milliseconds."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON file (default: results/raw/<experiment>_<device>_<timestamp>.json)."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate empty benchmark template without invoking actual inference."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Initializing benchmark framework...")
    logger.info(f"Target Device: {args.device} | Experiment: {args.experiment} | Precision: {args.precision}")

    record = create_empty_benchmark_record(
        experiment=args.experiment,
        model=args.model,
        dataset=args.dataset,
        device=args.device,
        precision=args.precision,
        sparsity=args.sparsity,
        chunk_ms=args.chunk_ms,
    )

    if args.dry_run:
        logger.info("Dry-run mode: Creating benchmark schema template without executing unverified trials.")
    else:
        logger.info(
            "Benchmark pipeline ready for real-device measurement runs. "
            "Metrics remain null until empirical execution logs are captured."
        )

    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_path = args.output or f"results/raw/{args.experiment}_{args.device}_{timestamp_str}.json"
    save_benchmark_record(record, output_path)


if __name__ == "__main__":
    main()

"""Inference latency benchmarking harness.

Measures cold-start, warm-start, and percentiles (P50, P90, P95) of model execution.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
import sys
from typing import Any, Callable, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.metrics import latency_statistics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("benchmark_latency")


def measure_execution_latencies(
    inference_fn: Callable[[], Any],
    warmup_runs: int = 10,
    benchmark_runs: int = 50
) -> Dict[str, Any]:
    """Profile cold-start, warm-start, and statistical latency percentiles.

    Args:
        inference_fn: Callable executing a single inference pass.
        warmup_runs: Iterations for warming up caches/JIT/DVFS.
        benchmark_runs: Measured iterations for statistical reporting.

    Returns:
        Structured dictionary with cold-start latency and latency distributions.
    """
    logger.info(f"Running latency benchmark: warmup={warmup_runs}, runs={benchmark_runs}")

    # Cold start: 1st execution
    t0 = time.perf_counter()
    inference_fn()
    t1 = time.perf_counter()
    cold_start_ms = (t1 - t0) * 1000.0

    # Warmup runs
    for _ in range(max(0, warmup_runs - 1)):
        inference_fn()

    # Timed benchmark runs
    measured_latencies_ms: List[float] = []
    for _ in range(benchmark_runs):
        start_t = time.perf_counter()
        inference_fn()
        end_t = time.perf_counter()
        measured_latencies_ms.append((end_t - start_t) * 1000.0)

    stats = latency_statistics(measured_latencies_ms)
    stats["cold_start_ms"] = round(cold_start_ms, 4)
    stats["warmup_runs"] = warmup_runs
    stats["benchmark_runs"] = benchmark_runs

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure model inference latencies.")
    parser.add_argument("--warmup-runs", type=int, default=10)
    parser.add_argument("--benchmark-runs", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("Latency benchmark module ready.")

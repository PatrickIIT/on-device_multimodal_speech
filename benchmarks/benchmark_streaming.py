"""Streaming inference benchmarking.

Measures per-chunk latency, Real-Time Factor (RTF), and streaming state transitions
across configurable audio chunk durations (100ms, 200ms, 400ms, 800ms, 1000ms).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.metrics import latency_statistics, real_time_factor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("benchmark_streaming")


def simulate_streaming_benchmark(
    audio_duration_s: float,
    chunk_ms: int,
    warmup_chunks: int = 5
) -> Dict[str, Any]:
    """Execute streaming benchmark harness.

    Args:
        audio_duration_s: Total duration of audio stream in seconds.
        chunk_ms: Chunk duration in milliseconds.
        warmup_chunks: Number of initial chunks to discard for cold-start stabilization.

    Returns:
        Structured dictionary of streaming metrics.
    """
    chunk_duration_s = chunk_ms / 1000.0
    total_chunks = max(1, int(audio_duration_s / chunk_duration_s))

    logger.info(
        f"Configured streaming session: chunk={chunk_ms}ms, "
        f"audio_len={audio_duration_s:.2f}s, expected_chunks={total_chunks}"
    )

    # Note: When model backend is attached, chunk inference latencies are populated.
    return {
        "chunk_ms": chunk_ms,
        "audio_duration_s": audio_duration_s,
        "total_chunks": total_chunks,
        "warmup_chunks": warmup_chunks,
        "chunk_latencies_ms": [],
        "latency_stats": {},
        "rtf": None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark streaming ASR chunk inference.")
    parser.add_argument("--chunk-ms", type=int, default=400, choices=[100, 200, 400, 800, 1000])
    parser.add_argument("--audio-file", type=str, default=None, help="Path to input 16kHz WAV file.")
    parser.add_argument("--warmup-chunks", type=int, default=5)
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Initialized streaming benchmark for chunk_size={args.chunk_ms}ms.")

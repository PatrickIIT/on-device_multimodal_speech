"""Memory and RAM utilization benchmarking.

Measures Resident Set Size (RSS), Virtual Memory Size (VMS), and peak allocation
during streaming model initialization and active inference.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any, Dict, Optional

try:
    import psutil
except ImportError:
    psutil = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("benchmark_memory")


class MemoryTracker:
    """Tracks process memory consumption in Megabytes (MB)."""

    def __init__(self) -> None:
        if psutil is None:
            logger.warning("psutil is not installed. Memory tracking will return None.")
            self._process = None
        else:
            self._process = psutil.Process(os.getpid())
        self._initial_rss_mb: Optional[float] = None
        self._peak_rss_mb: Optional[float] = None

    def start(self) -> None:
        """Record baseline memory before test execution."""
        if self._process:
            mem_info = self._process.memory_info()
            self._initial_rss_mb = mem_info.rss / (1024 * 1024)
            self._peak_rss_mb = self._initial_rss_mb
        else:
            self._initial_rss_mb = None
            self._peak_rss_mb = None

    def update(self) -> None:
        """Poll current memory and update peak tracking."""
        if self._process:
            current_rss = self._process.memory_info().rss / (1024 * 1024)
            if self._peak_rss_mb is None or current_rss > self._peak_rss_mb:
                self._peak_rss_mb = current_rss

    def summary(self) -> Dict[str, Optional[float]]:
        """Return memory statistics dictionary."""
        self.update()
        delta = None
        if self._peak_rss_mb is not None and self._initial_rss_mb is not None:
            delta = round(self._peak_rss_mb - self._initial_rss_mb, 2)

        return {
            "initial_rss_mb": round(self._initial_rss_mb, 2) if self._initial_rss_mb else None,
            "peak_rss_mb": round(self._peak_rss_mb, 2) if self._peak_rss_mb else None,
            "delta_rss_mb": delta,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track process memory consumption.")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    tracker = MemoryTracker()
    tracker.start()
    time.sleep(0.1)
    logger.info(f"Memory tracker initialized: {tracker.summary()}")

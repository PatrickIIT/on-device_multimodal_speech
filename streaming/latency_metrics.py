"""Streaming-specific latency metrics.

Tracks First Chunk Latency (FCL), Chunk Processing Time (CPT),
and User-Perceived Latency (UPL) / End-of-Utterance Latency (EUL).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional
import numpy as np

from benchmarks.metrics import latency_statistics


@dataclass
class StreamingLatencyTracker:
    """Tracks latency metrics throughout an interactive streaming session."""

    chunk_durations_s: List[float] = field(default_factory=list)
    chunk_processing_times_s: List[float] = field(default_factory=list)
    first_chunk_latency_s: Optional[float] = None
    stream_start_time: Optional[float] = None
    last_chunk_arrival_time: Optional[float] = None
    final_token_emit_time: Optional[float] = None

    def start_stream(self) -> None:
        """Mark the beginning of an audio stream."""
        self.stream_start_time = time.perf_counter()

    def record_chunk(self, chunk_duration_s: float, processing_time_s: float) -> None:
        """Record the execution duration of a single streaming chunk."""
        if self.first_chunk_latency_s is None:
            self.first_chunk_latency_s = processing_time_s

        self.chunk_durations_s.append(chunk_duration_s)
        self.chunk_processing_times_s.append(processing_time_s)
        self.last_chunk_arrival_time = time.perf_counter()

    def mark_utterance_end(self) -> None:
        """Mark the timestamp when the final audio sample was submitted."""
        self.last_chunk_arrival_time = time.perf_counter()

    def mark_final_token_emitted(self) -> None:
        """Mark the timestamp when the final transcription token was returned."""
        self.final_token_emit_time = time.perf_counter()

    def summary(self) -> Dict[str, Optional[float]]:
        """Compute summary statistics for streaming latency."""
        if not self.chunk_processing_times_s:
            return {
                "first_chunk_latency_ms": None,
                "end_of_utterance_latency_ms": None,
                "mean_cpt_ms": None,
                "p95_cpt_ms": None,
                "stream_rtf": None,
            }

        cpt_ms = [t * 1000.0 for t in self.chunk_processing_times_s]
        stats = latency_statistics(cpt_ms)

        eul_ms = None
        if self.final_token_emit_time and self.last_chunk_arrival_time:
            eul_ms = max(0.0, (self.final_token_emit_time - self.last_chunk_arrival_time) * 1000.0)

        total_proc = sum(self.chunk_processing_times_s)
        total_audio = sum(self.chunk_durations_s)
        rtf = total_proc / total_audio if total_audio > 0 else None

        return {
            "first_chunk_latency_ms": round(self.first_chunk_latency_s * 1000.0, 2) if self.first_chunk_latency_s else None,
            "end_of_utterance_latency_ms": round(eul_ms, 2) if eul_ms is not None else None,
            "mean_cpt_ms": stats["mean"],
            "p50_cpt_ms": stats["p50"],
            "p95_cpt_ms": stats["p95"],
            "stream_rtf": round(rtf, 4) if rtf is not None else None,
        }

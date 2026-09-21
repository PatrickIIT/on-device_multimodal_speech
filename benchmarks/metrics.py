"""Evaluation metrics for speech recognition quality and hardware efficiency.

This module provides standard implementations for Word Error Rate (WER),
Character Error Rate (CER), Real-Time Factor (RTF), compression ratios,
and statistical distributions of inference latency.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Union


def _levenshtein_distance(seq1: Sequence[Any], seq2: Sequence[Any]) -> int:
    """Compute the Levenshtein edit distance between two sequences.

    Args:
        seq1: Reference sequence (tokens, words, or characters).
        seq2: Hypothesis sequence.

    Returns:
        Integer minimum edit distance (insertions + deletions + substitutions).
    """
    n, m = len(seq1), len(seq2)
    if n == 0:
        return m
    if m == 0:
        return n

    # Use two rows for memory-efficient dynamic programming
    current_row = list(range(n + 1))
    for j in range(1, m + 1):
        previous_row = current_row
        current_row = [j] + [0] * n
        for i in range(1, n + 1):
            substitution_cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            current_row[i] = min(
                previous_row[i] + 1,                  # Insertion
                current_row[i - 1] + 1,                # Deletion
                previous_row[i - 1] + substitution_cost  # Substitution
            )

    return current_row[n]


def wer(
    reference: Union[str, Sequence[str]],
    hypothesis: Union[str, Sequence[str]]
) -> float:
    """Compute Word Error Rate (WER).

    WER = (Substitutions + Deletions + Insertions) / Total Reference Words

    Args:
        reference: Reference transcript string or list of reference strings.
        hypothesis: Hypothesis transcript string or list of hypothesis strings.

    Returns:
        Word Error Rate as a float (0.0 represents perfect match).

    Raises:
        ValueError: If inputs are mismatched list lengths.
    """
    if isinstance(reference, str) and isinstance(hypothesis, str):
        ref_words = reference.strip().split()
        hyp_words = hypothesis.strip().split()

        if len(ref_words) == 0:
            return 0.0 if len(hyp_words) == 0 else 1.0

        dist = _levenshtein_distance(ref_words, hyp_words)
        return float(dist) / float(len(ref_words))

    elif isinstance(reference, (list, tuple)) and isinstance(hypothesis, (list, tuple)):
        if len(reference) != len(hypothesis):
            raise ValueError(
                f"Reference length ({len(reference)}) must match hypothesis length ({len(hypothesis)})."
            )

        total_dist = 0
        total_ref_words = 0

        for ref_item, hyp_item in zip(reference, hypothesis):
            ref_words = ref_item.strip().split()
            hyp_words = hyp_item.strip().split()
            total_dist += _levenshtein_distance(ref_words, hyp_words)
            total_ref_words += len(ref_words)

        if total_ref_words == 0:
            return 0.0

        return float(total_dist) / float(total_ref_words)

    else:
        raise TypeError("Reference and hypothesis must both be strings or sequences of strings.")


def cer(
    reference: Union[str, Sequence[str]],
    hypothesis: Union[str, Sequence[str]],
    ignore_whitespace: bool = False
) -> float:
    """Compute Character Error Rate (CER).

    CER = (Substitutions + Deletions + Insertions) / Total Reference Characters

    Args:
        reference: Reference transcript string or list of reference strings.
        hypothesis: Hypothesis transcript string or list of hypothesis strings.
        ignore_whitespace: If True, whitespace characters are stripped before comparison.

    Returns:
        Character Error Rate as a float.

    Raises:
        ValueError: If inputs are mismatched list lengths.
    """
    if isinstance(reference, str) and isinstance(hypothesis, str):
        ref_chars = [c for c in reference if not (ignore_whitespace and c.isspace())]
        hyp_chars = [c for c in hypothesis if not (ignore_whitespace and c.isspace())]

        if len(ref_chars) == 0:
            return 0.0 if len(hyp_chars) == 0 else 1.0

        dist = _levenshtein_distance(ref_chars, hyp_chars)
        return float(dist) / float(len(ref_chars))

    elif isinstance(reference, (list, tuple)) and isinstance(hypothesis, (list, tuple)):
        if len(reference) != len(hypothesis):
            raise ValueError(
                f"Reference length ({len(reference)}) must match hypothesis length ({len(hypothesis)})."
            )

        total_dist = 0
        total_ref_chars = 0

        for ref_item, hyp_item in zip(reference, hypothesis):
            ref_chars = [c for c in ref_item if not (ignore_whitespace and c.isspace())]
            hyp_chars = [c for c in hyp_item if not (ignore_whitespace and c.isspace())]
            total_dist += _levenshtein_distance(ref_chars, hyp_chars)
            total_ref_chars += len(ref_chars)

        if total_ref_chars == 0:
            return 0.0

        return float(total_dist) / float(total_ref_chars)

    else:
        raise TypeError("Reference and hypothesis must both be strings or sequences of strings.")


def real_time_factor(inference_time_s: float, audio_duration_s: float) -> float:
    """Calculate Real-Time Factor (RTF).

    RTF = inference_time / audio_duration
    An RTF < 1.0 indicates faster-than-real-time performance.

    Args:
        inference_time_s: Total inference time in seconds.
        audio_duration_s: Total audio duration in seconds.

    Returns:
        Real-Time Factor as a float.

    Raises:
        ValueError: If audio_duration_s is non-positive or inference_time_s is negative.
    """
    if inference_time_s < 0:
        raise ValueError(f"Inference time cannot be negative: {inference_time_s}")
    if audio_duration_s <= 0:
        raise ValueError(f"Audio duration must be strictly positive: {audio_duration_s}")

    return float(inference_time_s) / float(audio_duration_s)


def compression_ratio(uncompressed_size: float, compressed_size: float) -> float:
    """Calculate model compression ratio.

    Compression Ratio = uncompressed_size / compressed_size

    Args:
        uncompressed_size: Baseline uncompressed model size (bytes or MB).
        compressed_size: Compressed model size (bytes or MB).

    Returns:
        Compression ratio (e.g., 4.0 for a 4x reduction).

    Raises:
        ValueError: If sizes are non-positive.
    """
    if uncompressed_size <= 0:
        raise ValueError("Uncompressed size must be positive.")
    if compressed_size <= 0:
        raise ValueError("Compressed size must be positive.")

    return float(uncompressed_size) / float(compressed_size)


def _percentile(sorted_data: Sequence[float], percentile: float) -> float:
    """Compute exact or interpolated percentile from a sorted sequence."""
    if not sorted_data:
        raise ValueError("Cannot calculate percentile of empty sequence.")
    if not (0.0 <= percentile <= 100.0):
        raise ValueError("Percentile must be between 0.0 and 100.0.")

    k = (len(sorted_data) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_data[int(k)])
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return float(d0 + d1)


def latency_statistics(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """Calculate statistical distribution for latency measurements.

    Args:
        latencies_ms: Sequence of latency measurements in milliseconds.

    Returns:
        Dictionary containing mean, median, p50, p90, p95, min, and max latencies.

    Raises:
        ValueError: If latencies_ms is empty.
    """
    if not latencies_ms:
        raise ValueError("Latencies list cannot be empty.")

    sorted_latencies = sorted(float(x) for x in latencies_ms)
    n = len(sorted_latencies)

    mean_val = sum(sorted_latencies) / n
    min_val = sorted_latencies[0]
    max_val = sorted_latencies[-1]
    p50_val = _percentile(sorted_latencies, 50.0)
    p90_val = _percentile(sorted_latencies, 90.0)
    p95_val = _percentile(sorted_latencies, 95.0)

    return {
        "count": n,
        "mean": round(mean_val, 4),
        "median": round(p50_val, 4),
        "p50": round(p50_val, 4),
        "p90": round(p90_val, 4),
        "p95": round(p95_val, 4),
        "min": round(min_val, 4),
        "max": round(max_val, 4),
    }

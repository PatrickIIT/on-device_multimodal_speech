"""Audio stream chunking and buffer management.

Slices raw audio signals into discrete temporal chunks for streaming inference
across configurable windows (100ms, 200ms, 400ms, 800ms, 1000ms).
"""

from __future__ import annotations

import math
from typing import Generator, List, Optional, Sequence, Union
import numpy as np


class AudioChunker:
    """Manages audio buffer framing and causal chunk generation."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size_ms: int = 400,
        lookahead_ms: int = 0,
    ) -> None:
        """Initialize AudioChunker.

        Args:
            sample_rate: Audio sampling rate in Hz (default: 16000).
            chunk_size_ms: Chunk duration in milliseconds (e.g., 100, 200, 400, 800, 1000).
            lookahead_ms: Right-context lookahead duration in milliseconds.

        Raises:
            ValueError: If sample_rate or chunk_size_ms is non-positive.
        """
        if sample_rate <= 0:
            raise ValueError(f"Sample rate must be positive, got {sample_rate}")
        if chunk_size_ms <= 0:
            raise ValueError(f"Chunk size must be positive, got {chunk_size_ms}")
        if lookahead_ms < 0:
            raise ValueError(f"Lookahead cannot be negative, got {lookahead_ms}")

        self.sample_rate = sample_rate
        self.chunk_size_ms = chunk_size_ms
        self.lookahead_ms = lookahead_ms

        self.chunk_samples = int((chunk_size_ms / 1000.0) * sample_rate)
        self.lookahead_samples = int((lookahead_ms / 1000.0) * sample_rate)
        self.total_step_samples = self.chunk_samples

    def chunk_audio(
        self,
        audio: np.ndarray,
        pad_final_chunk: bool = False
    ) -> List[np.ndarray]:
        """Split a full 1D audio waveform into discrete chunks.

        Args:
            audio: 1D NumPy float or int array containing audio samples.
            pad_final_chunk: If True, pads the trailing chunk with zeros to match chunk_samples.

        Returns:
            List of 1D numpy arrays, each representing one chunk.
        """
        if audio.ndim != 1:
            raise ValueError(f"Expected 1D audio array, got shape {audio.shape}")

        total_samples = len(audio)
        if total_samples == 0:
            return []

        chunks: List[np.ndarray] = []
        start_idx = 0

        while start_idx < total_samples:
            end_idx = start_idx + self.chunk_samples + self.lookahead_samples
            chunk = audio[start_idx:end_idx]

            if len(chunk) < (self.chunk_samples + self.lookahead_samples) and pad_final_chunk:
                target_len = self.chunk_samples + self.lookahead_samples
                padded_chunk = np.zeros(target_len, dtype=audio.dtype)
                padded_chunk[:len(chunk)] = chunk
                chunk = padded_chunk

            chunks.append(chunk)
            start_idx += self.total_step_samples

        return chunks

    def stream_generator(
        self,
        audio: np.ndarray
    ) -> Generator[np.ndarray, None, None]:
        """Generator yielding chunks sequentially for streaming simulation."""
        for chunk in self.chunk_audio(audio, pad_final_chunk=False):
            yield chunk


class StreamBuffer:
    """Stateful ring buffer for incoming real-time audio samples."""

    def __init__(self, chunk_samples: int) -> None:
        self.chunk_samples = chunk_samples
        self._buffer: List[float] = []

    def push_samples(self, samples: Union[Sequence[float], np.ndarray]) -> None:
        """Append incoming audio samples to internal buffer."""
        if isinstance(samples, np.ndarray):
            self._buffer.extend(samples.tolist())
        else:
            self._buffer.extend(samples)

    def has_chunk(self) -> bool:
        """Check if at least one complete chunk is ready."""
        return len(self._buffer) >= self.chunk_samples

    def pop_chunk(self) -> Optional[np.ndarray]:
        """Extract one chunk if available, otherwise return None."""
        if not self.has_chunk():
            return None
        chunk_data = self._buffer[:self.chunk_samples]
        self._buffer = self._buffer[self.chunk_samples:]
        return np.array(chunk_data, dtype=np.float32)

    def clear(self) -> None:
        """Reset internal buffer."""
        self._buffer.clear()

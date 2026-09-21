"""Streaming ASR inference engine abstraction.

Provides a modular interface for causal, chunked speech decoding
independent of specific neural network frameworks or runtimes.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, List, Optional
import numpy as np

from streaming.chunking import AudioChunker, StreamBuffer
from streaming.vad import BaseVAD, EnergyVAD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("streaming_inference")


class StreamingASR(abc.ABC):
    """Abstract base class for streaming Automatic Speech Recognition models."""

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_size_ms: int = 400,
        vad: Optional[BaseVAD] = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.chunk_size_ms = chunk_size_ms
        self.chunker = AudioChunker(sample_rate=sample_rate, chunk_size_ms=chunk_size_ms)
        self.buffer = StreamBuffer(chunk_samples=self.chunker.chunk_samples)
        self.vad = vad or EnergyVAD()
        self._partial_hypotheses: List[str] = []

    @abc.abstractmethod
    def process_chunk(self, chunk: np.ndarray) -> str:
        """Process a single temporal chunk and update streaming internal state.

        Args:
            chunk: 1D NumPy array representing one acoustic chunk.

        Returns:
            Decoded partial token string for the chunk.
        """
        pass

    def accept_audio(self, audio_samples: np.ndarray) -> List[str]:
        """Feed incoming streaming audio samples into the engine.

        Pushes samples to internal buffer, pulls complete chunks, executes VAD,
        and decodes active chunks.

        Args:
            audio_samples: 1D NumPy array of incoming raw audio.

        Returns:
            List of newly emitted partial transcripts for processed chunks.
        """
        self.buffer.push_samples(audio_samples)
        emitted: List[str] = []

        while self.buffer.has_chunk():
            chunk = self.buffer.pop_chunk()
            if chunk is None:
                break

            # If VAD detects silence, skip heavy acoustic inference
            if not self.vad.is_speech(chunk):
                continue

            partial = self.process_chunk(chunk)
            if partial:
                self._partial_hypotheses.append(partial)
                emitted.append(partial)

        return emitted

    def decode(self) -> str:
        """Return the concatenated transcript decoded so far."""
        return " ".join(self._partial_hypotheses).strip()

    def reset(self) -> None:
        """Reset internal acoustic state, buffers, and transcript history."""
        self.buffer.clear()
        self.vad.reset()
        self._partial_hypotheses.clear()


class GenericStreamingASR(StreamingASR):
    """Reference / template streaming ASR engine for evaluation harnesses."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        sample_rate: int = 16000,
        chunk_size_ms: int = 400,
    ) -> None:
        super().__init__(sample_rate=sample_rate, chunk_size_ms=chunk_size_ms)
        self.model_path = model_path
        logger.info(
            f"Initialized GenericStreamingASR: model={model_path}, "
            f"chunk_size={chunk_size_ms}ms, sample_rate={sample_rate}Hz"
        )

    def process_chunk(self, chunk: np.ndarray) -> str:
        """Process chunk through backend neural acoustic encoder."""
        # TODO: Connect ONNX Runtime / Sherpa-ONNX / ExecuTorch inference session
        return ""

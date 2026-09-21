"""Voice Activity Detection (VAD) interface and baseline algorithms.

Provides frame-level and chunk-level speech detection to gate acoustic model inference
and minimize wasted CPU cycles during silence.
"""

from __future__ import annotations

import abc
import math
from typing import Optional, Sequence
import numpy as np


class BaseVAD(abc.ABC):
    """Abstract interface for Voice Activity Detection."""

    @abc.abstractmethod
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Determine whether an audio chunk contains active speech.

        Args:
            audio_chunk: 1D NumPy array of audio samples.

        Returns:
            True if speech is detected, False otherwise.
        """
        pass

    @abc.abstractmethod
    def reset(self) -> None:
        """Reset internal state machine."""
        pass


class EnergyVAD(BaseVAD):
    """Energy-based threshold VAD for lightweight mobile gating."""

    def __init__(
        self,
        energy_threshold: float = 0.01,
        min_speech_frames: int = 2,
        min_silence_frames: int = 5,
    ) -> None:
        """Initialize EnergyVAD.

        Args:
            energy_threshold: Root Mean Square (RMS) energy threshold.
            min_speech_frames: Consecutive speech frames required to transition to ACTIVE.
            min_silence_frames: Consecutive silence frames required to transition to INACTIVE.
        """
        self.energy_threshold = energy_threshold
        self.min_speech_frames = min_speech_frames
        self.min_silence_frames = min_silence_frames

        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._is_active = False

    def compute_rms(self, chunk: np.ndarray) -> float:
        """Compute Root Mean Square (RMS) energy of audio chunk."""
        if len(chunk) == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(chunk.astype(np.float64)))))

    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Classify chunk as speech or silence based on RMS energy."""
        rms = self.compute_rms(audio_chunk)
        if rms >= self.energy_threshold:
            self._consecutive_speech += 1
            self._consecutive_silence = 0
            if self._consecutive_speech >= self.min_speech_frames:
                self._is_active = True
        else:
            self._consecutive_silence += 1
            self._consecutive_speech = 0
            if self._consecutive_silence >= self.min_silence_frames:
                self._is_active = False

        return self._is_active

    def reset(self) -> None:
        """Reset state counters."""
        self._consecutive_speech = 0
        self._consecutive_silence = 0
        self._is_active = False

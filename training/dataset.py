"""Speech dataset and manifest loading utilities.

Provides data structures for streaming Swahili speech datasets,
spectrogram extraction, and batch collation.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("dataset")


class SpeechDataset:
    """Manages audio dataset manifests and item retrieval."""

    def __init__(
        self,
        manifest_path: Union[str, Path],
        sample_rate: int = 16000,
        max_duration_s: float = 20.0,
        min_duration_s: float = 0.5,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.sample_rate = sample_rate
        self.max_duration_s = max_duration_s
        self.min_duration_s = min_duration_s
        self.samples: List[Dict[str, str]] = []

        if self.manifest_path.exists():
            self._load_manifest()
        else:
            logger.warning(f"Manifest file not found at {self.manifest_path}. Initialized empty dataset.")

    def _load_manifest(self) -> None:
        """Parse TSV manifest."""
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                self.samples.append(row)
        logger.info(f"Loaded {len(self.samples)} samples from {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Fetch sample metadata and audio path."""
        sample = self.samples[idx]
        return {
            "audio_filepath": sample.get("audio_filepath", ""),
            "duration": float(sample.get("duration", 0.0)),
            "text": sample.get("text", ""),
            "speaker_id": sample.get("speaker_id", ""),
        }

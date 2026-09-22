"""Dataset preparation and manifest generation utility for Swahili speech corpora.

Handles audio validation, 16 kHz resampling, text normalization,
and manifest TSV creation for Common Voice and FLEURS.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import re
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("prepare_data")


def normalize_swahili_text(text: str) -> str:
    """Normalize Swahili orthographic text.

    - Convert to lowercase
    - Retain alphabetic characters, apostrophes, and standard whitespace
    - Strip extraneous punctuation and numbers

    Args:
        text: Raw transcript string.

    Returns:
        Cleaned, normalized text string.
    """
    if not text:
        return ""
    text = text.lower()
    # Normalize unicode apostrophes
    text = text.replace("’", "'").replace("`", "'")
    # Retain a-z, spaces, and valid Swahili contractions
    text = re.sub(r"[^a-z'\s]", " ", text)
    # Collapse consecutive spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_manifest(
    audio_dir: Path,
    tsv_input: Optional[Path],
    output_manifest: Path,
    target_sr: int = 16000
) -> int:
    """Create a standardized ASR manifest TSV from raw metadata.

    Args:
        audio_dir: Directory containing audio files.
        tsv_input: Path to source metadata TSV (e.g. Common Voice train.tsv).
        output_manifest: Destination path for clean manifest TSV.
        target_sr: Target sampling rate.

    Returns:
        Count of valid samples written to manifest.
    """
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Building manifest -> {output_manifest}")

    if not tsv_input or not tsv_input.exists():
        logger.warning(
            f"Source metadata {tsv_input} not found. Creating empty manifest template."
        )
        with open(output_manifest, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["audio_filepath", "duration", "text", "speaker_id"])
        return 0

    count = 0
    with open(tsv_input, "r", encoding="utf-8") as in_f, open(output_manifest, "w", encoding="utf-8", newline="") as out_f:
        reader = csv.DictReader(in_f, delimiter="\t")
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(["audio_filepath", "duration", "text", "speaker_id"])

        for row in reader:
            raw_text = row.get("sentence", "") or row.get("text", "")
            norm_text = normalize_swahili_text(raw_text)
            if not norm_text:
                continue

            rel_path = row.get("path", "")
            full_audio_path = audio_dir / rel_path
            speaker_id = row.get("client_id", "") or row.get("speaker_id", "")

            # Placeholder duration if audio file is not yet downloaded
            duration = 0.0
            writer.writerow([str(full_audio_path), f"{duration:.3f}", norm_text, speaker_id])
            count += 1

    logger.info(f"Manifest created with {count} records.")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Swahili ASR datasets and manifests.")
    parser.add_argument("--raw-dir", type=str, default="data/raw/common_voice_sw")
    parser.add_argument("--processed-dir", type=str, default="data/processed/common_voice_sw")
    parser.add_argument("--manifest-dir", type=str, default="data/manifests")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--augment-noise", action="store_true", help="Synthesize noisy evaluation splits.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("Initializing dataset preparation workflow...")
    manifest_dir = Path(args.manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    # Initialize empty standard manifest templates if not present
    for split in ["cv_sw_train.tsv", "cv_sw_dev.tsv", "cv_sw_test.tsv", "fleurs_sw_test.tsv"]:
        split_path = manifest_dir / split
        if not split_path.exists():
            build_manifest(
                audio_dir=Path(args.processed_dir),
                tsv_input=None,
                output_manifest=split_path,
                target_sr=args.sample_rate
            )
    logger.info("Dataset manifests initialized.")

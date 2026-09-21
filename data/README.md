# Data Directory

This directory stores datasets, audio manifests, and split definitions for the speech recognition research pipeline.

> **CRITICAL RULE**: Do **NOT** commit raw audio files, processed audio, or downloaded corpora to Git. All dataset directories below are excluded via `.gitignore`.

---

## Directory Organization

* `raw/`: Unprocessed downloaded source datasets (Common Voice, FLEURS, Noise sources, A13 device recordings).
* `processed/`: Resampled 16 kHz mono WAV files and normalized audio outputs.
* `manifests/`: Manifest files (TSV/JSON) linking audio file paths, durations, and normalized text transcripts.
* `splits/`: Deterministic train/validation/test and noise-augmented split lists.

---

## Instructions for Data Setup

1. **Download Datasets**:
   - Download Mozilla Common Voice Swahili from the official [Mozilla Common Voice Portal](https://commonvoice.mozilla.org/).
   - Download FLEURS Swahili from the [Hugging Face Datasets Hub](https://huggingface.co/datasets/google/fleurs).
   - Place downloaded archives in `data/raw/<dataset_name>/`.

2. **Preprocess and Build Manifests**:
   Run the data preparation script:
   ```bash
   python scripts/prepare_data.py \
       --raw-dir data/raw/common_voice_sw \
       --processed-dir data/processed/common_voice_sw \
       --manifest-dir data/manifests \
       --sample-rate 16000
   ```

3. **Verify Integrity**:
   Ensure manifests reference valid audio files and standard 16 kHz mono format before launching training or benchmarking experiments.

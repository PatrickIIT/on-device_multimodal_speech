# Dataset Specifications & Protocols

## Overview

This document specifies the datasets utilized in the research project, their intended experimental roles, preprocessing requirements, licensing conditions, and storage conventions.

> **IMPORTANT**: Datasets must NEVER be committed to Git. All raw and processed audio corpora must remain local or in designated external storage.

---

## Summary Matrix

| Dataset | Language / Focus | Role | Primary Split Usage | Sampling Rate | Format |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Mozilla Common Voice** | Swahili (`sw`) | Main ASR Corpus | Train / Validation / Test | 16 kHz | Single-channel WAV/FLAC |
| **Google FLEURS** | Swahili (`sw_ke` / `sw_tz`) | Cross-Dataset Evaluation | Test / Validation | 16 kHz | Single-channel WAV |
| **Speech Commands** | Keyword / Streaming | Streaming Baseline | Train / Val / Test | 16 kHz | Single-channel WAV |
| **Noisy Test Sets** | Swahili + Noise Mixes | Robustness Evaluation | Test only | 16 kHz | Single-channel WAV |
| **Samsung A13 Recordings** | Swahili (In-Situ) | Out-of-Domain Device Test | Test only | 16 kHz | Single-channel WAV |

---

## 1. Mozilla Common Voice Swahili

### Purpose
Mozilla Common Voice (Swahili subset) serves as the primary acoustic and textual corpus for training, validating, and testing streaming speech recognition models. It features a diverse demographic distribution of speakers recorded via varied browser and mobile microphones.

### Role in Pipeline
* **Train**: Acoustic model parameter estimation, knowledge distillation teacher training, quantization-aware training.
* **Validation**: Model selection, early stopping, hyperparameter tuning.
* **Test**: In-domain standard WER/CER benchmarking.

### Preprocessing Requirements
1. **Audio Formatting**: Convert all MP3 files to single-channel (mono) 16,000 Hz, 16-bit PCM WAV.
2. **Text Normalization**:
   * Lowercase all transcriptions.
   * Strip non-standard punctuation (retaining apostrophes where linguistically relevant in Swahili contractions).
   * Filter out empty or corrupted audio clips.
   * Remove utterances with duration $< 0.5\text{ s}$ or $> 20.0\text{ s}$ for streaming training stability.
3. **Manifest Creation**: Generate JSON/TSV manifests containing `audio_path`, `duration`, `text`, and `speaker_id`.

### Licensing & Attribution
* **License**: Creative Commons CC0 (Public Domain Dedication).
* **Citation**: Mozilla Common Voice Contributors.

---

## 2. Google FLEURS Swahili (`sw_ke`, `sw_tz`)

### Purpose
FLEURS (Few-shot Language Evaluation on Universal Robust Speech) provides a standardized multi-domain read-speech evaluation benchmark. It is used to test out-of-domain acoustic and dialect generalization across Kenyan and Tanzanian Swahili varieties.

### Role in Pipeline
* **Test**: Zero-shot cross-dataset evaluation of models trained on Common Voice.
* **Validation**: Calibration and threshold selection for hardware-aware selection.

### Preprocessing Requirements
1. Resample to mono 16 kHz, 16-bit PCM WAV.
2. Normalize text using identical Swahili orthography rules as Common Voice.
3. Construct evaluation manifests linking audio paths to normalized reference transcripts.

### Licensing & Attribution
* **License**: Creative Commons Attribution 4.0 International (CC BY 4.0).
* **Citation**: Conneau et al., "FLEURS: Few-shot Learning Evaluation on Universal Robust Speech", 2022.

---

## 3. Speech Commands (Multilingual / Swahili Adapted)

### Purpose
Provides short-duration ($\sim 1.0\text{ s}$) isolated keyword and command utterances. This dataset provides a controlled environment for testing ultra-low latency chunking (100 ms, 200 ms), keyword spotting, and VAD triggering.

### Role in Pipeline
* **Streaming Calibration**: Calibration of causal chunk buffers and VAD state-machine latency.
* **Lightweight Testing**: Rapid smoke testing of exported mobile runtimes.

### Preprocessing Requirements
1. Standardize duration to exactly 1.0 second (zero-padding if necessary).
2. Resample to 16 kHz mono WAV.

### Licensing & Attribution
* **License**: Creative Commons CC BY 4.0.

---

## 4. Noisy Evaluation Sets (Synthetic Robustness Testbed)

### Purpose
To evaluate the resilience of quantized, pruned, and streaming models against environmental noise commonly encountered on mobile devices in public and transit spaces.

### Role in Pipeline
* **Test Only**: Strict robustness evaluation across SNR levels ($0\text{ dB}$, $5\text{ dB}$, $10\text{ dB}$, $20\text{ dB}$, and clean).

### Noise Profiles
1. **Traffic**: Urban vehicular street noise.
2. **Office**: Keyboards, ambient chatter, HVAC.
3. **Crowd / Babble**: Multi-speaker background chatter.
4. **Wind**: Direct microphone wind buffeting.
5. **Fan**: Low-frequency periodic cooling fan noise.
6. **Music**: Background instrumental / acoustic music.

### Generation Procedure
Generated deterministically using `scripts/prepare_data.py --augment-noise` by mixing Common Voice / FLEURS test splits with noise sources at exact target SNRs.

---

## 5. Samsung Galaxy A13 In-Situ Calibration Recordings

### Purpose
A small, controlled out-of-domain evaluation set recorded directly through the internal microphone hardware of the Samsung Galaxy A13 in typical indoor/outdoor African ambient conditions.

### Role in Pipeline
* **Hardware Validation Only**: Measures the acoustic transfer function and AGC (Automatic Gain Control) characteristics of the Samsung Galaxy A13 microphone hardware.
* **Scientific Constraint**: These recordings do **not** replace the official standardized test sets (Common Voice, FLEURS) for comparative academic reporting.

### Recording Specifications
* Captured via onboard microphone hardware at native 16 kHz / 48 kHz resampled to 16 kHz mono PCM.
* Annotated with manual ground-truth orthographic transcriptions.

---

## Local Directory Layout

When downloaded and prepared locally, the data directory must conform to:

```text
data/
├── raw/
│   ├── common_voice_sw/
│   ├── fleurs_sw/
│   ├── speech_commands/
│   ├── noise_sources/
│   └── a13_recordings/
├── processed/
│   ├── common_voice_sw_16k/
│   ├── fleurs_sw_16k/
│   └── noisy_test_sets/
├── manifests/
│   ├── cv_sw_train.tsv
│   ├── cv_sw_dev.tsv
│   ├── cv_sw_test.tsv
│   └── fleurs_sw_test.tsv
└── splits/
    └── e8_noise_splits.json
```

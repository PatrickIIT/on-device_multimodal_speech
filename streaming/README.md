# Streaming Audio & Inference Pipeline

This module implements causal audio chunking, Voice Activity Detection (VAD), modular streaming inference engines, and streaming latency metrics.

---

## Streaming Pipeline Architecture

```text
Microphone / Audio Signal
           ↓
   [AudioChunker] (100 / 200 / 400 / 800 / 1000 ms)
           ↓
      [EnergyVAD]  (Silence Gating)
           ↓
   [StreamingASR]  (Acoustic Encoder & Decoder)
           ↓
  Partial Transcripts Stream
```

---

## Module Overview

| File | Purpose | Key Classes / Functions |
| :--- | :--- | :--- |
| `chunking.py` | Temporal framing and stream buffering | `AudioChunker`, `StreamBuffer` |
| `vad.py` | Silence detection and compute gating | `BaseVAD`, `EnergyVAD` |
| `streaming_inference.py` | Modular streaming decoding interface | `StreamingASR`, `GenericStreamingASR` |
| `latency_metrics.py` | Streaming latency tracking | `StreamingLatencyTracker` (FCL, CPT, UPL/EUL) |

---

## Streaming Chunking Configurations

The streaming pipeline supports parameterized chunk durations:
* **100 ms**: Ultra-low latency edge mode.
* **200 ms**: Standard conversational interactive streaming.
* **400 ms**: Balanced latency vs. acoustic accuracy baseline.
* **800 ms**: High-context streaming mode.
* **1000 ms**: Near-offline chunk window.

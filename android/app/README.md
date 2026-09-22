# Android Native Streaming ASR Application

This directory contains the project template for the standalone native Android demonstration app.

---

## Architectural Flow

```text
Android AudioRecord (16kHz Mono PCM)
         ↓
JNI Native Streaming Bridge
         ↓
Sherpa-ONNX / ONNX Runtime C++ Engine
         ↓
Live Swahili Transcription Stream UI
         ↓
(Future Extension) On-Device SLM Intent Parser
```

---

## Prerequisites for Build

* **Android Studio**: Flamingo (2022.2.1) or newer
* **Android NDK**: Version 25.1+
* **Target SDK**: Android 13 (API Level 33)
* **Minimum SDK**: Android 8.0 (API Level 26)

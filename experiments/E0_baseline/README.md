# Experiment E0: FP32 Baseline Reference

## Objective
Establish the uncompressed full-precision (FP32) baseline reference against which all subsequent compression, streaming, and hardware-aware optimization experiments will be compared.

## Hypothesis
The uncompressed FP32 streaming model will achieve the lowest Word Error Rate (WER) and Character Error Rate (CER), but will exhibit the highest memory consumption, longest execution latency, and highest Real-Time Factor (RTF) on the Samsung Galaxy A13 CPU.

## Variables
* **Independent Variable**: Model precision fixed to full FP32 (uncompressed).
* **Dependent Variables**: WER, CER, model size (MB), peak RAM (MB), P50/P90/P95 latency (ms), RTF, energy drain (Wh).
* **Control Variables**: Dataset split (Common Voice Swahili test set), sampling rate (16 kHz mono), chunk size (400 ms), test environment (Samsung Galaxy A13).

## Dataset
* **Primary**: Mozilla Common Voice Swahili (`cv_sw_test.tsv`).
* **Secondary**: Google FLEURS Swahili test split (`fleurs_sw_test.tsv`).

## Model
* **Architecture**: Streaming Conformer / Emformer baseline.
* **Precision**: FP32 (32-bit floating point).
* **Checkpoint**: Reference checkpoint trained on Swahili Common Voice train split.

## Procedure
1. Verify dataset manifest integrity and audio format (16 kHz mono WAV).
2. Load uncompressed FP32 baseline checkpoint in evaluation runtime.
3. Warm up inference engine with 10 dummy audio chunks.
4. Execute full test split decoding with fixed 400 ms chunk size.
5. Record per-utterance predictions, timing traces, memory usage, and sysfs battery power measurements.
6. Calculate WER, CER, RTF, and latency distributions.

## Metrics
* **Quality**: WER, CER.
* **Efficiency**: Model Size (MB), Peak RAM (MB), Mean/P50/P90/P95 Latency (ms), RTF, Cold-start latency (ms).

## Expected Output Files
* `results/raw/E0_baseline_samsung_a13_<timestamp>.json`
* `results/tables/E0_baseline_summary.csv`
* Prediction transcripts logged in `results/raw/E0_baseline_predictions.tsv`

# Experiment E2: INT8 Quantization Study

## Objective
Evaluate Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT) to 8-bit integer precision (INT8) for streaming Swahili speech models on ARM Cortex-A55 cores.

## Hypothesis
INT8 quantization will achieve a ~4x reduction in model storage, reduce peak RAM by over 40%, and achieve RTF < 0.5 on the Samsung Galaxy A13 with less than 2% relative increase in WER compared to the FP32 baseline.

## Variables
* **Independent Variable**: 8-bit integer quantization (PTQ symmetric per-channel vs. QAT).
* **Dependent Variables**: WER, CER, model size (MB), peak RAM (MB), P50/P90/P95 latency (ms), RTF, energy drain.
* **Control Variables**: Model architecture, training dataset, 400 ms chunk size.

## Dataset
* Mozilla Common Voice Swahili (Calibration: dev split, Evaluation: test split).

## Model
* Streaming Conformer quantized to INT8 (weights: qint8, activations: quint8).

## Procedure
1. Calibrate activation dynamic ranges on 256 representative dev split utterances.
2. Generate INT8 quantized ONNX / runtime graphs using symmetric per-channel quantization.
3. Deploy to device and execute benchmark protocol.
4. Record latency, RAM, and speech recognition hypotheses.

## Metrics
* WER, CER, Model Size (MB), Peak RAM (MB), Mean/P50/P90/P95 Latency (ms), RTF, Battery Energy (Wh).

## Expected Output Files
* `results/raw/E2_int8_samsung_a13_<timestamp>.json`
* Exported artifact: `models/exported/streaming_conformer_int8.onnx`

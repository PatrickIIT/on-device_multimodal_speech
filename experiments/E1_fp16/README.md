# Experiment E1: FP16 Precision Evaluation

## Objective
Evaluate the impact of half-precision floating-point (FP16) representation on model size, inference speed, RAM usage, and acoustic accuracy on the target mobile architecture.

## Hypothesis
Converting weights and activations to FP16 will reduce model storage footprint by 50% with negligible (< 0.5% relative) WER degradation, but CPU inference latency gains may be limited on Cortex-A55 cores without FP16 vectorized compute kernels.

## Variables
* **Independent Variable**: Numerical precision set to FP16 (16-bit float).
* **Dependent Variables**: WER, CER, model disk size (MB), peak RAM (MB), P50/P90/P95 latency (ms), RTF.
* **Control Variables**: Architecture, weights, chunk size (400 ms), test dataset.

## Dataset
* Mozilla Common Voice Swahili test set (`cv_sw_test.tsv`).

## Model
* **Architecture**: Streaming Conformer baseline.
* **Precision**: FP16.

## Procedure
1. Export FP32 baseline to FP16 ONNX representation.
2. Verify operator compatibility and validate numerical tolerances.
3. Deploy to Samsung Galaxy A13 evaluation harness.
4. Execute 50 warmup runs and benchmark full test split.
5. Record memory and latency distributions.

## Metrics
* WER, CER, Compression Ratio, Model Size (MB), Peak RAM (MB), Latency Percentiles (ms), RTF.

## Expected Output Files
* `results/raw/E1_fp16_samsung_a13_<timestamp>.json`
* Exported artifact: `models/exported/streaming_conformer_fp16.onnx`

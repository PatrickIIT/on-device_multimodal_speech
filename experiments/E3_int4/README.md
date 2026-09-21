# Experiment E3: INT4 Aggressive Quantization

## Objective
Investigate the limits of low-bit weight quantization by compressing the acoustic model to 4-bit integer representation (INT4) using group-wise scaling.

## Hypothesis
INT4 quantization will reduce model storage footprint by ~7x–8x relative to FP32, but may induce noticeable degradation in Swahili WER (> 10% relative) and may require specialized dequantization micro-kernels to translate into CPU latency improvements on Cortex-A55.

## Variables
* **Independent Variable**: 4-bit integer weight quantization with block sizes {16, 32, 64}.
* **Dependent Variables**: WER, CER, model storage size, runtime latency, memory bandwidth utilization.
* **Control Variables**: Evaluation dataset, acoustic chunk size (400 ms).

## Dataset
* Mozilla Common Voice Swahili test set.

## Model
* Streaming Conformer with INT4 weight quantization and INT8/FP16 activation handling.

## Procedure
1. Apply block-wise INT4 weight quantization across linear and projection layers.
2. Verify operator support in target mobile runtime.
3. Benchmark inference execution latency and evaluate WER over the test corpus.

## Metrics
* WER, CER, Model Size (MB), Peak RAM (MB), Latency (Mean, P50, P90, P95 ms), RTF.

## Expected Output Files
* `results/raw/E3_int4_samsung_a13_<timestamp>.json`
* Exported model: `models/exported/streaming_conformer_int4.onnx`

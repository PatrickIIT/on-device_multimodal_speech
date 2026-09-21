# Experiment E9: Mobile Inference Runtime Comparison

## Objective
Benchmark execution efficiency, memory overhead, and threading scalability across different mobile inference runtimes (ONNX Runtime mobile, Sherpa-ONNX, and ExecuTorch) running identical quantized model graphs on Samsung Galaxy A13.

## Hypothesis
Sherpa-ONNX and C++ native ONNX Runtime will achieve lower memory footprints and shorter per-chunk latency on the Exynos 850 CPU compared to standard Python-wrapped runtimes due to reduced heap allocation and optimized ARM NEON kernels.

## Variables
* **Independent Variable**: Inference runtime engine $\in \{\text{sherpa-onnx}, \text{onnxruntime-mobile}, \text{executorch}\}$.
* **Dependent Variables**: Cold-start latency (ms), warm-start latency (ms), peak RSS RAM (MB), CPU core scaling efficiency.
* **Control Variables**: Model graph (INT8 streaming conformer), chunk size (400 ms), test audio.

## Dataset
* Common Voice Swahili test set.

## Model
* Standardized INT8 streaming model exported to ONNX and ExecuTorch formats.

## Procedure
1. Export model graph to target runtime formats (.onnx, .pte).
2. Execute identical audio streams through each runtime harness on the Galaxy A13.
3. Record initialization overhead, per-chunk inference latency, and memory footprints.

## Metrics
* Cold-start latency (ms), Warm-start P95 latency (ms), Peak RAM (MB), Model load time (ms).

## Expected Output Files
* `results/raw/E9_runtime_sherpa_a13.json`
* `results/raw/E9_runtime_ort_a13.json`
* `results/raw/E9_runtime_executorch_a13.json`

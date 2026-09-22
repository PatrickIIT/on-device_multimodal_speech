# Experiment E5: Joint Pruning and Quantization

## Objective
Investigate the compound interaction between structured pruning and integer quantization (INT8/INT4) to discover compact, low-latency deployment configurations.

## Hypothesis
A moderately pruned model (e.g., 40% channel sparsity) combined with INT8 quantization will outperform an aggressive unpruned INT4 model in both WER and real-world execution speed on ARM Cortex-A55 cores due to better cache utilization and dense INT8 instruction support.

## Variables
* **Independent Variables**: Pruning sparsity $\in \{0.20, 0.40, 0.60\}$ crossed with precision $\in \{\text{INT8}, \text{INT4}\}$.
* **Dependent Variables**: WER, CER, model disk size (MB), latency percentiles, peak RAM (MB), RTF.
* **Control Variables**: Training data, test split, 400 ms chunk size, runtime environment.

## Dataset
* Mozilla Common Voice Swahili (Train/Dev/Test).

## Model
* Structurally pruned and quantized Streaming Conformer variants.

## Procedure
1. Take pruned models from E4 (20%, 40%, 60%).
2. Apply Quantization-Aware Training (QAT) or calibrated PTQ to INT8 and INT4.
3. Export composite artifacts to ONNX.
4. Benchmark latency, memory, and WER on Samsung Galaxy A13.

## Metrics
* WER, CER, Model Size (MB), Peak RAM (MB), Mean/P95 Latency (ms), RTF.

## Expected Output Files
* `results/raw/E5_prune20_int8_a13.json`
* `results/raw/E5_prune40_int8_a13.json`
* `results/raw/E5_prune40_int4_a13.json`
* `results/tables/E5_pruning_quantization_matrix.csv`

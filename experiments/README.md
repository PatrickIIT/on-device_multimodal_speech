# Experiment Protocols & Registry

This directory contains experimental protocols, parameter matrices, execution specifications, and reproducibility guidelines for the research study.

> **CRITICAL RULE**: Do not insert fabricated or simulated experimental numbers into any experiment protocol. All results must be generated via reproducible script execution and logged in `results/raw/` and `results/tables/`.

---

## Experiment Matrix

| ID | Title | Method / Focus | Target Hardware | Config File |
| :--- | :--- | :--- | :--- | :--- |
| **E0** | [FP32 Baseline](E0_baseline/README.md) | Uncompressed FP32 reference model | Samsung A13 / CPU | `configs/baseline.yaml` |
| **E1** | [FP16 Precision](E1_fp16/README.md) | Half-precision float conversion | Samsung A13 / CPU | `configs/quantization.yaml` |
| **E2** | [INT8 Quantization](E2_int8/README.md) | Symmetric/asymmetric 8-bit integer PTQ/QAT | Samsung A13 / CPU | `configs/quantization.yaml` |
| **E3** | [INT4 Quantization](E3_int4/README.md) | 4-bit block-wise quantization | Samsung A13 / CPU | `configs/quantization.yaml` |
| **E4** | [Structured Pruning](E4_pruning/README.md) | Channel / head pruning (20%, 40%, 60%, 80%) | Samsung A13 / CPU | `configs/pruning.yaml` |
| **E5** | [Pruning + Quantization](E5_pruning_quantization/README.md) | Joint structured sparsity and INT8/INT4 | Samsung A13 / CPU | `configs/proposed_method.yaml` |
| **E6** | [Knowledge Distillation](E6_distillation/README.md) | Teacher-student cross-architecture compression | Samsung A13 / CPU | `configs/distillation.yaml` |
| **E7** | [Streaming Chunk Study](E7_streaming/README.md) | Chunk duration sweep (100–1000 ms) | Samsung A13 / CPU | `configs/streaming.yaml` |
| **E8** | [Noise Robustness](E8_noise_robustness/README.md) | Noise conditions (0–20 dB SNR, 6 noise types) | Samsung A13 / CPU | `configs/baseline.yaml` |
| **E9** | [Runtime Comparison](E9_runtime/README.md) | ONNX Runtime vs. Sherpa-ONNX vs. ExecuTorch | Samsung A13 / CPU | `configs/hardware_a13.yaml` |
| **E10** | [Hardware-Aware Selection](E10_hardware_aware/README.md) | Constrained multi-objective candidate search | Samsung A13 / CPU | `configs/proposed_method.yaml` |

---

## Execution Command

To run or reproduce any experiment:

```bash
python scripts/reproduce_experiment.py \
    --experiment <EXPERIMENT_ID> \
    --config <CONFIG_PATH>
```

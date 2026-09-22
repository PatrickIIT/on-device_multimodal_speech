# Experiment Specifications Matrix

This document defines the 11 planned research experiments (E0 through E10) investigating streaming Swahili speech recognition model compression on the Samsung Galaxy A13.

> **RULE**: All experimental result fields in this documentation remain empty until measurements are captured on real hardware.

---

## Experiment Summary Registry

| Experiment | Title | Description | Primary Config | Target Device |
| :--- | :--- | :--- | :--- | :--- |
| **E0** | FP32 Baseline | Uncompressed floating point baseline | `configs/baseline.yaml` | Samsung A13 |
| **E1** | FP16 Precision | Half-precision float conversion | `configs/quantization.yaml` | Samsung A13 |
| **E2** | INT8 Quantization | 8-bit integer PTQ and QAT | `configs/quantization.yaml` | Samsung A13 |
| **E3** | INT4 Quantization | 4-bit group-wise weight quantization | `configs/quantization.yaml` | Samsung A13 |
| **E4** | Structured Pruning | Channel / head pruning (20%, 40%, 60%, 80%) | `configs/pruning.yaml` | Samsung A13 |
| **E5** | Pruning + Quantization | Joint structured sparsity and INT8/INT4 | `configs/proposed_method.yaml` | Samsung A13 |
| **E6** | Knowledge Distillation | Cross-architecture teacher-student distillation | `configs/distillation.yaml` | Samsung A13 |
| **E7** | Streaming Chunk Study | Chunk size evaluation (100, 200, 400, 800, 1000 ms) | `configs/streaming.yaml` | Samsung A13 |
| **E8** | Noise Robustness | Robustness across SNR levels (0–20 dB) and 6 noise types | `configs/baseline.yaml` | Samsung A13 |
| **E9** | Runtime Comparison | ONNX Runtime vs. Sherpa-ONNX vs. ExecuTorch | `configs/hardware_a13.yaml` | Samsung A13 |
| **E10** | Hardware-Aware Selection | Multi-objective constrained configuration search | `configs/proposed_method.yaml` | Samsung A13 |

---

## Detailed Protocols

### E0: Baseline FP32 Reference
* **Objective**: Measure baseline uncompressed acoustic model performance.
* **Metrics**: WER, CER, Model Size (MB), Peak RAM (MB), Latency Mean/P95 (ms), RTF.

### E1: FP16 Precision Evaluation
* **Objective**: Measure storage reduction and latency gains from FP16 conversion on ARM Cortex-A55.

### E2: INT8 Quantization Study
* **Objective**: Evaluate 8-bit integer quantization using ARM NEON `SDOT` integer instructions.

### E3: INT4 Aggressive Low-Bit Quantization
* **Objective**: Quantify acoustic accuracy retention and decompression overhead under 4-bit weights.

### E4: Structured Pruning Across Sparsity Targets
* **Objective**: Sweep structured channel sparsity (20%, 40%, 60%, 80%) with AdamW fine-tuning.

### E5: Compound Pruning and Quantization
* **Objective**: Evaluate whether moderate channel pruning (40%) + INT8 outperforms unpruned INT4.

### E6: Cross-Architecture Knowledge Distillation
* **Objective**: Transfer dark knowledge from a high-capacity teacher to a compact edge student model.

### E7: Streaming Acoustic Chunk-Size Study
* **Objective**: Sweep causal chunk durations $\in \{100, 200, 400, 800, 1000\}\text{ ms}$ to plot the latency-WER curve.

### E8: Environmental Noise Robustness
* **Objective**: Benchmark resilience to traffic, office, babble, wind, fan, and music noise at 0, 5, 10, 20 dB SNR.

### E9: Mobile Inference Runtime Benchmarking
* **Objective**: Compare ONNX Runtime Mobile, Sherpa-ONNX, and ExecuTorch on identical quantized graphs.

### E10: Hardware-Aware Multi-Objective Selection
* **Objective**: Search candidate grid $\Theta = \mathcal{Q} \times \mathcal{S} \times \mathcal{C}$ subject to $R \le 250\text{ MB}, L \le 150\text{ ms}$.

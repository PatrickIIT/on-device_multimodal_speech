# Compression Modules

This directory contains research implementations for model compression, profile representation, and hardware-aware configuration selection tailored for edge mobile deployment.

---

## Architecture Overview

```text
Baseline FP32 Model
      ↓
Quantization (FP16 / INT8 / INT4)
      ↓
Structured Pruning (Channel / Head / Layer)
      ↓
Cross-Architecture Knowledge Distillation
      ↓
Hardware-Aware Configuration Selection
```

---

## Modules

| File | Role | Description |
| :--- | :--- | :--- |
| `quantization.py` | Precision reduction | Configurable PTQ and QAT pipelines for FP16, INT8, and INT4 |
| `pruning.py` | Structural sparsity | Channel, head, and block pruning targeting 20%, 40%, 60%, 80% sparsity |
| `distillation.py` | Knowledge distillation | Multi-task loss combining CTC, Cross-Entropy, and feature matching |
| `model_profiles.py` | Metadata abstraction | `ModelProfile` dataclass managing model attributes and null measurements |
| `compression_pipeline.py` | End-to-end pipeline | Unified pipeline orchestrating multi-stage compression workflows |
| `hardware_aware.py` | Research selector | Multi-objective constrained candidate selector under mobile hardware limits |

---

## Hardware-Aware Selector Objective

The candidate selection objective is defined as:

$$J = \text{WER} + \lambda_1 \cdot \frac{\text{Latency}}{L_{\text{ref}}} + \lambda_2 \cdot \frac{\text{RAM}}{R_{\text{ref}}} + \lambda_3 \cdot \frac{\text{Energy}}{E_{\text{ref}}}$$

$$\text{subject to:} \quad \text{RAM} \le R_{\max}, \quad \text{Latency}_{p95} \le L_{\max}, \quad \frac{\text{WER} - \text{WER}_{\text{base}}}{\text{WER}_{\text{base}}} \le \Delta W_{\max}$$

*Status: Research formulation under active investigation.*

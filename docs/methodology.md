# Research Methodology

## 1. Overview of Experimental Workflow

The research methodology follows a principled, 11-stage empirical investigation to evaluate model compression, causal chunking, and hardware-aware selection for streaming Swahili speech recognition on low-resource Android devices.

```text
[Stage 1: Establish FP32 Baseline]
              ↓
[Stage 2: Quantization Analysis (FP16, INT8, INT4)]
              ↓
[Stage 3: Structured Pruning (20%, 40%, 60%, 80%)]
              ↓
[Stage 4: Joint Pruning + Quantization]
              ↓
[Stage 5: Cross-Architecture Knowledge Distillation]
              ↓
[Stage 6: Streaming Chunk-Size Study (100–1000 ms)]
              ↓
[Stage 7: Environmental Noise Robustness (0–20 dB SNR)]
              ↓
[Stage 8: Mobile Runtime Comparison (ORT, Sherpa, ExecuTorch)]
              ↓
[Stage 9: Hardware-Aware Configuration Selection]
              ↓
[Stage 10: Multi-Factor Ablation Studies]
              ↓
[Stage 11: Real-Device In-Situ Validation on Samsung A13]
```

---

## 2. Detailed Methodology Stages

### Stage 1: Establish Uncompressed FP32 Baseline (E0)
* **Goal**: Establish the ceiling for acoustic recognition accuracy and the upper bound for computational latency, RAM usage, and energy consumption.
* **Corpus**: Mozilla Common Voice Swahili (v17+) training and validation sets.
* **Architecture**: Streaming Conformer / Emformer with causal chunked self-attention.

### Stage 2: Precision Quantization (E1, E2, E3)
* **Goal**: Characterize the sensitivity of streaming speech recognition to numerical precision degradation.
* **Methods**:
  * FP16: Half-precision float representation.
  * INT8: Dynamic and static post-training quantization (PTQ) and Quantization-Aware Training (QAT).
  * INT4: Block-wise 4-bit weight quantization.

### Stage 3: Structured Pruning (E4)
* **Goal**: Induce parameter sparsity that translates directly into arithmetic speedups on mobile CPUs without requiring sparse matrix hardware kernels.
* **Sparsity Targets**: 20%, 40%, 60%, and 80% structured channel/head sparsity.
* **Criterion**: L1-norm magnitude ranking followed by gradient-based fine-tuning.

### Stage 4: Joint Pruning and Quantization (E5)
* **Goal**: Explore the combined effect of structured channel reduction and integer quantization.
* **Hypothesis**: Moderate pruning + INT8 quantization will offer superior Pareto efficiency compared to unpruned INT4 quantization on Cortex-A55 cores.

### Stage 5: Cross-Architecture Knowledge Distillation (E6)
* **Goal**: Guide the training of a compact, pruned edge model using soft targets from a high-capacity teacher model.
* **Loss Formulation**:
  $$\mathcal{L}_{\text{total}} = \alpha_{\text{CE}} \mathcal{L}_{\text{CE}} + \alpha_{\text{CTC}} \mathcal{L}_{\text{CTC}} + \alpha_{\text{KD}} T^2 \mathcal{L}_{\text{KL}}(P_{\text{student}}, P_{\text{teacher}}) + \alpha_{\text{feat}} \mathcal{L}_{\text{MSE}}$$

### Stage 6: Streaming Chunk-Size Study (E7)
* **Goal**: Evaluate the latency vs. WER curve across causal temporal windows:
  * 100 ms (interactive low-latency)
  * 200 ms (conversational)
  * 400 ms (balanced baseline)
  * 800 ms (high context)
  * 1000 ms (near-offline)

### Stage 7: Environmental Noise Robustness (E8)
* **Goal**: Quantify accuracy degradation of compressed streaming models in noisy acoustic settings.
* **SNRs**: Clean ($\infty$), 20 dB, 10 dB, 5 dB, 0 dB.
* **Noise Types**: Traffic, office, babble/crowd, wind, fan, music.

### Stage 8: Mobile Runtime Comparison (E9)
* **Goal**: Benchmark runtime graph execution engines on identical INT8 quantized model graphs.
* **Evaluated Runtimes**: ONNX Runtime Mobile, Sherpa-ONNX, ExecuTorch (XNNPACK).

### Stage 9: Hardware-Aware Configuration Selection (E10)
* **Goal**: Formulate and evaluate candidate search over the combinatorial space $\Theta = \mathcal{Q} \times \mathcal{S} \times \mathcal{C}$ under explicit hardware constraints:
  $$\min_{\theta \in \Theta} J(\theta) = \text{WER}(\theta) + \lambda_1 \text{Lat}_{\text{norm}}(\theta) + \lambda_2 \text{RAM}_{\text{norm}}(\theta) + \lambda_3 \text{Energy}_{\text{norm}}(\theta)$$
  $$\text{subject to:} \quad \text{RAM}(\theta) \le 250\text{ MB}, \quad \text{Latency}_{p95}(\theta) \le 150\text{ ms}$$

### Stage 10: Multi-Factor Ablation Studies
* Validate contributions of individual loss terms, pruning granularities, and context lookahead lengths.

### Stage 11: Real-Device In-Situ Validation
* Final deployment on Samsung Galaxy A13 hardware measuring real-time battery drain, thermal throttling profiles, and out-of-domain recording intelligibility.

# Research Plan: Hardware-Aware Compression of Streaming Speech Models for Low-Resource Android Devices

## Status
**Working Research Document — Hypotheses and experimental protocols under active investigation.**

---

## 1. Problem Statement

Deploying Automatic Speech Recognition (ASR) models on mobile devices in emerging markets presents acute computational challenges. Low-resource devices—exemplified by entry-level Android smartphones such as the Samsung Galaxy A13—are constrained by low-power CPU architectures (ARM Cortex-A55), limited RAM bandwidth, aggressive OS background memory reclamation (Low Memory Killer), and rapid thermal throttling. 

Furthermore, low-resource African languages such as Swahili (Kiswahili) suffer from comparatively limited labeled training corpora, requiring streaming models to generalize effectively while enduring aggressive model compression techniques (quantization, pruning, distillation).

---

## 2. Motivation

Most edge speech recognition benchmarks focus on flagship mobile silicon (e.g., Apple A-series / Qualcomm Snapdragon 8-series) equipped with dedicated Neural Processing Units (NPUs) and 8–16 GB of high-bandwidth memory. However, the vast majority of mobile users in regions where Swahili is spoken rely on entry-tier smartphones powered by efficiency-oriented octa-core processors with 3–4 GB of total system RAM.

Existing compression techniques are often evaluated in isolation:
* Quantization reduces weight bit-width but does not reduce internal tensor dimensions or operation count.
* Pruning introduces structural sparsity but may degrade cache locality on CPUs lacking sparse execution kernels.
* Streaming causal chunking trades off acoustic context for latency, interacting unpredictably with quantized weights.

There is a critical need for a holistic, hardware-aware methodology that jointly navigates these trade-offs directly on low-resource mobile hardware.

---

## 3. Research Question

> **How can we jointly optimize model compression (quantization, structured pruning, knowledge distillation) and streaming configuration (chunk size, context buffers) so that speech models maintain recognition quality while minimizing latency, memory, and energy on low-resource Android devices?**

---

## 4. Hypotheses

* **Hypothesis 1 (H1 — Quantization Threshold)**: 8-bit integer quantization (INT8) will preserve Word Error Rate (WER) within a 5% relative degradation of the uncompressed FP32 baseline while reducing peak RAM and Real-Time Factor (RTF) by at least 40% on ARM Cortex-A55 cores.
* **Hypothesis 2 (H2 — Joint Compression Synergy)**: Combining structured pruning (e.g., 40% channel sparsity) with knowledge distillation prior to INT8 quantization will yield lower WER and lower latency on the Samsung Galaxy A13 than applying aggressive INT4 post-training quantization alone.
* **Hypothesis 3 (H3 — Acoustic Context & Compression Coupling)**: Highly compressed models (e.g., INT4 or high-sparsity pruned models) will exhibit disproportionate acoustic degradation under smaller streaming chunk sizes ($\le 200\text{ ms}$) due to reduced representational capacity under truncated acoustic contexts.
* **Hypothesis 4 (H4 — Hardware-Aware Feasibility)**: A multi-objective search algorithm accounting for real-device hardware constraints (peak memory limit, CPU thermal throttling curve) will identify deployment configurations that satisfy real-time interactive constraints where unguided compression heuristics fail.

---

## 5. Datasets

The research protocol relies on standardized public benchmarks and controlled acoustic test sets:

1. **Mozilla Common Voice Swahili (v17+)**: Primary training, validation, and in-domain evaluation corpus.
2. **Google FLEURS Swahili**: In-domain and cross-dataset evaluation set to measure acoustic and dialect generalization.
3. **Speech Commands (Multilingual/Swahili adaptation)**: Short-duration streaming intent/keyword benchmark for latency calibration.
4. **Synthetic Noise Test Sets**: Clean evaluation splits mixed at 0 dB, 5 dB, 10 dB, and 20 dB SNRs with environmental noise (traffic, office, crowd, wind, fan).
5. **Samsung Galaxy A13 Calibration Set**: Small out-of-domain evaluation set recorded directly via the Galaxy A13 microphone to measure real-device acoustic degradation. *(Note: Used exclusively as supplementary validation, never replacing public benchmarks).*

---

## 6. Model Architectures

The model exploration strategy focuses on lightweight, streaming-capable acoustic architectures:

* **Streaming Conformer / Emformer / Zipformer**: Primary candidate family utilizing causal attention, chunked self-attention, and depthwise separable convolutions.
* **Distilled Whisper / Tiny-Wav2Vec2 Variants**: Reference sequence-to-sequence baseline architectures for cross-architectural compression comparisons.
* **Acoustic Tokenizer / CTC-Transducer Decoders**: Low-overhead decoders designed to avoid quadratic beam-search latency on mobile CPUs.

---

## 7. Compression Methods

We systematically investigate:

1. **Precision Quantization**:
   * Baseline FP32
   * FP16 (Half-precision floating point)
   * INT8 (Symmetric and asymmetric post-training quantization [PTQ] and Quantization-Aware Training [QAT])
   * INT4 (Group-wise and sub-channel weight quantization)
2. **Structured Pruning**:
   * Block and channel pruning on convolutional feature extractors
   * Attention head pruning and intermediate feed-forward dimension pruning on transformer layers
   * Evaluated sparsity targets: 20%, 40%, 60%, 80%
3. **Cross-Architecture Knowledge Distillation**:
   * Teacher-student loss formulation combining Connectionist Temporal Classification (CTC) loss, Cross-Entropy loss, and intermediate layer feature matching (MSE/cosine distance).

---

## 8. Streaming Experiments

Streaming performance is evaluated across a spectrum of audio chunk durations:
* **100 ms**: Low-latency interactive streaming.
* **200 ms**: Standard conversational streaming.
* **400 ms**: Intermediate chunk size.
* **800 ms**: High-context streaming.
* **1000 ms**: Near-offline chunking.

Each chunk configuration will be analyzed for:
* First Chunk Latency (FCL)
* Chunk Processing Time (CPT)
* User-Perceived Latency (UPL) / End-of-Utterance Latency (EUL)
* Word Error Rate degradation as a function of lookahead context.

---

## 9. Hardware Experiments & Target Device

### Target Platform
* **Device**: Samsung Galaxy A13 (SM-A135F)
* **CPU**: Samsung Exynos 850 (8x Cortex-A55 @ 2.0 GHz)
* **GPU**: ARM Mali-G52 MP1
* **RAM**: 3 GB / 4 GB LPDDR4X
* **Operating System**: Android 12 / 13

### Execution Environment
1. **Termux / CLI**: Automated benchmarking via compiled C++ / Python runtimes (ONNX Runtime mobile, Sherpa-ONNX, ExecuTorch).
2. **Android Application Harness**: Native Android APK measuring production foreground service lifecycle, JNI invocation overhead, and battery drain.

---

## 10. Evaluation Metrics

### Speech Quality
* **Word Error Rate (WER)**: Token-level edit distance.
* **Character Error Rate (CER)**: Character-level edit distance.

### Efficiency & System Performance
* **Storage Footprint**: Total model binary size in MB.
* **Memory**: Peak Resident Set Size (RSS) and Proportional Set Size (PSS) in MB.
* **Latency**: Mean, median, P50, P90, P95, cold-start, and warm-start execution times in milliseconds.
* **Real-Time Factor (RTF)**: $\text{RTF} = \frac{\text{Inference Time}}{\text{Audio Duration}}$.
* **Energy**: Battery discharge energy measured in Watt-hours (Wh) or microampere-hours ($\mu\text{Ah}$).
* **Thermal Dynamics**: CPU core temperatures and throttling event frequency during 10-minute continuous streaming benchmarks.

---

## 11. Proposed Hardware-Aware Method

We formulate the hardware-aware configuration selection problem as:

$$\min_{\theta \in \Theta} J(\theta) = \text{WER}(\theta) + \lambda_1 \cdot \text{Latency}(\theta) + \lambda_2 \cdot \text{RAM}(\theta) + \lambda_3 \cdot \text{Energy}(\theta)$$

subject to:
$$\text{RAM}_{\text{peak}}(\theta) \le R_{\max}, \quad \text{Latency}_{p95}(\theta) \le L_{\max}, \quad \text{WER}(\theta) - \text{WER}_{\text{baseline}} \le \Delta W_{\max}$$

Where $\Theta$ represents the combinatorial candidate space of:
$$\Theta = \mathcal{Q} \times \mathcal{S} \times \mathcal{C}$$
* $\mathcal{Q} \in \{\text{FP32}, \text{FP16}, \text{INT8}, \text{INT4}\}$
* $\mathcal{S} \in \{0\%, 20\%, 40\%, 60\%, 80\%\}$
* $\mathcal{C} \in \{100\text{ ms}, 200\text{ ms}, 400\text{ ms}, 800\text{ ms}, 1000\text{ ms}\}$

*Note: The selection algorithm is an empirical research formulation under investigation.*

---

## 12. Ablation Plan

1. **Ablation 1 (Standalone vs. Combined Compression)**: Compare isolated INT8 quantization, isolated 40% pruning, and joint pruned-quantized models.
2. **Ablation 2 (Impact of Distillation Loss Components)**: Evaluate student performance with:
   - Output logit distillation only
   - Hidden-state representation distillation only
   - Combined multi-loss distillation
3. **Ablation 3 (Chunk Size vs. Sparsity Sensitivity)**: Measure whether highly pruned models experience disproportionate degradation on 100 ms vs 800 ms chunks.
4. **Ablation 4 (Runtime Engine Overhead)**: Measure latency and memory differences across ONNX Runtime, Sherpa-ONNX, and ExecuTorch on identical quantized models.

---

## 13. Threats to Validity

* **Internal Validity**: Background OS processes, dynamic frequency scaling (DVFS), and thermal throttling on Android can introduce measurement variance in latency and energy. Mitigation: Standardized multi-run protocols, warm-up iterations, and CPU governor monitoring.
* **External Validity**: Results measured on the Exynos 850 / Cortex-A55 may not directly generalize to devices with heterogeneous big.LITTLE architectures (e.g., Cortex-A78 + A55). Mitigation: Document hardware architectural parameters precisely and plan multi-device validation.
* **Construct Validity**: Low WER does not always guarantee high subjective speech intelligibility for domain-specific Swahili terminology. Mitigation: Joint WER and CER evaluation with phoneme error analysis.

---

## 14. Reproducibility Plan

Every experiment conducted in this project will produce:
1. Exact configuration YAML archive.
2. SHA256 checksums of trained and exported model artifacts.
3. Dataset split manifests with persistent record IDs.
4. Raw unedited JSON execution logs containing latency distributions, memory snapshots, and recognition hypotheses.
5. Deterministic random seed controls across PyTorch, NumPy, and hardware execution backends.

---

## 15. Expected Contributions

1. **Empirical Benchmark**: The first rigorous, open-source benchmark of streaming Swahili ASR models compressed for low-tier Android hardware (Samsung Galaxy A13).
2. **Methodological Insights**: Systematic analysis of the coupling between streaming chunk sizes and non-linear compression schemes (structured pruning + INT4/INT8 quantization).
3. **Hardware-Aware Selection Framework**: A modular, reproducible toolkit for finding Pareto-optimal streaming ASR configurations under explicit mobile hardware and memory constraints.
4. **Open Research Artifacts**: Fully reproducible codebase, configuration files, and export scripts for edge speech AI research.

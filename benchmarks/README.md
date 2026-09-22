# Benchmarking Framework

This module provides tools for profiling on-device Automatic Speech Recognition (ASR) quality, latency, memory, real-time factor, and energy dissipation.

---

## Core Benchmark CLI

Run unified benchmark execution and schema generation:

```bash
python benchmarks/benchmark.py \
    --experiment E0_baseline \
    --model TBD \
    --dataset common_voice_swahili \
    --device samsung_a13 \
    --precision fp32 \
    --output results/raw/E0_baseline_a13.json
```

For a dry-run schema generation:
```bash
python benchmarks/benchmark.py --dry-run
```

---

## Module Overview

| Script | Purpose | Metrics Produced |
| :--- | :--- | :--- |
| `metrics.py` | Mathematical metric definitions | WER, CER, RTF, Compression Ratio, Latency Percentiles |
| `benchmark.py` | Master benchmarking orchestrator | JSON schema records matching project standard |
| `benchmark_asr.py` | Speech recognition accuracy | Manifest-based WER & CER evaluation |
| `benchmark_streaming.py` | Streaming chunk profiling | Per-chunk latency, buffer state, RTF across chunk sizes |
| `benchmark_latency.py` | Latency distribution profiler | Cold-start, Warm-start, P50, P90, P95 latency |
| `benchmark_memory.py` | RAM and footprint tracker | Peak RSS, initial RSS, delta RAM |
| `benchmark_energy.py` | Power and thermal monitor | Battery current (mA), voltage (mV), SoC temp (°C) |

---

## Benchmark Output Schema

All benchmarks output JSON objects strictly adhering to the schema:

```json
{
  "experiment": "E0_baseline",
  "model": "streaming_conformer_base",
  "dataset": "common_voice_swahili",
  "device": "samsung_a13",
  "precision": "fp32",
  "sparsity": null,
  "chunk_ms": 400,
  "wer": null,
  "cer": null,
  "model_size_mb": null,
  "parameters": null,
  "flops": null,
  "latency_mean_ms": null,
  "latency_p50_ms": null,
  "latency_p90_ms": null,
  "latency_p95_ms": null,
  "ram_peak_mb": null,
  "cpu_percent": null,
  "rtf": null,
  "energy_wh": null,
  "thermal_state": "",
  "timestamp": "2026-09-21T00:00:00Z"
}
```

*Note: Metric values remain `null` until measured empirically.*

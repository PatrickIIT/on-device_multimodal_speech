# Experiment E7: Streaming Chunk-Size Study

## Objective
Quantify the trade-off between acoustic recognition quality (WER) and interactive decoding latency across varying causal streaming chunk durations (100 ms, 200 ms, 400 ms, 800 ms, 1000 ms).

## Hypothesis
Increasing chunk size from 100 ms to 400 ms will yield substantial WER improvements due to expanded acoustic context, while marginal quality gains will plateau beyond 800 ms. Smaller chunks (100–200 ms) will minimize First Chunk Latency (FCL) but incur higher cumulative buffer management overhead.

## Variables
* **Independent Variable**: Audio chunk duration $\in \{100\text{ ms}, 200\text{ ms}, 400\text{ ms}, 800\text{ ms}, 1000\text{ ms}\}$.
* **Dependent Variables**: WER, CER, First Chunk Latency (ms), Chunk Processing Time (ms), End-of-Utterance Latency (ms), RTF.
* **Control Variables**: Model architecture, precision (INT8), test audio recordings.

## Dataset
* Mozilla Common Voice Swahili test set.

## Model
* Streaming Conformer (INT8 quantized).

## Procedure
1. Initialize streaming ASR engine with target chunk size.
2. Stream test audio files in chunks of specified duration.
3. Record per-chunk timestamp markers (arrival time, processing start, completion).
4. Compute cumulative transcript and streaming latency metrics.

## Metrics
* WER, CER, First Chunk Latency (ms), P50/P95 Chunk Processing Time (ms), End-of-Utterance Latency (ms), Stream RTF.

## Expected Output Files
* `results/raw/E7_streaming_chunk_100ms.json`
* `results/raw/E7_streaming_chunk_200ms.json`
* `results/raw/E7_streaming_chunk_400ms.json`
* `results/raw/E7_streaming_chunk_800ms.json`
* `results/raw/E7_streaming_chunk_1000ms.json`

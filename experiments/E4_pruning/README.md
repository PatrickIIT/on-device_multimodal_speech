# Experiment E4: Structured Pruning Across Sparsity Levels

## Objective
Evaluate the impact of structured pruning (channel, attention-head, and feed-forward dimension pruning) across sparsity levels of 20%, 40%, 60%, and 80% on streaming speech recognition accuracy and mobile runtime speed.

## Hypothesis
Structured pruning up to 40% channel sparsity followed by short fine-tuning will retain WER within 3% of baseline while providing linear reductions in MACs and proportional CPU speedups. Above 60% sparsity, severe acoustic representation loss is expected.

## Variables
* **Independent Variable**: Target structured sparsity ratio $\in \{0.20, 0.40, 0.60, 0.80\}$.
* **Dependent Variables**: WER, CER, active parameter count, FLOPs/MACs, inference latency, RAM.
* **Control Variables**: Model architecture, fine-tuning epochs (10), chunk size (400 ms).

## Dataset
* Mozilla Common Voice Swahili (Train for fine-tuning, Test for evaluation).

## Model
* Streaming Conformer with structured channel pruning on depthwise convolutions and linear layers.

## Procedure
1. Compute L1-norm / magnitude importance scores across layers.
2. Prune channels to target sparsity (20%, 40%, 60%, 80%).
3. Fine-tune pruned models for 10 epochs using AdamW.
4. Export and benchmark pruned models on Samsung Galaxy A13.

## Metrics
* WER, CER, Active Parameter Count, FLOPs, Latency (ms), RTF, Model Size (MB).

## Expected Output Files
* `results/raw/E4_pruning_sparsity_20_a13.json`
* `results/raw/E4_pruning_sparsity_40_a13.json`
* `results/raw/E4_pruning_sparsity_60_a13.json`
* `results/raw/E4_pruning_sparsity_80_a13.json`

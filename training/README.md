# Training & Evaluation Module

This module contains utilities for training streaming acoustic models, managing speech datasets, computing composite losses, and running validation evaluations.

---

## Directory Organization

* `train.py`: Acoustic model training entrypoint.
* `evaluate.py`: Evaluation runner for computing WER/CER across manifest files.
* `dataset.py`: Manifest-based speech dataset loader and spectrogram extractor.
* `losses.py`: CTC loss, Cross-Entropy loss with label smoothing, and composite distillation loss functions.
* `utils.py`: Seed initialization and checkpointing routines.

---

## Usage

### Train Baseline Model
```bash
python training/train.py --config configs/baseline.yaml
```

### Evaluate Model Checkpoint
```bash
python training/evaluate.py \
    --manifest data/manifests/cv_sw_test.tsv \
    --checkpoint models/checkpoints/baseline_fp32.pt
```

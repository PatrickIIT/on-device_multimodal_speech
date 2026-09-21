# Model Export & Validation

This directory contains export pipelines and validation utilities for deploying acoustic models onto mobile runtimes.

---

## Supported Export Targets

1. **ONNX (Open Neural Network Exchange)**:
   - Target runtimes: ONNX Runtime Mobile, Sherpa-ONNX.
   - Script: `export_onnx.py`
2. **ExecuTorch (.pte)**:
   - Target runtime: PyTorch Edge / XNNPACK delegate.
   - Script: `export_executorch.py`

---

## Export Validation

Validate graph structure and input/output tensor shapes:

```bash
python export/validate_export.py --model-path models/exported/streaming_model.onnx
```

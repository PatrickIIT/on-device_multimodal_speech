# Standalone Mobile Benchmarking Harness

This directory contains standalone benchmarking harnesses designed to measure pure C++ / native runtime inference overhead on mobile ARM processors without Python runtime noise.

---

## Supported Mobile Runtimes

* **Sherpa-ONNX C++ CLI**: Low-overhead streaming ASR runtime optimized for embedded Linux / Android.
* **ONNX Runtime Mobile C API**: Minimal build with selective kernel registration for ARM NEON.
* **ExecuTorch Runtime**: PyTorch lightweight edge execution engine with XNNPACK backend.

---

## Measurement Protocol

1. **CPU Governor Locking**: If root access is available, set CPU scaling governor to `performance` to eliminate DVFS latency variance; otherwise, perform 10 warmup runs before logging measurements.
2. **Thermal Monitoring**: Monitor `/sys/class/thermal/thermal_zone*/temp` before and after benchmark batches.
3. **Memory Snapshots**: Log `/proc/self/status` VmRSS to capture true heap footprint.

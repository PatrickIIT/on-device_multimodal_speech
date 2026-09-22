# Android Deployment & Mobile Evaluation

This directory contains deployment harnesses, Termux benchmarking scripts, and native Android application prototypes for evaluating streaming speech models on the **Samsung Galaxy A13**.

---

## Directory Organization

* `termux/`: Automated benchmarking scripts and toolchains for direct on-device execution via Termux CLI.
* `benchmark/`: Standalone C++ / Python benchmarking binaries for low-overhead mobile profiling.
* `app/`: Native Android application project for production foreground service and microphone streaming.

---

## Target Hardware Specifications: Samsung Galaxy A13

* **Model ID**: SM-A135F / SM-A137F
* **SoC**: Samsung Exynos 850 (8nm LPP)
* **CPU**: 8x ARM Cortex-A55 @ 2.0 GHz
* **GPU**: ARM Mali-G52 MP1
* **Memory**: 3 GB / 4 GB LPDDR4X RAM
* **Target OS**: Android 12 / 13 / 14

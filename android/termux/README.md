# On-Device Benchmarking via Termux

This directory provides scripts and workflows for executing automated speech recognition benchmarks directly on the Samsung Galaxy A13 via the Termux terminal environment.

---

## Setup on Samsung Galaxy A13

1. **Install Termux** from F-Droid (avoid outdated Google Play releases).
2. **Update Packages & Toolchain**:
   ```bash
   pkg update && pkg upgrade
   pkg install python git clang cmake ninja libsndfile
   ```
3. **Clone Repository on Device**:
   ```bash
   git clone https://github.com/PatrickIIT/on-device_multimodal_speech.git
   cd on-device_multimodal_speech
   pip install -r requirements.txt
   ```
4. **Run Benchmark Harness**:
   ```bash
   python benchmarks/benchmark.py \
       --experiment E0_baseline \
       --device samsung_a13 \
       --output results/raw/E0_baseline_termux.json
   ```

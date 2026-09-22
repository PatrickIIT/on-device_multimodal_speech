# Target Hardware & Empirical Measurement Protocol

## 1. Target Device: Samsung Galaxy A13

The primary reference device for this research project is the **Samsung Galaxy A13** (SM-A135F / SM-A137F), representing common entry-tier smartphone hardware in emerging markets.

### Hardware Specifications

| Component | Specification | Research Implications |
| :--- | :--- | :--- |
| **SoC** | Samsung Exynos 850 (8nm LPP) | Energy-efficient octa-core SoC without big performance cores |
| **CPU** | 8x ARM Cortex-A55 @ 2.0 GHz | In-order execution pipeline; sensitive to cache misses and branch mispredictions |
| **ISA Support** | ARMv8.2-A (64-bit), NEON, DotProd | Supports `SDOT`/`UDOT` instructions for accelerated 8-bit integer matrix multiplication |
| **GPU** | ARM Mali-G52 MP1 | Limited compute shaders; CPU execution is preferred for low-latency streaming |
| **NPU** | None | All neural inference must execute efficiently on CPU |
| **RAM** | 3 GB / 4 GB LPDDR4X | Aggressive OS Low Memory Killer (LMK); models must maintain peak RSS $\le 250\text{ MB}$ |
| **Battery** | 5,000 mAh Li-Ion | Real-device energy benchmarks monitor discharge current under continuous streaming |
| **OS** | Android 12 / 13 / 14 | Linux kernel 5.4+ with standard Android power management frameworks |

---

## 2. On-Device Measurement Protocol

### Setup via Termux / ADB
Benchmarking is automated via ADB shell or direct on-device execution in Termux:

1. **Environment Isolation**:
   * Terminate background third-party applications.
   * Lock screen brightness to 0% and enable Airplane Mode (to eliminate network telemetry noise).
   * Disable battery saver modes that force CPU frequency throttling.

2. **Warmup & Cache Stabilization**:
   * Execute 10 non-recorded warmup inference iterations before starting measurement capture.
   * Allow 5 seconds of cool-down time between benchmark batches to prevent thermal throttling skew.

3. **Power & Battery Sysfs Polling**:
   * Current draw: `/sys/class/power_supply/battery/current_now` ($\mu\text{A}$)
   * Voltage: `/sys/class/power_supply/battery/voltage_now` ($\mu\text{V}$)
   * Temperature: `/sys/class/power_supply/battery/temp` (tenths of °C)

4. **Memory Profiling**:
   * Poll `/proc/self/status` for `VmRSS` (Resident Set Size) and `VmHWM` (High Water Mark peak RAM).

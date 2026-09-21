"""Energy and battery discharge benchmarking for mobile hardware.

Interfaces with Linux/Android sysfs power management nodes
(`/sys/class/power_supply/battery/`) to monitor energy drain and thermal status.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("benchmark_energy")


class EnergyMonitor:
    """Monitors battery current, voltage, and SoC temperature on Android devices."""

    BATTERY_SYSFS = Path("/sys/class/power_supply/battery")

    def __init__(self) -> None:
        self.is_available = self.BATTERY_SYSFS.exists()
        if not self.is_available:
            logger.info("Android battery sysfs interface not detected (running in non-Android or sandboxed environment).")

    def read_voltage_mv(self) -> Optional[float]:
        """Read instantaneous battery voltage in millivolts."""
        node = self.BATTERY_SYSFS / "voltage_now"
        if node.exists():
            try:
                # Value typically in microvolts
                val = int(node.read_text().strip())
                return val / 1000.0 if val > 100000 else float(val)
            except Exception:
                return None
        return None

    def read_current_ma(self) -> Optional[float]:
        """Read instantaneous battery current in milliamperes."""
        node = self.BATTERY_SYSFS / "current_now"
        if node.exists():
            try:
                # Value typically in microamperes
                val = int(node.read_text().strip())
                return abs(val) / 1000.0 if abs(val) > 10000 else float(abs(val))
            except Exception:
                return None
        return None

    def read_thermal_celsius(self) -> Optional[float]:
        """Read battery / SoC thermal temperature in degrees Celsius."""
        node = self.BATTERY_SYSFS / "temp"
        if node.exists():
            try:
                val = int(node.read_text().strip())
                # Android sysfs temp is commonly in tenths of a degree Celsius
                return val / 10.0 if val > 100 else float(val)
            except Exception:
                return None
        return None

    def capture_snapshot(self) -> Dict[str, Optional[float]]:
        """Capture instantaneous power and thermal state."""
        return {
            "voltage_mv": self.read_voltage_mv(),
            "current_ma": self.read_current_ma(),
            "temperature_c": self.read_thermal_celsius(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile mobile energy consumption.")
    parser.add_argument("--duration-seconds", type=int, default=10, help="Duration to sample power metrics.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    monitor = EnergyMonitor()
    logger.info(f"Power monitor snapshot: {monitor.capture_snapshot()}")

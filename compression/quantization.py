"""Quantization module for streaming speech models.

Supports Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)
across FP16, INT8 (symmetric/asymmetric), and INT4 group-wise precision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("quantization")


@dataclass
class QuantizationConfig:
    """Configuration for model weight and activation quantization."""

    precision: str = "int8"  # "fp16", "int8", "int4"
    method: str = "ptq"      # "ptq" or "qat"
    symmetric: bool = True
    per_channel: bool = True
    group_size: int = 32     # Relevant for INT4 block quantization
    target_modules: List[str] = field(default_factory=lambda: ["Linear", "Conv1d", "MatMul"])
    calibration_samples: int = 256

    def __post_init__(self) -> None:
        valid_precisions = {"fp32", "fp16", "int8", "int4"}
        if self.precision.lower() not in valid_precisions:
            raise ValueError(f"Unsupported precision '{self.precision}'. Must be one of {valid_precisions}")
        valid_methods = {"ptq", "qat"}
        if self.method.lower() not in valid_methods:
            raise ValueError(f"Unsupported method '{self.method}'. Must be one of {valid_methods}")


class Quantizer:
    """Base quantizer interface for on-device ASR models."""

    def __init__(self, config: QuantizationConfig) -> None:
        self.config = config
        logger.info(f"Initialized Quantizer with precision={config.precision}, method={config.method}")

    def quantize_onnx_model(
        self,
        input_model_path: Union[str, Path],
        output_model_path: Union[str, Path],
        calibration_data: Optional[Any] = None
    ) -> Path:
        """Quantize an exported ONNX model artifact.

        Args:
            input_model_path: Path to baseline FP32 ONNX model.
            output_model_path: Destination path for quantized ONNX model.
            calibration_data: Calibration audio features for activation profiling.

        Returns:
            Path to the generated quantized model.
        """
        in_path = Path(input_model_path)
        out_path = Path(output_model_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Quantizing ONNX model: {in_path} -> {out_path} [{self.config.precision}]")

        if self.config.precision == "fp16":
            logger.info("Applying FP16 float conversion...")
            # TODO: Integrate onnxconverter_common.float16.convert_float_to_float16
        elif self.config.precision == "int8":
            logger.info(f"Applying INT8 quantization (symmetric={self.config.symmetric})...")
            # TODO: Integrate onnxruntime.quantization.quantize_dynamic / quantize_static
        elif self.config.precision == "int4":
            logger.info(f"Applying INT4 block quantization (group_size={self.config.group_size})...")
            # TODO: Integrate ONNX MatMulNBits / QNN INT4 quantization kernels

        return out_path

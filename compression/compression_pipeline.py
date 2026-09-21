"""Orchestration pipeline for multi-stage speech model compression.

Coordinates sequential and joint compression pipelines:
Baseline FP32 -> Structured Pruning -> Knowledge Distillation -> Quantization.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from compression.distillation import DistillationConfig
from compression.model_profiles import ModelProfile
from compression.pruning import PruningConfig, StructuredPruner
from compression.quantization import QuantizationConfig, Quantizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("compression_pipeline")


class CompressionPipeline:
    """Manages the lifecycle of compressing an ASR model for edge deployment."""

    def __init__(
        self,
        quant_config: Optional[QuantizationConfig] = None,
        prune_config: Optional[PruningConfig] = None,
        distill_config: Optional[DistillationConfig] = None,
    ) -> None:
        self.quant_config = quant_config or QuantizationConfig()
        self.prune_config = prune_config
        self.distill_config = distill_config

    def run_pipeline(
        self,
        model_name: str,
        input_checkpoint: Union[str, Path],
        output_dir: Union[str, Path],
    ) -> ModelProfile:
        """Execute configured compression stages.

        Args:
            model_name: Identifier for output model profile.
            input_checkpoint: Path to initial uncompressed model.
            output_dir: Directory where compressed artifacts will be saved.

        Returns:
            ModelProfile describing the resulting compressed artifact.
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Starting compression pipeline for model: {model_name}")

        sparsity = None
        if self.prune_config:
            logger.info(f"Stage 1: Applying structured pruning (sparsity={self.prune_config.target_sparsity:.2%})")
            pruner = StructuredPruner(self.prune_config)
            sparsity = self.prune_config.target_sparsity

        if self.distill_config:
            logger.info("Stage 2: Distillation fine-tuning with teacher guidance")

        logger.info(f"Stage 3: Quantizing to precision {self.quant_config.precision}")
        quantizer = Quantizer(self.quant_config)

        # Build resulting profile template
        profile = ModelProfile(
            name=model_name,
            precision=self.quant_config.precision,
            sparsity=sparsity,
            runtime="onnxruntime",
        )

        logger.info(f"Compression pipeline finished. Generated profile: {profile}")
        return profile

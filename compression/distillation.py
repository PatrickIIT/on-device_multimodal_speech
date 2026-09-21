"""Knowledge Distillation module for cross-architecture acoustic compression.

Defines the distillation framework combining task losses (CTC, Cross-Entropy)
with distillation losses (soft-target KL-Divergence, intermediate feature matching).
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("distillation")


@dataclass
class DistillationConfig:
    """Configuration parameters for knowledge distillation."""

    temperature: float = 2.0
    alpha_ce: float = 0.5
    alpha_kd: float = 0.5
    alpha_feature: float = 0.0
    feature_matching_enabled: bool = False

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(f"Temperature must be positive, got {self.temperature}")
        if not (0.0 <= self.alpha_ce <= 1.0):
            raise ValueError("alpha_ce must be in range [0.0, 1.0]")
        if not (0.0 <= self.alpha_kd <= 1.0):
            raise ValueError("alpha_kd must be in range [0.0, 1.0]")


class DistillationLoss:
    """Computes combined ground-truth and distillation objectives.

    L_total = alpha_ce * L_ground_truth + alpha_kd * (T^2 * L_distillation) + alpha_feat * L_feature
    """

    def __init__(self, config: DistillationConfig) -> None:
        self.config = config

    def compute_kd_loss(self, student_logits: Any, teacher_logits: Any) -> float:
        """Compute soft-target cross-entropy or KL divergence between teacher and student.

        Args:
            student_logits: Logits from student network.
            teacher_logits: Logits from frozen teacher network.

        Returns:
            Computed scalar distillation loss.
        """
        # TODO: Implement PyTorch KL-divergence: F.kl_div(log_softmax(student/T), softmax(teacher/T)) * (T^2)
        return 0.0

    def compute_feature_loss(self, student_feats: Any, teacher_feats: Any) -> float:
        """Compute intermediate representation matching loss (MSE or Cosine Distance)."""
        # TODO: Implement layer projection and MSE loss
        return 0.0

    def compute_total_loss(
        self,
        ground_truth_loss: float,
        student_logits: Any,
        teacher_logits: Any,
        student_feats: Optional[Any] = None,
        teacher_feats: Optional[Any] = None
    ) -> Dict[str, float]:
        """Compute composite multi-task distillation loss."""
        kd_loss = self.compute_kd_loss(student_logits, teacher_logits)
        feat_loss = 0.0
        if self.config.feature_matching_enabled and student_feats is not None and teacher_feats is not None:
            feat_loss = self.compute_feature_loss(student_feats, teacher_feats)

        total = (
            self.config.alpha_ce * ground_truth_loss
            + self.config.alpha_kd * kd_loss
            + self.config.alpha_feature * feat_loss
        )

        return {
            "total_loss": total,
            "ground_truth_loss": ground_truth_loss,
            "kd_loss": kd_loss,
            "feature_loss": feat_loss,
        }

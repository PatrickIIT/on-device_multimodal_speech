"""Loss functions for ASR training and knowledge distillation."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("losses")


class ASRLoss:
    """Computes Connectionist Temporal Classification (CTC) and Cross-Entropy loss."""

    def __init__(self, blank_id: int = 0, label_smoothing: float = 0.0) -> None:
        self.blank_id = blank_id
        self.label_smoothing = label_smoothing

    def compute_ctc_loss(
        self,
        log_probs: Any,
        targets: Any,
        input_lengths: Any,
        target_lengths: Any
    ) -> float:
        """Calculate CTC loss across acoustic frames.

        Args:
            log_probs: Tensor of shape (T, N, C) containing log probabilities.
            targets: Target token sequence tensor.
            input_lengths: Sequence lengths of acoustic encoder output.
            target_lengths: Sequence lengths of ground-truth target tokens.

        Returns:
            Scalar loss value.
        """
        # TODO: Implement torch.nn.functional.ctc_loss
        return 0.0


class CompositeDistillationLoss:
    """Multi-task distillation loss combining hard task loss and soft teacher distillation."""

    def __init__(
        self,
        alpha_ctc: float = 0.4,
        alpha_kd: float = 0.6,
        temperature: float = 2.0
    ) -> None:
        self.alpha_ctc = alpha_ctc
        self.alpha_kd = alpha_kd
        self.temperature = temperature

    def __call__(
        self,
        student_log_probs: Any,
        teacher_log_probs: Any,
        ground_truth_targets: Any
    ) -> Dict[str, float]:
        """Compute composite objective."""
        # TODO: Implement PyTorch tensor distillation computation
        return {
            "total_loss": 0.0,
            "ctc_loss": 0.0,
            "kd_loss": 0.0,
        }

"""Hardware-aware candidate configuration selection prototype.

Status: Research formulation under investigation.

This module evaluates candidate speech model compression configurations against
explicit hardware constraints (maximum RAM, maximum P95 latency, maximum energy,
and maximum relative WER degradation) and ranks feasible configurations
using a multi-objective cost formulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("hardware_aware")


@dataclass
class HardwareConstraints:
    """Constraints imposed by target mobile device architecture."""

    max_ram_mb: float = 250.0            # Hard ceiling to avoid OS Low Memory Killer
    max_latency_ms: float = 150.0        # Hard ceiling for real-time per-chunk budget
    max_energy_wh: Optional[float] = None  # Battery consumption threshold
    max_wer_degradation: float = 0.05    # Max relative increase over baseline WER


@dataclass
class ObjectiveWeights:
    """Weights for the composite multi-objective cost function."""

    lambda_latency: float = 0.35
    lambda_ram: float = 0.35
    lambda_energy: float = 0.30

    def __post_init__(self) -> None:
        total = self.lambda_latency + self.lambda_ram + self.lambda_energy
        if abs(total - 1.0) > 1e-4:
            logger.debug(f"Objective weights sum to {total:.4f} (expected ~1.0). Normalizing...")
            self.lambda_latency /= total
            self.lambda_ram /= total
            self.lambda_energy /= total


@dataclass
class CandidateEvaluation:
    """Measured performance of a candidate compression configuration."""

    config_id: str
    precision: str
    sparsity: float
    chunk_ms: int
    wer: float
    latency_p95_ms: float
    ram_peak_mb: float
    energy_wh: Optional[float] = None


class HardwareAwareSelector:
    """Prototype multi-objective selector under explicit hardware constraints."""

    def __init__(
        self,
        constraints: HardwareConstraints,
        weights: Optional[ObjectiveWeights] = None,
        baseline_wer: Optional[float] = None,
    ) -> None:
        self.constraints = constraints
        self.weights = weights or ObjectiveWeights()
        self.baseline_wer = baseline_wer
        logger.info(
            f"Initialized HardwareAwareSelector [Max RAM: {constraints.max_ram_mb} MB, "
            f"Max Latency: {constraints.max_latency_ms} ms, Max WER Deg: {constraints.max_wer_degradation:.1%}]"
        )

    def is_feasible(self, candidate: CandidateEvaluation) -> Tuple[bool, List[str]]:
        """Check whether a candidate satisfies all hardware and accuracy constraints.

        Args:
            candidate: Evaluated candidate configuration metrics.

        Returns:
            Tuple of (is_feasible_bool, list_of_violation_reasons).
        """
        violations: List[str] = []

        if candidate.ram_peak_mb > self.constraints.max_ram_mb:
            violations.append(
                f"Peak RAM ({candidate.ram_peak_mb:.1f} MB) exceeds limit ({self.constraints.max_ram_mb:.1f} MB)"
            )

        if candidate.latency_p95_ms > self.constraints.max_latency_ms:
            violations.append(
                f"P95 Latency ({candidate.latency_p95_ms:.1f} ms) exceeds limit ({self.constraints.max_latency_ms:.1f} ms)"
            )

        if (
            self.constraints.max_energy_wh is not None
            and candidate.energy_wh is not None
            and candidate.energy_wh > self.constraints.max_energy_wh
        ):
            violations.append(
                f"Energy ({candidate.energy_wh:.4f} Wh) exceeds limit ({self.constraints.max_energy_wh:.4f} Wh)"
            )

        if self.baseline_wer is not None and self.baseline_wer > 0:
            rel_deg = (candidate.wer - self.baseline_wer) / self.baseline_wer
            if rel_deg > self.constraints.max_wer_degradation:
                violations.append(
                    f"Relative WER degradation ({rel_deg:.2%}) exceeds threshold ({self.constraints.max_wer_degradation:.2%})"
                )

        return len(violations) == 0, violations

    def compute_cost(
        self,
        candidate: CandidateEvaluation,
        max_latency_ref: float,
        max_ram_ref: float,
        max_energy_ref: float = 1.0,
    ) -> float:
        """Compute normalized multi-objective scalar cost J(theta).

        J = WER + lambda1 * (Latency / L_ref) + lambda2 * (RAM / R_ref) + lambda3 * (Energy / E_ref)
        """
        norm_latency = candidate.latency_p95_ms / max(1e-6, max_latency_ref)
        norm_ram = candidate.ram_peak_mb / max(1e-6, max_ram_ref)
        norm_energy = (candidate.energy_wh or 0.0) / max(1e-6, max_energy_ref)

        cost = (
            candidate.wer
            + self.weights.lambda_latency * norm_latency
            + self.weights.lambda_ram * norm_ram
            + self.weights.lambda_energy * norm_energy
        )
        return float(cost)

    def select_best_configuration(
        self, candidates: List[CandidateEvaluation]
    ) -> Dict[str, Any]:
        """Filter feasible configurations and select the candidate with minimal cost.

        Args:
            candidates: List of evaluated candidate configurations.

        Returns:
            Dictionary containing 'feasible_candidates', 'selected_candidate', and 'selection_reason'.
        """
        if not candidates:
            return {
                "feasible_candidates": [],
                "selected_candidate": None,
                "selection_reason": "No candidates provided for evaluation.",
            }

        feasible_list: List[Tuple[CandidateEvaluation, float]] = []
        rejected_summary: List[Dict[str, Any]] = []

        # Find reference scaling maximums among all candidates
        max_lat = max((c.latency_p95_ms for c in candidates), default=1.0)
        max_ram = max((c.ram_peak_mb for c in candidates), default=1.0)
        max_energy = max(((c.energy_wh or 0.0) for c in candidates), default=1.0) or 1.0

        for cand in candidates:
            feasible, reasons = self.is_feasible(cand)
            if feasible:
                cost = self.compute_cost(cand, max_lat, max_ram, max_energy)
                feasible_list.append((cand, cost))
            else:
                rejected_summary.append({
                    "config_id": cand.config_id,
                    "violations": reasons,
                })

        if not feasible_list:
            return {
                "feasible_candidates": [],
                "selected_candidate": None,
                "selection_reason": (
                    f"Zero candidates satisfied all constraints. {len(rejected_summary)} rejected."
                ),
                "rejections": rejected_summary,
            }

        # Sort by scalar cost ascending
        feasible_list.sort(key=lambda item: item[1])
        best_cand, best_cost = feasible_list[0]

        reason = (
            f"Selected candidate '{best_cand.config_id}' with lowest objective cost J={best_cost:.4f} "
            f"(WER={best_cand.wer:.4f}, P95 Latency={best_cand.latency_p95_ms:.1f}ms, RAM={best_cand.ram_peak_mb:.1f}MB) "
            f"out of {len(feasible_list)} feasible candidates."
        )

        return {
            "feasible_candidates": [c.config_id for c, _ in feasible_list],
            "selected_candidate": best_cand,
            "selected_cost": round(best_cost, 6),
            "selection_reason": reason,
            "rejections": rejected_summary,
        }

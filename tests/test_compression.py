"""Unit tests for compression, model profile metadata, and hardware-aware selection."""

import pytest

from compression.distillation import DistillationConfig, DistillationLoss
from compression.hardware_aware import (
    CandidateEvaluation,
    HardwareAwareSelector,
    HardwareConstraints,
    ObjectiveWeights,
)
from compression.model_profiles import ModelProfile
from compression.pruning import PruningConfig
from compression.quantization import QuantizationConfig


class TestModelProfiles:
    def test_model_profile_creation(self):
        profile = ModelProfile(
            name="conformer_int8",
            parameters=25_000_000,
            model_size_mb=25.4,
            precision="int8",
            sparsity=0.40,
            runtime="onnxruntime",
        )
        assert profile.name == "conformer_int8"
        assert profile.precision == "int8"
        assert profile.sparsity == 0.40

        d = profile.to_dict()
        assert d["model_size_mb"] == 25.4
        reconstructed = ModelProfile.from_dict(d)
        assert reconstructed.name == profile.name

    def test_invalid_sparsity(self):
        with pytest.raises(ValueError):
            ModelProfile(name="test", sparsity=1.5)

    def test_invalid_model_size(self):
        with pytest.raises(ValueError):
            ModelProfile(name="test", model_size_mb=-10.0)


class TestCompressionConfigs:
    def test_quantization_config(self):
        cfg = QuantizationConfig(precision="int8", method="ptq")
        assert cfg.precision == "int8"

        with pytest.raises(ValueError):
            QuantizationConfig(precision="int2")

        with pytest.raises(ValueError):
            QuantizationConfig(method="invalid_method")

    def test_pruning_config(self):
        cfg = PruningConfig(granularity="channel", target_sparsity=0.40)
        assert cfg.target_sparsity == 0.40

        with pytest.raises(ValueError):
            PruningConfig(target_sparsity=1.2)

        with pytest.raises(ValueError):
            PruningConfig(granularity="invalid_granularity")

    def test_distillation_config(self):
        cfg = DistillationConfig(temperature=2.0, alpha_ce=0.5, alpha_kd=0.5)
        assert cfg.temperature == 2.0

        with pytest.raises(ValueError):
            DistillationConfig(temperature=-1.0)
        with pytest.raises(ValueError):
            DistillationConfig(alpha_ce=1.5)


class TestHardwareAwareSelector:
    def test_selector_feasibility_and_ranking(self):
        constraints = HardwareConstraints(
            max_ram_mb=250.0,
            max_latency_ms=150.0,
            max_wer_degradation=0.10,
        )
        selector = HardwareAwareSelector(
            constraints=constraints,
            weights=ObjectiveWeights(lambda_latency=0.35, lambda_ram=0.35, lambda_energy=0.30),
            baseline_wer=0.15,
        )

        cand_feasible_1 = CandidateEvaluation(
            config_id="c_int8_prune40",
            precision="int8",
            sparsity=0.40,
            chunk_ms=400,
            wer=0.155,             # ~3.3% relative degradation -> feasible
            latency_p95_ms=90.0,   # < 150 -> feasible
            ram_peak_mb=180.0,     # < 250 -> feasible
            energy_wh=0.015,
        )

        cand_feasible_2 = CandidateEvaluation(
            config_id="c_int8_unpruned",
            precision="int8",
            sparsity=0.0,
            chunk_ms=400,
            wer=0.152,             # ~1.3% relative degradation
            latency_p95_ms=130.0,  # < 150 -> feasible
            ram_peak_mb=230.0,     # < 250 -> feasible
            energy_wh=0.022,
        )

        cand_infeasible_ram = CandidateEvaluation(
            config_id="c_fp32_baseline",
            precision="fp32",
            sparsity=0.0,
            chunk_ms=400,
            wer=0.150,
            latency_p95_ms=320.0,  # > 150 -> violation
            ram_peak_mb=450.0,     # > 250 -> violation
            energy_wh=0.050,
        )

        # Check individual feasibility
        is_f1, reasons1 = selector.is_feasible(cand_feasible_1)
        assert is_f1
        assert len(reasons1) == 0

        is_f_fp32, reasons_fp32 = selector.is_feasible(cand_infeasible_ram)
        assert not is_f_fp32
        assert len(reasons_fp32) >= 2

        # Evaluate selection
        result = selector.select_best_configuration([
            cand_feasible_1,
            cand_feasible_2,
            cand_infeasible_ram,
        ])

        assert len(result["feasible_candidates"]) == 2
        assert "c_fp32_baseline" not in result["feasible_candidates"]
        assert result["selected_candidate"] is not None
        assert result["selected_candidate"].config_id == "c_int8_prune40"

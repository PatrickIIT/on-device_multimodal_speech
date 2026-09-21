"""Unit tests for configuration file validation."""

from pathlib import Path
import pytest
import yaml

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"


class TestConfigFiles:
    def test_all_config_files_exist(self):
        expected_configs = [
            "baseline.yaml",
            "quantization.yaml",
            "pruning.yaml",
            "distillation.yaml",
            "streaming.yaml",
            "hardware_a13.yaml",
            "proposed_method.yaml",
        ]
        for cfg_name in expected_configs:
            cfg_path = CONFIGS_DIR / cfg_name
            assert cfg_path.exists(), f"Missing expected configuration file: {cfg_path}"

    def test_configs_load_valid_yaml(self):
        for cfg_path in CONFIGS_DIR.glob("*.yaml"):
            with open(cfg_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
            assert isinstance(content, dict), f"Config {cfg_path.name} must parse as a dictionary"
            assert len(content) > 0, f"Config {cfg_path.name} cannot be empty"

    def test_baseline_config_structure(self):
        with open(CONFIGS_DIR / "baseline.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert "experiment" in cfg
        assert cfg["experiment"]["name"] == "E0_baseline"
        assert "dataset" in cfg
        assert "streaming" in cfg
        assert "hardware" in cfg
        assert cfg["hardware"]["device"] == "samsung_a13"

    def test_hardware_a13_config_structure(self):
        with open(CONFIGS_DIR / "hardware_a13.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert "device" in cfg
        assert "Samsung Galaxy A13" in cfg["device"]["name"]
        assert "runtime" in cfg
        assert "constraints" in cfg
        assert "measurement" in cfg

    def test_proposed_method_config_structure(self):
        with open(CONFIGS_DIR / "proposed_method.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        assert "experiment" in cfg
        assert cfg["experiment"]["name"] == "E10_hardware_aware"
        assert "objective" in cfg
        assert "hardware_constraints" in cfg
        assert "search_space" in cfg

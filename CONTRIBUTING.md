# Contributing to Hardware-Aware Compression of Streaming Speech Models

Thank you for your interest in contributing to this research project. This repository is dedicated to reproducible empirical research on hardware-aware compression of streaming automatic speech recognition (ASR) models for resource-constrained Android smartphones (specifically targeting the Samsung Galaxy A13).

## Core Research Principles

1. **Absolute Experimental Integrity**:
   - **Never fabricate, manipulate, or insert placeholder numbers as real results.**
   - All published tables, figures, and benchmark reports must be generated strictly from reproducible execution scripts and recorded artifacts.
   - If an experiment has not yet been executed, leave metrics as `null` or marked with `TODO`.

2. **Strict Reproducibility**:
   - Every experiment must record: configuration YAML, model checkpoint/export hash, dataset manifest version, random seed, software dependencies, and hardware/device profile.
   - Scripts must execute deterministically given a fixed random seed.

3. **Low-Resource Hardware Focus**:
   - Algorithms and implementations must keep real-world embedded and mobile constraints (CPU throttling, RAM limits, battery thermal state) at the forefront.

---

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/PatrickIIT/on-device_multimodal_speech.git
   cd on-device_multimodal_speech
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Install the package in editable mode**:
   ```bash
   pip install -e .
   ```

4. **Run the test suite**:
   ```bash
   pytest tests/
   ```

---

## Contribution Workflow

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/compression-strategy-name
   ```
2. Write modular, well-documented code with type annotations and docstrings.
3. Add unit tests in `tests/` covering new logic, edge cases, and configuration validation.
4. Ensure all unit tests pass:
   ```bash
   pytest
   ```
5. Follow PEP 8 style conventions.
6. Open a Pull Request with a clear explanation of changes, mathematical motivation, and reproducibility verification.

---

## Code Quality Standards

- **Python Version**: Python 3.10+
- **Type Hints**: Use standard `typing` annotations where practical.
- **Paths**: Use `pathlib.Path` instead of hardcoded string concatenations or platform-specific separators.
- **Configurations**: Use modular YAML configs under `configs/`.
- **Logging**: Use Python standard `logging` module rather than arbitrary print statements.
- **No Large Files**: Never commit raw audio files, large datasets, or binary model checkpoints to the repository.

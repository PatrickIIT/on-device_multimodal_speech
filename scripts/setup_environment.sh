#!/usr/bin/env bash
# ==============================================================================
# Environment Setup Script for on-device_multimodal_speech
# Sets up Python virtual environment and verifies tooling.
# ==============================================================================

set -euo pipefail

echo "=== Initializing Research Environment ==="

# Check Python version
python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Detected Python version: ${python_version}"

# Create virtual environment if not present
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment in .venv/..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
# shellcheck source=/dev/null
source .venv/bin/activate

echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

if [ -f "requirements.txt" ]; then
    echo "Installing project dependencies from requirements.txt..."
    pip install -r requirements.txt
fi

echo "Installing project in editable development mode..."
pip install -e .

echo "Ensuring directory structure..."
mkdir -p data/raw data/processed data/manifests data/splits
mkdir -p results/raw results/tables results/figures
mkdir -p models/checkpoints models/exported

echo "Running unit test verification..."
pytest tests/

echo "=== Environment Setup Completed Successfully ==="

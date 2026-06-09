#!/bin/bash
set -euo pipefail

ENV_NAME="nla-train-env"
PYTHON="python3.10"

$PYTHON --version

echo "==> Creating virtual environment: $ENV_NAME"
$PYTHON -m venv "$ENV_NAME"
source "$ENV_NAME/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip --quiet

echo "==> Installing core deps"
pip install --quiet \
    "torch>=2.3.0" \
    "numpy>=1.26.0" \
    "scipy>=1.13.0" \
    "tqdm>=4.66.0" \
    "pyyaml>=6.0" \
    "rich>=13.7.0" \
    "httpx>=0.27.0" \
    "orjson>=3.9.0" \
    "pandas>=2.2.0" \
    "pyarrow>=16.0.0" \
    "accelerate>=0.34.0" \
    "safetensors>=0.4.0" \
    "tokenizers>=0.20.0"

echo "==> Installing PEFT + TRL"
pip install --quiet "peft>=0.12.0" "trl>=0.11.0"

echo "==> Installing datasets + huggingface_hub"
pip install --quiet "datasets>=2.20.0" "huggingface_hub>=0.24.0"

echo "==> Installing SGLang 0.5.10 + transformers 5.3.0"
pip install --quiet "sglang[all]==0.5.10"
pip install --quiet "transformers==5.3.0" "huggingface_hub>=1.5.0"

echo "==> Installing this package in editable mode"
pip install --quiet -e .

echo ""
echo "Done. To activate:"
echo "  source $ENV_NAME/bin/activate"

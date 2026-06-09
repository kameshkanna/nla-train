#!/bin/bash
# One-command environment setup for nla-train.
# Tested on Lambda Stack 22.04 (python3.10 default).
#
# Usage:
#   bash setup.sh
#   source nla-train-env/bin/activate

set -euo pipefail

ENV_NAME="nla-train-env"
PYTHON="python3.10"

echo "==> Checking Python"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: $PYTHON not found."
    echo "  Lambda Stack 22.04: python3.10 is the default interpreter."
    exit 1
fi
"$PYTHON" --version

echo "==> Creating virtual environment: $ENV_NAME"
"$PYTHON" -m venv "$ENV_NAME"
source "$ENV_NAME/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip --quiet

echo "==> Installing core ML deps"
pip install --quiet \
    "torch>=2.3.0" \
    "numpy>=1.26.0" \
    "scipy>=1.13.0" \
    "tqdm>=4.66.0" \
    "pyyaml>=6.0" \
    "rich>=13.7.0"

echo "==> Installing HuggingFace stack"
# transformers 4.45+ for LoRA + GRPO; avoid 5.x for now (sglang compat)
pip install --quiet \
    "transformers>=4.45.0,<5.0.0" \
    "tokenizers>=0.20.0" \
    "accelerate>=0.34.0" \
    "safetensors>=0.4.0" \
    "huggingface_hub>=0.24.0" \
    "datasets>=2.20.0"

echo "==> Installing PEFT + TRL for LoRA and GRPO"
pip install --quiet \
    "peft>=0.12.0" \
    "trl>=0.11.0"

echo "==> Installing FlashAttention-2 (H100 SXM5 — sm90a)"
# flash-attn must be built against the installed torch.
# The --no-build-isolation flag avoids re-downloading torch during build.
pip install --quiet flash-attn --no-build-isolation || \
    echo "WARNING: flash-attn build failed — training will fall back to SDPA (still fast on H100)"

echo "==> Installing SGLang (AV/AR serving during labeling)"
# Used only for stage2 labeling via kitft checkpoints.
pip install --quiet "sglang[all]==0.5.10"
# Restore transformers pin that sglang may have bumped
pip install --quiet "transformers>=4.45.0,<5.0.0"

echo "==> Installing data pipeline deps"
pip install --quiet \
    "pyarrow>=16.0.0" \
    "pandas>=2.2.0" \
    "orjson>=3.9.0" \
    "httpx[asyncio]>=0.27.0"

echo "==> Installing this package in editable mode"
pip install --quiet -e .

echo ""
echo "Done. To activate:"
echo "  source $ENV_NAME/bin/activate"

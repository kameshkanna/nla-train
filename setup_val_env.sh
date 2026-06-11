#!/bin/bash
# Validation env: SGLang-compatible stack, isolated from nla-rl-env.
#
# nla-rl-env has vLLM 0.16.0 + torch 2.9.1 which conflicts with SGLang.
# This env uses torch 2.5.1 (SGLang-compatible) with no vLLM.
#
# Usage:
#   bash setup_val_env.sh
#   source nla-val-env/bin/activate
#   bash scripts/run_validation.sh

set -euo pipefail

ENV_NAME="nla-val-env"
PYTHON="python3.10"

echo "==> Python version"
$PYTHON --version

echo "==> Creating venv: $ENV_NAME"
$PYTHON -m venv "$ENV_NAME"
source "$ENV_NAME/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip --quiet

echo "==> Installing torch 2.5.1 (CUDA 12.1 — SGLang compatible)"
pip install --quiet \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

echo "==> Installing SGLang"
pip install --quiet "sglang[all]==0.4.1"

echo "==> Installing transformers + PEFT + accelerate"
pip install --quiet \
    "transformers==4.47.0" \
    "peft==0.14.0" \
    "accelerate==1.2.1"

echo "==> Installing data + util deps"
pip install --quiet \
    "pyarrow>=21.0.0" \
    "numpy>=1.26.0" \
    "scipy>=1.13.0" \
    "tqdm>=4.66.0" \
    "pyyaml>=6.0" \
    "safetensors>=0.4.0" \
    "tokenizers>=0.20.0" \
    "huggingface_hub>=0.24.0" \
    "pandas>=2.2.0"

echo "==> Installing this package in editable mode"
pip install --quiet -e .

echo ""
echo "==> Verifying key versions:"
python -c "
import torch, transformers, peft, sglang
print(f'  torch:        {torch.__version__}')
print(f'  transformers: {transformers.__version__}')
print(f'  peft:         {peft.__version__}')
print(f'  sglang:       {sglang.__version__}')
print(f'  CUDA:         {torch.cuda.is_available()}')
"

echo ""
echo "Done. Activate with:"
echo "  source $ENV_NAME/bin/activate"
echo "Then run:"
echo "  bash scripts/run_validation.sh"

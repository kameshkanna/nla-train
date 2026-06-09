#!/bin/bash
# Setup for nla-train on Lambda Stack 22.04 (Python 3.10, CUDA 12.x).
# Two venvs: nla-train-env (training) and sglang-env (serving).

set -euo pipefail

PYTHON="python3.10"

$PYTHON --version

# ── Training env ──────────────────────────────────────────────────────────────
echo "==> nla-train-env"
$PYTHON -m venv nla-train-env
source nla-train-env/bin/activate

pip install -q --upgrade pip setuptools wheel

pip install -q \
    torch numpy scipy tqdm pyyaml rich \
    "transformers>=4.45,<5" tokenizers accelerate safetensors \
    "huggingface_hub>=0.24" "datasets>=2.20" \
    "peft>=0.12" "trl>=0.11" \
    "pyarrow>=16" pandas orjson httpx

# Pre-built flash-attn wheel — no compilation, ~30s install
# Matches: torch 2.x, CUDA 12.x, Python 3.10, Linux x86_64
TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
CUDA_VER=$(python -c "import torch; print('cu' + torch.version.cuda.replace('.','')[:4])")
echo "==> flash-attn (torch=$TORCH_VER cuda=$CUDA_VER)"
pip install -q \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4/flash_attn-2.7.4+${CUDA_VER}torch${TORCH_VER}cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
    || echo "WARNING: flash-attn wheel not found for this config — falling back to SDPA"

pip install -q -e .
deactivate

# ── SGLang env (separate — needs transformers==5.3.0) ─────────────────────────
echo "==> sglang-env"
$PYTHON -m venv sglang-env
source sglang-env/bin/activate
pip install -q --upgrade pip setuptools wheel
pip install -q "sglang[all]==0.5.10"
deactivate

echo ""
echo "Done."
echo "  Training:  source nla-train-env/bin/activate"
echo "  SGLang:    source sglang-env/bin/activate"

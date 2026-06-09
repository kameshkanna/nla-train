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

pip install -q --no-deps .  # non-editable, avoids setuptools backend issues
deactivate

# ── SGLang env ────────────────────────────────────────────────────────────────
echo "==> sglang-env"
$PYTHON -m venv sglang-env
source sglang-env/bin/activate
pip install -q --upgrade pip setuptools wheel
pip install -q "sglang[all]==0.5.10"
deactivate

echo ""
echo "Done. Using SDPA attention (torch 2.9.1 — no flash-attn wheel available)."
echo "  Training:  source nla-train-env/bin/activate"
echo "  SGLang:    source sglang-env/bin/activate"

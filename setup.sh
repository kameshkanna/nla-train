#!/bin/bash
# Setup for nla-train on Lambda Stack 22.04 (Python 3.10).
# Creates: nla-train-env (training) and sglang-env (serving).

set -euo pipefail

PYTHON="python3.10"
ROOT="$(cd "$(dirname "$0")" && pwd)"

$PYTHON --version

# ── Training env ──────────────────────────────────────────────────────────────
echo "==> nla-train-env"
rm -rf nla-train-env
$PYTHON -m venv nla-train-env
source nla-train-env/bin/activate

pip install -q --upgrade pip setuptools wheel 2>/dev/null || true

pip install -q \
    torch numpy scipy tqdm pyyaml rich \
    "transformers>=4.45,<5" tokenizers accelerate safetensors \
    "huggingface_hub>=0.24" "datasets>=2.20" \
    "peft>=0.12" "trl>=0.11" \
    "pyarrow>=16" pandas orjson httpx

# Add repo root to sys.path via .pth file — avoids all setuptools/build issues
echo "$ROOT" > nla-train-env/lib/python3.10/site-packages/nla_train_repo.pth

deactivate

# ── SGLang env ────────────────────────────────────────────────────────────────
echo "==> sglang-env"
rm -rf sglang-env
$PYTHON -m venv sglang-env
source sglang-env/bin/activate
pip install -q --upgrade pip setuptools wheel 2>/dev/null || true
pip install -q "sglang[all]==0.5.10"
deactivate

echo ""
echo "Done."
echo "  Training:  source nla-train-env/bin/activate"
echo "  SGLang:    source sglang-env/bin/activate"

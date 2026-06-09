#!/bin/bash
# One-command environment setup for nla-train.
# Tested on Lambda Stack 22.04 (python3.10 default).
#
# Creates two virtual environments:
#   nla-train-env   — training stack (transformers 4.x, peft, trl, flash-attn)
#   sglang-env      — SGLang serving only (requires transformers==5.3.0, conflicts with training)
#
# Usage:
#   bash setup.sh
#   source nla-train-env/bin/activate      # for training / datagen stages 0,1,3
#   source sglang-env/bin/activate         # for stage 2 labeling / serving kitft checkpoints

set -euo pipefail

PYTHON="python3.10"

echo "==> Checking Python"
if ! command -v "$PYTHON" &>/dev/null; then
    echo "ERROR: $PYTHON not found. Lambda Stack 22.04: python3.10 is the default."
    exit 1
fi
"$PYTHON" --version

# ─── Training environment ─────────────────────────────────────────────────────

echo ""
echo "==> Creating training venv: nla-train-env"
"$PYTHON" -m venv nla-train-env
source nla-train-env/bin/activate

echo "==> Upgrading pip + build tools"
pip install --upgrade pip setuptools wheel --quiet

echo "==> Installing core ML deps"
pip install --quiet \
    "torch>=2.3.0" \
    "numpy>=1.26.0" \
    "scipy>=1.13.0" \
    "tqdm>=4.66.0" \
    "pyyaml>=6.0" \
    "rich>=13.7.0"

echo "==> Installing HuggingFace stack (transformers 4.x)"
pip install --quiet \
    "transformers>=4.45.0,<5.0.0" \
    "tokenizers>=0.20.0" \
    "accelerate>=0.34.0" \
    "safetensors>=0.4.0" \
    "huggingface_hub>=0.24.0" \
    "datasets>=2.20.0"

echo "==> Installing PEFT + TRL"
pip install --quiet \
    "peft>=0.12.0" \
    "trl>=0.11.0"

echo "==> Installing FlashAttention-2 (H100 SXM5 — sm90a)"
# wheel + --no-build-isolation reuses already-installed torch/ninja
pip install --quiet wheel
pip install --quiet flash-attn --no-build-isolation || \
    echo "WARNING: flash-attn build failed — will fall back to SDPA (still fast on H100)"

echo "==> Installing data pipeline deps"
pip install --quiet \
    "pyarrow>=16.0.0" \
    "pandas>=2.2.0" \
    "orjson>=3.9.0" \
    "httpx>=0.27.0"

echo "==> Installing nla-train package (editable)"
pip install --quiet --upgrade setuptools
pip install --quiet -e .

deactivate

# ─── SGLang serving environment ──────────────────────────────────────────────

echo ""
echo "==> Creating SGLang venv: sglang-env (transformers==5.3.0 required by sglang)"
"$PYTHON" -m venv sglang-env
source sglang-env/bin/activate

pip install --upgrade pip setuptools wheel --quiet
pip install --quiet "sglang[all]==0.5.10"

deactivate

# ─── Done ────────────────────────────────────────────────────────────────────

echo ""
echo "================================================================"
echo "Setup complete. Two environments created:"
echo ""
echo "  Training / datagen (stages 0,1,3 + all training scripts):"
echo "    source nla-train-env/bin/activate"
echo ""
echo "  SGLang server (stage 2 labeling — serve kitft checkpoint):"
echo "    source sglang-env/bin/activate"
echo "    python -m sglang.launch_server --model-path kitft/nla-qwen2.5-7b-L20-av \\"
echo "        --port 30000 --disable-radix-cache --mem-fraction-static 0.45"
echo "================================================================"

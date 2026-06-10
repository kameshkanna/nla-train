#!/bin/bash
# Sets up a dedicated venv for GRPO RL training with vLLM-backed generation.
#
# Stack (researched for CUDA 12.8, Python 3.10):
#   torch 2.9.1+cu128  ← required by vLLM 0.16.0
#   vLLM 0.16.0        ← highest version compatible with torch 2.9.1
#   TRL 1.5.1          ← use_vllm=True, vllm_mode="colocate" API
#   transformers 4.57.6 ← vLLM 0.16.0 requires transformers <5
#   peft 0.19.1
#   accelerate 1.13.0
#
# Usage:
#   bash setup_rl_env.sh
#   source nla-rl-env/bin/activate
#   bash scripts/run_rl.sh

set -euo pipefail

ENV_NAME="nla-rl-env"
PYTHON="python3.10"

echo "==> Python version"
$PYTHON --version

echo "==> Creating venv: $ENV_NAME"
$PYTHON -m venv "$ENV_NAME"
source "$ENV_NAME/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip --quiet

echo "==> Installing torch 2.9.1 (CUDA 12.8)"
pip install --quiet \
    torch==2.9.1 \
    torchvision==0.24.1 \
    torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128

echo "==> Installing vLLM 0.16.0"
pip install --quiet vllm==0.16.0

echo "==> Installing TRL 1.5.1 with vLLM + PEFT extras"
pip install --quiet "trl[vllm,peft]==1.5.1"

echo "==> Pinning transformers to 4.57.6 (vLLM requires <5)"
pip install --quiet "transformers==4.57.6"

echo "==> Installing remaining ML deps"
pip install --quiet \
    "accelerate==1.13.0" \
    "peft==0.19.1" \
    "datasets>=5.0.0" \
    "pyarrow>=21.0.0" \
    "numpy>=1.26.0" \
    "scipy>=1.13.0" \
    "tqdm>=4.66.0" \
    "pyyaml>=6.0" \
    "rich>=13.7.0" \
    "httpx>=0.27.0" \
    "orjson>=3.9.0" \
    "pandas>=2.2.0" \
    "safetensors>=0.4.0" \
    "tokenizers>=0.20.0" \
    "huggingface_hub>=0.24.0"

echo "==> Installing this package in editable mode"
pip install --quiet -e .

echo ""
echo "==> Verifying key versions:"
python -c "
import torch, vllm, trl, transformers, peft
print(f'  torch:          {torch.__version__}')
print(f'  vllm:           {vllm.__version__}')
print(f'  trl:            {trl.__version__}')
print(f'  transformers:   {transformers.__version__}')
print(f'  peft:           {peft.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
"

echo ""
echo "Done. Activate with:"
echo "  source $ENV_NAME/bin/activate"
echo "Then run:"
echo "  bash scripts/run_rl.sh"

#!/bin/bash
# Setup for nla-train on Lambda Stack 22.04 (Python 3.10).

set -euo pipefail

PYTHON="python3.10"
ROOT="$(cd "$(dirname "$0")" && pwd)"

$PYTHON --version

echo "==> nla-train-env"
rm -rf nla-train-env
$PYTHON -m venv nla-train-env
source nla-train-env/bin/activate

pip install -q --upgrade pip

pip install -q \
    torch numpy scipy tqdm pyyaml rich \
    "transformers>=4.45,<5" tokenizers accelerate safetensors \
    "huggingface_hub>=0.24" "datasets>=2.20" \
    "peft>=0.12" "trl>=0.11" \
    "pyarrow>=16" pandas orjson httpx

# Add repo root to sys.path — no build system needed
echo "$ROOT" > nla-train-env/lib/python3.10/site-packages/nla_train_repo.pth

deactivate

echo ""
echo "Done. source nla-train-env/bin/activate"
echo ""
echo "NOTE: For SGLang (stage 2 labeling), use the system Python — it's pre-installed:"
echo "  python3 -m sglang.launch_server --model-path kitft/nla-qwen2.5-7b-L20-av \\"
echo "      --port 30000 --disable-radix-cache --mem-fraction-static 0.45"

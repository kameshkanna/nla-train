#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
python3.10 --version

echo "==> nla-train-env"
rm -rf nla-train-env
python3.10 -m venv --system-site-packages nla-train-env
source nla-train-env/bin/activate

# uv is 10-100x faster than pip — use it if available, else fall back to pip
if command -v uv &>/dev/null; then
    INSTALL="uv pip install -q"
else
    INSTALL="pip install -q"
fi

$INSTALL \
    "transformers>=4.45,<5" tokenizers accelerate safetensors \
    "huggingface_hub>=0.24" "datasets>=2.20" \
    "peft>=0.12" "trl>=0.11" \
    "pyarrow>=16" orjson httpx

echo "$ROOT" > nla-train-env/lib/python3.10/site-packages/nla_train_repo.pth

deactivate
echo "Done. source nla-train-env/bin/activate"

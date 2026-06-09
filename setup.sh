#!/bin/bash
set -euo pipefail

python3.10 --version

echo "==> nla-train-env"
rm -rf nla-train-env
python3.10 -m venv nla-train-env
source nla-train-env/bin/activate

pip install -q --upgrade pip

pip install -q \
    torch numpy scipy tqdm pyyaml rich \
    "transformers>=4.45,<5" tokenizers accelerate safetensors \
    "huggingface_hub>=0.24" "datasets>=2.20" \
    "peft>=0.12" "trl>=0.11" \
    "pyarrow>=16" pandas orjson httpx \
    "sglang[all]==0.5.10"

# sglang pins transformers==5.3.0 — restore our version after
pip install -q "transformers>=4.45,<5"

pip install -q -e .

deactivate
echo "Done. source nla-train-env/bin/activate"
echo "SGLang for stage 2: python3 -m sglang.launch_server (uses system python)"

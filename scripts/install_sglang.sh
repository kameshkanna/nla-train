#!/bin/bash
# Run this once before stage 2 labeling, inside nla-train-env.
# Kept separate because sglang takes 5+ minutes to install.
set -euo pipefail
pip install --quiet "sglang[all]==0.5.10"
pip install --quiet "transformers==5.3.0" "huggingface_hub>=1.5.0"
echo "SGLang ready."

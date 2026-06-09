#!/bin/bash
# Full data generation pipeline: Stage 0 → 1 → 2 → 3
#
# Prerequisites:
#   1. source nla-train-env/bin/activate
#   2. sglang serve --model-path kitft/nla-qwen2.5-7b-L20-av \
#          --port 30000 --disable-radix-cache --mem-fraction-static 0.45
#
# Usage:
#   bash scripts/run_datagen.sh [config]

set -euo pipefail

CONFIG="${1:-configs/qwen7b_layer20.yaml}"

echo "==> Stage 0: Activation extraction (vLLM)"
python -m nla_train.datagen.stage0_extract \
    --config "$CONFIG" \
    --resume

echo "==> Stage 1: Document-level split"
python -m nla_train.datagen.stage1_split \
    --config "$CONFIG"

echo "==> Stage 2: AV labeling via SGLang (kitft checkpoint)"
echo "    Make sure SGLang server is running on port 30000 before this step."
python -m nla_train.datagen.stage2_label \
    --config "$CONFIG"

echo "==> Stage 3: Pack final training datasets"
python -m nla_train.datagen.stage3_pack \
    --config "$CONFIG"

echo "==> Data generation complete."
echo "    Outputs: data/train/av_sft_train.parquet, ar_sft_train.parquet, rl_train.parquet"

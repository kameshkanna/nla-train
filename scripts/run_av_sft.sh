#!/bin/bash
# AV SFT training.
#
# Usage:
#   bash scripts/run_av_sft.sh [config]

set -euo pipefail

CONFIG="${1:-configs/qwen7b_layer20.yaml}"

echo "==> AV SFT: Training Activation Verbalizer"
python -m nla_train.av_sft \
    --config "$CONFIG" \
    --data-dir data/train \
    --nla-meta data/labeled/nla_meta_av.yaml

echo "==> AV SFT complete. Checkpoint: checkpoints/av_sft/final"

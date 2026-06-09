#!/bin/bash
# AR SFT training.
#
# Usage:
#   bash scripts/run_ar_sft.sh [config]

set -euo pipefail

CONFIG="${1:-configs/qwen7b_layer20.yaml}"

echo "==> AR SFT: Training Activation Reconstructor"
python -m nla_train.ar_sft \
    --config "$CONFIG" \
    --data-dir data/train \
    --nla-meta data/labeled/nla_meta_av.yaml

echo "==> AR SFT complete. Checkpoint: checkpoints/ar_sft/final"

#!/bin/bash
# Validate our trained AV checkpoint against kitft reference.
#
# Usage:
#   bash scripts/run_validation.sh [our_av_ckpt] [ar_ckpt] [kitft_av_ckpt]
#
# Examples:
#   # Ours only (no kitft comparison):
#   bash scripts/run_validation.sh
#
#   # Side-by-side with kitft:
#   bash scripts/run_validation.sh \
#       checkpoints/grpo/final_av \
#       checkpoints/ar_sft/final \
#       kitft/nla-qwen2.5-7b-L20-av

set -euo pipefail

OUR_AV="${1:-checkpoints/grpo/final_av}"
AR_CKPT="${2:-checkpoints/ar_sft/final}"
KITFT_AV="${3:-}"

export CUDA_VISIBLE_DEVICES=0

KITFT_ARG=""
if [ -n "$KITFT_AV" ]; then
    KITFT_ARG="--kitft-av-checkpoint $KITFT_AV"
fi

echo "==> NLA Validation"
echo "    Our AV: $OUR_AV"
echo "    AR:     $AR_CKPT"
[ -n "$KITFT_AV" ] && echo "    kitft AV: $KITFT_AV"

python -m nla_train.validate \
    --config configs/qwen7b_layer20.yaml \
    --our-av-checkpoint "$OUR_AV" \
    --ar-checkpoint "$AR_CKPT" \
    --nla-meta data/labeled/nla_meta_av.yaml \
    --data-dir data/train \
    --n-samples 100 \
    $KITFT_ARG

echo "==> Validation complete. Results: results/validation/validation_results.json"

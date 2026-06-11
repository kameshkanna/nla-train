#!/bin/bash
# Token-level NLA evaluation — Neuronpedia style.
# Shows per-token RMSE and explanations for the best-reconstructed tokens.
#
# Usage:
#   bash scripts/run_token_eval.sh "Your text here." [av_ckpt] [ar_ckpt]

set -euo pipefail

TEXT="${1:-Photo caption: A golden retriever puppy sitting in a field of sunflowers,}"
AV_CKPT="${2:-checkpoints/grpo/final_av}"
AR_CKPT="${3:-checkpoints/ar_sft/final}"

export CUDA_VISIBLE_DEVICES=0

echo "==> Token-level NLA evaluation"
echo "    Text: $TEXT"
echo "    AV: $AV_CKPT"

echo ""
echo "--- Pass 1: full token sweep ---"
python -m nla_train.token_eval \
    --config configs/qwen7b_layer20.yaml \
    --av-checkpoint "$AV_CKPT" \
    --ar-checkpoint "$AR_CKPT" \
    --nla-meta data/labeled/nla_meta_av.yaml \
    --text "$TEXT" \
    --top-k 5

echo ""
echo "--- Pass 2: focus on last comma (planning probe — what does model plan to complete?) ---"
python -m nla_train.token_eval \
    --config configs/qwen7b_layer20.yaml \
    --av-checkpoint "$AV_CKPT" \
    --ar-checkpoint "$AR_CKPT" \
    --nla-meta data/labeled/nla_meta_av.yaml \
    --text "$TEXT" \
    --focus-token "," \
    --max-new-tokens 150 \
    --min-position 0

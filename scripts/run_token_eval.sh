#!/bin/bash
# Token-level NLA evaluation — Neuronpedia style.
# Shows per-token RMSE and explanations for the best-reconstructed tokens.
#
# Usage:
#   bash scripts/run_token_eval.sh "Your text here." [av_ckpt] [ar_ckpt]

set -euo pipefail

# Default: poetry planning probe — check if model plans rhyme at newline after first line.
# The activation at '\n' after "grab it" should explain the model is planning "rabbit" rhyme.
TEXT="${1:-A rhyming couplet: He saw a carrot and had to grab it,}"
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
echo "--- Pass 2: focus on last comma (planning probe) ---"
python -m nla_train.token_eval \
    --config configs/qwen7b_layer20.yaml \
    --av-checkpoint "$AV_CKPT" \
    --ar-checkpoint "$AR_CKPT" \
    --nla-meta data/labeled/nla_meta_av.yaml \
    --text "$TEXT" \
    --focus-token "," \
    --max-new-tokens 150

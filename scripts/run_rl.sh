#!/bin/bash
# RL GRPO training (joint AV + AR).
#
# Usage:
#   bash scripts/run_rl.sh [config] [av_checkpoint] [ar_checkpoint]

set -euo pipefail

CONFIG="${1:-configs/qwen7b_layer20.yaml}"
AV_CKPT="${2:-checkpoints/av_sft/final}"
AR_CKPT="${3:-checkpoints/ar_sft/final}"

export PYTORCH_ALLOC_CONF=expandable_segments:True
# Single GPU: Trainer sees n_gpu=1, no DataParallel. AV + vLLM colocate + AR all on cuda:0.
# 80GB H100: AV ~15GB + AR ~10.5GB + vLLM 35% (~28GB) + grads ~5GB ≈ 58GB total.
export CUDA_VISIBLE_DEVICES=0
echo "==> RL GRPO: Joint AV + AR training"
echo "    AV checkpoint: $AV_CKPT"
echo "    AR checkpoint: $AR_CKPT"

accelerate launch \
    --config_file accelerate_config.yaml \
    -m nla_train.rl_grpo \
    --config "$CONFIG" \
    --data-dir data/train \
    --nla-meta data/labeled/nla_meta_av.yaml \
    --av-checkpoint "$AV_CKPT" \
    --ar-checkpoint "$AR_CKPT"

echo "==> RL GRPO complete."
echo "    Final AV: checkpoints/grpo/final_av"
echo "    Final AR: checkpoints/grpo/final_ar"

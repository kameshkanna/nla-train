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
# Both GPUs visible: GPU 0 = AV + vLLM colocate, GPU 1 = AR reward model.
# Simulate a 1-process distributed job: Trainer sees local_rank=0 → uses cuda:0 only,
# skips nn.DataParallel wrapping. WORLD_SIZE=1 → torch.distributed init succeeds with
# no peers. Both GPUs remain visible so AR can use cuda:1.
export CUDA_VISIBLE_DEVICES=0,1
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=29500
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

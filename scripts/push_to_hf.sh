#!/bin/bash
# Push AV checkpoint + model card to HuggingFace Hub.
#
# Prerequisites:
#   huggingface-cli login   (run once, needs HF write token)
#
# Usage:
#   bash scripts/push_to_hf.sh [hf_repo_id] [av_checkpoint]
#
# Example:
#   bash scripts/push_to_hf.sh kameshkanna/nla-qwen2.5-7b-L20-av checkpoints/grpo/final_av

set -euo pipefail

HF_REPO="${1:-kameshkanna/nla-qwen2.5-7b-L20-av}"
AV_CKPT="${2:-checkpoints/grpo/final_av}"
META="data/labeled/nla_meta_av.yaml"

echo "==> Pushing NLA AV checkpoint to HuggingFace"
echo "    Repo:       $HF_REPO"
echo "    Checkpoint: $AV_CKPT"

# Copy model card into checkpoint dir for upload
cp MODEL_CARD.md "$AV_CKPT/README.md"

# Copy nla_meta.yaml so users can load injection params
cp "$META" "$AV_CKPT/nla_meta.yaml"

# Upload
python - <<EOF
from huggingface_hub import HfApi
import os

api = HfApi()
repo_id = "$HF_REPO"

# Create repo if it doesn't exist
try:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"Repo ready: {repo_id}")
except Exception as e:
    print(f"Repo create: {e}")

# Upload checkpoint folder
api.upload_folder(
    folder_path="$AV_CKPT",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload NLA Qwen2.5-7B Layer 20 AV checkpoint",
)
print(f"Done: https://huggingface.co/{repo_id}")
EOF

echo "==> Upload complete: https://huggingface.co/$HF_REPO"

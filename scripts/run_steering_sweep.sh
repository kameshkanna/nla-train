#!/usr/bin/env bash
# Sweep steering × AV evaluation across k_scale values and probe layers.
#
# K follows the actbak norm-profile formula: K_ℓ = mean_norm_ℓ/√d × k_scale
# At L20 (mean_norm≈320, d=3584): k_scale=1 → K≈5.35, k_scale=2 → K≈10.7, etc.
#
# Usage:
#   bash scripts/run_steering_sweep.sh [OPTIONS]
#
# Options:
#   --layers      Space-separated layer list  (default: "18 19 20 21 22")
#   --k-scales    Space-separated k_scale values (default: "1.0 2.0 3.0 5.0")
#   --config      Path to yaml config           (default: configs/qwen7b_layer20.yaml)
#   --av-ckpt     Path to AV checkpoint         (default: checkpoints/grpo/final_av)
#   --nla-meta    Path to nla_meta_av.yaml       (default: data/labeled/nla_meta_av.yaml)
#   --out-dir     Output directory              (default: experiments/results)
#   --batch       Base batch size               (default: 4096)
#   --av-batch    AV batch size                 (default: 4096)
#   --single-layer Run each layer as a separate call (avoids OOM on large layer sets)
#
# Examples:
#   # Full 5-layer sweep, default k_scale grid (4 runs)
#   bash scripts/run_steering_sweep.sh
#
#   # Quick single-layer validation
#   bash scripts/run_steering_sweep.sh --layers "20" --k-scales "1.0 2.0"
#
#   # All layers as separate calls (lower peak memory)
#   bash scripts/run_steering_sweep.sh --single-layer

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
LAYERS="18 19 20 21 22"
K_SCALES="1.0 2.0 3.0 5.0"
CONFIG="configs/qwen7b_layer20.yaml"
AV_CKPT="checkpoints/grpo/final_av"
NLA_META="data/labeled/nla_meta_av.yaml"
OUT_DIR="experiments/results"
BATCH=4096
AV_BATCH=4096
SINGLE_LAYER=0

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --layers)       LAYERS="$2";   shift 2 ;;
        --k-scales)     K_SCALES="$2"; shift 2 ;;
        --config)       CONFIG="$2";   shift 2 ;;
        --av-ckpt)      AV_CKPT="$2";  shift 2 ;;
        --nla-meta)     NLA_META="$2"; shift 2 ;;
        --out-dir)      OUT_DIR="$2";  shift 2 ;;
        --batch)        BATCH="$2";    shift 2 ;;
        --av-batch)     AV_BATCH="$2"; shift 2 ;;
        --single-layer) SINGLE_LAYER=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

read -ra LAYER_ARR <<< "$LAYERS"
read -ra KS_ARR    <<< "$K_SCALES"

TOTAL=${#KS_ARR[@]}
echo "Steering sweep (actbak k_scale method)"
echo "  k_scales:  $K_SCALES"
echo "  layers:    $LAYERS"
echo "  batch:     $BATCH  av_batch: $AV_BATCH"
echo "  total runs: $TOTAL"
echo ""

RUN=0
for KS in "${KS_ARR[@]}"; do
    RUN=$(( RUN + 1 ))
    echo "━━━ Run $RUN/$TOTAL  k_scale=$KS ━━━"

    if [[ $SINGLE_LAYER -eq 1 ]]; then
        for L in "${LAYER_ARR[@]}"; do
            echo "  → layer $L"
            python experiments/steering_av_eval.py \
                --config      "$CONFIG" \
                --av-checkpoint "$AV_CKPT" \
                --nla-meta    "$NLA_META" \
                --output-dir  "$OUT_DIR" \
                --probe-layers "$L" \
                --k-scale     "$KS" \
                --base-batch-size "$BATCH" \
                --av-batch-size   "$AV_BATCH"
        done
    else
        python experiments/steering_av_eval.py \
            --config      "$CONFIG" \
            --av-checkpoint "$AV_CKPT" \
            --nla-meta    "$NLA_META" \
            --output-dir  "$OUT_DIR" \
            --probe-layers "${LAYER_ARR[@]}" \
            --k-scale     "$KS" \
            --base-batch-size "$BATCH" \
            --av-batch-size   "$AV_BATCH"
    fi

    echo "  ✓ done  →  steering_eval_ks${KS}_L${LAYERS// /-}.json"
    echo ""
done

echo "Sweep complete — $RUN runs saved to $OUT_DIR"

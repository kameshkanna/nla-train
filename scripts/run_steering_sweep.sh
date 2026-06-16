#!/usr/bin/env bash
# Sweep steering × AV evaluation across safety-k, french-k, and probe layers.
#
# Usage:
#   bash scripts/run_steering_sweep.sh [OPTIONS]
#
# Options:
#   --layers      Space-separated layer list  (default: "18 19 20 21 22")
#   --safety-ks   Space-separated safety K values (default: "10 20 30 40")
#   --french-ks   Space-separated french K values (default: "5 10 15 20")
#   --config      Path to yaml config           (default: configs/qwen7b_layer20.yaml)
#   --av-ckpt     Path to AV checkpoint         (default: checkpoints/grpo/final_av)
#   --nla-meta    Path to nla_meta_av.yaml       (default: data/labeled/nla_meta_av.yaml)
#   --out-dir     Output directory              (default: experiments/results)
#   --batch       Base batch size               (default: 1024)
#   --av-batch    AV batch size                 (default: 1024)
#   --single-layer Run each layer independently (one call per layer, faster on OOM)
#
# Examples:
#   # Full 5-layer sweep, default k grid
#   bash scripts/run_steering_sweep.sh
#
#   # Quick single-layer probe to validate vectors
#   bash scripts/run_steering_sweep.sh --layers "20" --safety-ks "20 30" --french-ks "10 15"
#
#   # All layers independently (avoids OOM when running all 5 together)
#   bash scripts/run_steering_sweep.sh --single-layer

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
LAYERS="18 19 20 21 22"
SAFETY_KS="10 20 30 40"
FRENCH_KS="5 10 15 20"
CONFIG="configs/qwen7b_layer20.yaml"
AV_CKPT="checkpoints/grpo/final_av"
NLA_META="data/labeled/nla_meta_av.yaml"
OUT_DIR="experiments/results"
BATCH=1024
AV_BATCH=1024
SINGLE_LAYER=0

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --layers)      LAYERS="$2";      shift 2 ;;
        --safety-ks)   SAFETY_KS="$2";   shift 2 ;;
        --french-ks)   FRENCH_KS="$2";   shift 2 ;;
        --config)      CONFIG="$2";      shift 2 ;;
        --av-ckpt)     AV_CKPT="$2";     shift 2 ;;
        --nla-meta)    NLA_META="$2";    shift 2 ;;
        --out-dir)     OUT_DIR="$2";     shift 2 ;;
        --batch)       BATCH="$2";       shift 2 ;;
        --av-batch)    AV_BATCH="$2";    shift 2 ;;
        --single-layer) SINGLE_LAYER=1; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

read -ra LAYER_ARR  <<< "$LAYERS"
read -ra SAFETY_ARR <<< "$SAFETY_KS"
read -ra FRENCH_ARR <<< "$FRENCH_KS"

TOTAL=$(( ${#SAFETY_ARR[@]} * ${#FRENCH_ARR[@]} ))
echo "Steering sweep: ${#SAFETY_ARR[@]} safety-k × ${#FRENCH_ARR[@]} french-k = $TOTAL combinations"
echo "  Layers: $LAYERS"
echo "  Safety K values:  $SAFETY_KS"
echo "  French K values:  $FRENCH_KS"
echo ""

RUN=0
for SK in "${SAFETY_ARR[@]}"; do
    for FK in "${FRENCH_ARR[@]}"; do
        RUN=$(( RUN + 1 ))
        echo "━━━ Run $RUN/$TOTAL  safety-k=$SK  french-k=$FK ━━━"

        if [[ $SINGLE_LAYER -eq 1 ]]; then
            # One call per layer — avoids capturing all layers in one forward pass
            for L in "${LAYER_ARR[@]}"; do
                echo "  → layer $L"
                python experiments/steering_av_eval.py \
                    --config "$CONFIG" \
                    --av-checkpoint "$AV_CKPT" \
                    --nla-meta "$NLA_META" \
                    --output-dir "$OUT_DIR" \
                    --probe-layers "$L" \
                    --safety-k "$SK" \
                    --french-k "$FK" \
                    --base-batch-size "$BATCH" \
                    --av-batch-size "$AV_BATCH"
            done
        else
            # All layers in one call — faster, uses more memory
            python experiments/steering_av_eval.py \
                --config "$CONFIG" \
                --av-checkpoint "$AV_CKPT" \
                --nla-meta "$NLA_META" \
                --output-dir "$OUT_DIR" \
                --probe-layers "${LAYER_ARR[@]}" \
                --safety-k "$SK" \
                --french-k "$FK" \
                --base-batch-size "$BATCH" \
                --av-batch-size "$AV_BATCH"
        fi

        echo "  ✓ done"
        echo ""
    done
done

echo "Sweep complete — $RUN runs saved to $OUT_DIR"

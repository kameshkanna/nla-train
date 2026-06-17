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
# K selection follows the actbak norm-profile formula:
#   K_ℓ = (mean_norm_ℓ / √d) × k_scale
# where mean_norm is measured over 80 prompts and d=3584 (Qwen2.5-7B hidden size).
# At L20: base K ≈ 5.35, so k_scale=1.0 injects ~1.7% of residual norm.
# Actbak's published ramp eval sweeps k_scale 0.5→3.0; we extend to 5.0 to
# check if AV descriptions shift at stronger interventions.
# --safety-k / --french-k are absolute overrides that bypass this formula entirely.
LAYERS="18 19 20 21 22"
K_SCALES="1.0 2.0 3.0 5.0"
SAFETY_KS=""   # empty = use K_SCALES × norm-profile (actbak method)
FRENCH_KS=""   # empty = use K_SCALES × norm-profile
CONFIG="configs/qwen7b_layer20.yaml"
AV_CKPT="checkpoints/grpo/final_av"
NLA_META="data/labeled/nla_meta_av.yaml"
OUT_DIR="experiments/results"
BATCH=2048
AV_BATCH=2048
SINGLE_LAYER=0

# ── Arg parsing ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --layers)       LAYERS="$2";     shift 2 ;;
        --k-scales)     K_SCALES="$2";   shift 2 ;;
        --safety-ks)    SAFETY_KS="$2";  shift 2 ;;
        --french-ks)    FRENCH_KS="$2";  shift 2 ;;
        --config)       CONFIG="$2";     shift 2 ;;
        --av-ckpt)      AV_CKPT="$2";    shift 2 ;;
        --nla-meta)     NLA_META="$2";   shift 2 ;;
        --out-dir)      OUT_DIR="$2";    shift 2 ;;
        --batch)        BATCH="$2";      shift 2 ;;
        --av-batch)     AV_BATCH="$2";   shift 2 ;;
        --single-layer) SINGLE_LAYER=1;  shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

read -ra LAYER_ARR <<< "$LAYERS"

# If --safety-ks / --french-ks are given, use them as absolute K overrides.
# Otherwise sweep k_scale values (actbak method: K_ℓ = profile_K_ℓ × k_scale).
if [[ -n "$SAFETY_KS" || -n "$FRENCH_KS" ]]; then
    MODE="absolute"
    [[ -z "$SAFETY_KS" ]] && SAFETY_KS="$K_SCALES"
    [[ -z "$FRENCH_KS" ]] && FRENCH_KS="$K_SCALES"
    read -ra SAFETY_ARR <<< "$SAFETY_KS"
    read -ra FRENCH_ARR <<< "$FRENCH_KS"
    TOTAL=$(( ${#SAFETY_ARR[@]} * ${#FRENCH_ARR[@]} ))
    echo "Steering sweep (ABSOLUTE K): ${#SAFETY_ARR[@]} safety-k × ${#FRENCH_ARR[@]} french-k = $TOTAL combinations"
    echo "  Layers:   $LAYERS"
    echo "  Safety K: $SAFETY_KS"
    echo "  French K: $FRENCH_KS"
else
    MODE="scale"
    read -ra KS_ARR <<< "$K_SCALES"
    TOTAL=${#KS_ARR[@]}
    echo "Steering sweep (actbak K = profile_K × k_scale): $TOTAL k_scale values"
    echo "  Layers:   $LAYERS"
    echo "  k_scales: $K_SCALES"
    echo "  (L20 base K ≈ 5.35 → effective K at scales: $(python3 -c "
ks=[${K_SCALES// /,}]; print(', '.join(f'{k}×5.35={k*5.35:.1f}' for k in ks))
" 2>/dev/null || echo 'see norm profile')"
fi
echo ""

run_one() {
    local extra_args="$1"
    local label="$2"
    echo "━━━ $label ━━━"
    if [[ $SINGLE_LAYER -eq 1 ]]; then
        for L in "${LAYER_ARR[@]}"; do
            echo "  → layer $L"
            python experiments/steering_av_eval.py \
                --config "$CONFIG" \
                --av-checkpoint "$AV_CKPT" \
                --nla-meta "$NLA_META" \
                --output-dir "$OUT_DIR" \
                --probe-layers "$L" \
                --base-batch-size "$BATCH" \
                --av-batch-size "$AV_BATCH" \
                $extra_args
        done
    else
        python experiments/steering_av_eval.py \
            --config "$CONFIG" \
            --av-checkpoint "$AV_CKPT" \
            --nla-meta "$NLA_META" \
            --output-dir "$OUT_DIR" \
            --probe-layers "${LAYER_ARR[@]}" \
            --base-batch-size "$BATCH" \
            --av-batch-size "$AV_BATCH" \
            $extra_args
    fi
    echo "  ✓ done"
    echo ""
}

RUN=0
if [[ "$MODE" == "scale" ]]; then
    for KS in "${KS_ARR[@]}"; do
        RUN=$(( RUN + 1 ))
        run_one "--k-scale $KS" "Run $RUN/$TOTAL  k_scale=$KS"
    done
else
    for SK in "${SAFETY_ARR[@]}"; do
        for FK in "${FRENCH_ARR[@]}"; do
            RUN=$(( RUN + 1 ))
            run_one "--safety-k $SK --french-k $FK" \
                    "Run $RUN/$TOTAL  safety-k=$SK  french-k=$FK"
        done
    done
fi

echo "Sweep complete — $RUN runs saved to $OUT_DIR"

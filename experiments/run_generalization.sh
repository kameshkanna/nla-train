#!/bin/bash
# End-to-end cross-layer generalization experiment.
#
# Runs Steps 1-5 in sequence. Each step is idempotent (skips if output exists).
#
# Usage:
#   bash experiments/run_generalization.sh \
#       [--config CONFIG] \
#       [--av-checkpoint AV_CKPT] \
#       [--ar-checkpoint AR_CKPT] \
#       [--nla-meta NLA_META] \
#       [--data-dir DATA_DIR] \
#       [--results-dir RESULTS_DIR] \
#       [--figures-dir FIGURES_DIR] \
#       [--n-per-domain N] \
#       [--av-batch-size N] \
#       [--ar-batch-size N]
#
# Defaults assume you run from the repo root (nla-train/).

set -euo pipefail

CONFIG="configs/qwen7b_layer20.yaml"
AV_CKPT="checkpoints/grpo/final_av"
AR_CKPT="checkpoints/ar_sft/final"
NLA_META="data/labeled/nla_meta_av.yaml"
DATA_DIR="experiments/data"
RESULTS_DIR="experiments/results"
FIGURES_DIR="experiments/figures"
N_PER_DOMAIN=400
AV_BATCH=8
AR_BATCH=32

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)        CONFIG="$2";        shift 2 ;;
        --av-checkpoint) AV_CKPT="$2";      shift 2 ;;
        --ar-checkpoint) AR_CKPT="$2";      shift 2 ;;
        --nla-meta)      NLA_META="$2";     shift 2 ;;
        --data-dir)      DATA_DIR="$2";     shift 2 ;;
        --results-dir)   RESULTS_DIR="$2";  shift 2 ;;
        --figures-dir)   FIGURES_DIR="$2";  shift 2 ;;
        --n-per-domain)  N_PER_DOMAIN="$2"; shift 2 ;;
        --av-batch-size) AV_BATCH="$2";     shift 2 ;;
        --ar-batch-size) AR_BATCH="$2";     shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "======================================================================"
echo "NLA Cross-Layer Generalization Experiment"
echo "======================================================================"
echo "  Config:        $CONFIG"
echo "  AV checkpoint: $AV_CKPT"
echo "  AR checkpoint: $AR_CKPT"
echo "  NLA meta:      $NLA_META"
echo "  Data dir:      $DATA_DIR"
echo "  Results dir:   $RESULTS_DIR"
echo "  Figures dir:   $FIGURES_DIR"
echo "  N per domain:  $N_PER_DOMAIN  (total $(( N_PER_DOMAIN * 5 )) texts)"
echo "======================================================================"

mkdir -p "$DATA_DIR" "$RESULTS_DIR" "$FIGURES_DIR"

# ---- Step 1: Extract all-layer activations ----
if [[ -f "$DATA_DIR/activations.npy" && -f "$DATA_DIR/meta.json" ]]; then
    echo ""
    echo "[SKIP] Step 1 — activations.npy already exists"
else
    echo ""
    echo "[STEP 1] Extracting activations from all 28 layers..."
    python experiments/extract_all_layers.py \
        --config "$CONFIG" \
        --output-dir "$DATA_DIR" \
        --n-per-domain "$N_PER_DOMAIN"
fi

# ---- Step 2: AV sweep — generate descriptions ----
if [[ -f "$DATA_DIR/descriptions_raw.json" && -f "$DATA_DIR/descriptions_normalized.json" ]]; then
    echo ""
    echo "[SKIP] Step 2 — descriptions already exist"
else
    echo ""
    echo "[STEP 2] Running AV sweep ($(( N_PER_DOMAIN * 5 )) texts × 28 layers × 2 arms)..."
    python experiments/run_av_sweep.py \
        --config "$CONFIG" \
        --data-dir "$DATA_DIR" \
        --av-checkpoint "$AV_CKPT" \
        --nla-meta "$NLA_META" \
        --output-dir "$DATA_DIR" \
        --batch-size "$AV_BATCH"
fi

# ---- Step 3: AR sweep — reconstruct from descriptions ----
RECON_DONE=true
for arm in raw normalized; do
    [[ -f "$DATA_DIR/reconstructions_${arm}.npy" ]] || RECON_DONE=false
done
for bl in baseline_random baseline_shuffled baseline_mean; do
    [[ -f "$DATA_DIR/${bl}.npy" ]] || RECON_DONE=false
done

if $RECON_DONE; then
    echo ""
    echo "[SKIP] Step 3 — reconstructions and baselines already exist"
else
    echo ""
    echo "[STEP 3] Running AR reconstruction sweep..."
    python experiments/run_ar_sweep.py \
        --config "$CONFIG" \
        --data-dir "$DATA_DIR" \
        --ar-checkpoint "$AR_CKPT" \
        --output-dir "$DATA_DIR" \
        --batch-size "$AR_BATCH"
fi

# ---- Step 4: Compute metrics ----
if [[ -f "$RESULTS_DIR/metrics.json" && -f "$RESULTS_DIR/wilcoxon_results.json" ]]; then
    echo ""
    echo "[SKIP] Step 4 — metrics.json already exists"
else
    echo ""
    echo "[STEP 4] Computing metrics (CS, FVE, Recall, nRMSE, Wilcoxon+BH)..."
    python experiments/compute_metrics.py \
        --data-dir "$DATA_DIR" \
        --output-dir "$RESULTS_DIR"
fi

# ---- Step 5: Generate figures ----
echo ""
echo "[STEP 5] Generating figures..."
python experiments/plot_results.py \
    --data-dir "$DATA_DIR" \
    --results-dir "$RESULTS_DIR" \
    --output-dir "$FIGURES_DIR"

echo ""
echo "======================================================================"
echo "DONE — Cross-layer generalization experiment complete"
echo ""
echo "Key outputs:"
echo "  Metrics:  $RESULTS_DIR/metrics.json"
echo "  Wilcoxon: $RESULTS_DIR/wilcoxon_results.json"
echo "  Figures:  $FIGURES_DIR/"
ls -lh "$FIGURES_DIR"/*.png 2>/dev/null || true
echo "======================================================================"

"""
Step 5: Generate all paper-ready figures from metrics.json.

Produces 6 figures:
  fig1_layer_profile.png   — Line plots: CS, FVE, nRMSE, Recall@10 across all 28 layers
                             for raw, normalized, baselines. Markers at layer 20 (training layer).
  fig2_domain_heatmap.png  — Heatmap: CS per (layer × domain) for normalized arm.
  fig3_clc_matrix.png      — Cross-Layer Consistency matrix:
                             CLC[i,j] = mean CS between layer-i AV descriptions replayed through
                             layer-j AR reconstruction vs ground truth layer-j activations.
                             (uses reconstructions directly as proxy — no re-inference needed)
  fig4_pca_umap.png        — 2D PCA + UMAP of layer-20 vs layer-10, layer-15, layer-25
                             activations and reconstructions, colored by domain.
  fig5_norm_sensitivity.png — Effect of norm scaling: raw vs normalized CS across layers,
                              with shaded error band.
  fig6_qualitative_grid.png — 4 × 4 grid of sample descriptions across layers 5, 10, 15, 20, 25.

Usage:
    python experiments/plot_results.py \
        --data-dir experiments/data \
        --results-dir experiments/results \
        --output-dir experiments/figures
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

logger = logging.getLogger(__name__)

LAYER_COLORS = {
    "raw": "#2196F3",
    "normalized": "#4CAF50",
    "random": "#9E9E9E",
    "shuffled": "#FF9800",
    "mean_baseline": "#E91E63",
}

DOMAIN_COLORS = {
    "fineweb": "#1976D2",
    "wikipedia": "#388E3C",
    "pubmed": "#F57C00",
    "github": "#7B1FA2",
    "reddit": "#D32F2F",
}


def _set_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.8,
    })


def fig1_layer_profile(
    metrics: dict[str, Any],
    output_path: Path,
    training_layer: int = 20,
) -> None:
    """4-panel line plot: CS, FVE, nRMSE, Recall@10 across all layers."""
    n_layers = metrics["meta"]["n_layers"]
    layers = list(range(n_layers))

    arms = ["raw", "normalized", "random", "shuffled"]
    metric_keys = [
        ("cs_mean", "Cosine Similarity ↑"),
        ("fve", "FVE ↑"),
        ("nrmse", "nRMSE ↓"),
        ("recall@10", "Recall@10 ↑"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (metric_key, ylabel) in zip(axes, metric_keys):
        for arm in arms:
            vals = []
            for l in layers:
                layer_data = metrics["layers"].get(str(l), {})
                arm_data = layer_data.get(arm, {})
                vals.append(arm_data.get(metric_key, np.nan))

            color = LAYER_COLORS.get(arm, "black")
            ls = "--" if "baseline" in arm or arm in ("random", "shuffled") else "-"
            ax.plot(layers, vals, color=color, linestyle=ls, label=arm, alpha=0.85)

        ax.axvline(x=training_layer, color="red", linestyle=":", alpha=0.6, linewidth=1.5)
        ax.text(training_layer + 0.3, ax.get_ylim()[0], "L20\n(train)", fontsize=7.5,
                color="red", alpha=0.7, va="bottom")
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, n_layers - 1)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("AV Layer Generalization: Qwen2.5-7B (N=2000, 5 domains)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def fig2_domain_heatmap(
    metrics: dict[str, Any],
    output_path: Path,
    arm: str = "normalized",
) -> None:
    """CS heatmap: layers × domains for one arm."""
    n_layers = metrics["meta"]["n_layers"]
    domains = metrics["meta"]["domains"]
    layers = list(range(n_layers))

    mat = np.full((n_layers, len(domains)), np.nan)
    for l in layers:
        layer_data = metrics["layers"].get(str(l), {})
        arm_data = layer_data.get(arm, {})
        by_domain = arm_data.get("by_domain", {})
        for d_idx, domain in enumerate(domains):
            mat[l, d_idx] = by_domain.get(domain, {}).get("cs_mean", np.nan)

    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(domains)))
    ax.set_xticklabels(domains, rotation=30, ha="right")
    ax.set_yticks(range(0, n_layers, 4))
    ax.set_yticklabels([f"L{l}" for l in range(0, n_layers, 4)])
    ax.set_ylabel("Layer")
    ax.set_xlabel("Domain")
    ax.set_title(f"Cosine Similarity by Layer × Domain ({arm} arm)", fontsize=12, fontweight="bold")

    # Mark training layer
    ax.axhline(y=20, color="red", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(len(domains) - 0.5, 20.5, "L20", color="red", fontsize=8, ha="right")

    plt.colorbar(im, ax=ax, fraction=0.03, label="Cosine Similarity")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def fig3_clc_matrix(
    data_dir: Path,
    output_path: Path,
    layers_subset: list[int] | None = None,
) -> None:
    """
    Cross-Layer Consistency (CLC) matrix.

    CLC[i, j] = mean cosine similarity between:
      - reconstructions from arm at layer i
      - reconstructions from arm at layer j

    This shows how consistent the AV's description space is across layers —
    do descriptions of layer 5 activations produce reconstructions similar
    to descriptions of layer 20 activations?

    Uses reconstructions_normalized.npy since norms are comparable.
    """
    recon_path = data_dir / "reconstructions_normalized.npy"
    if not recon_path.exists():
        logger.warning("Missing reconstructions_normalized.npy — skipping CLC matrix")
        return

    recons = np.load(recon_path).astype(np.float32)  # (N, n_layers, d)
    N, n_layers, d = recons.shape

    if layers_subset is None:
        layers_subset = list(range(n_layers))

    L = len(layers_subset)
    clc = np.zeros((L, L), dtype=np.float32)

    for i_idx, i in enumerate(layers_subset):
        ri = recons[:, i]
        ri_norm = ri / (np.linalg.norm(ri, axis=1, keepdims=True) + 1e-8)
        for j_idx, j in enumerate(layers_subset):
            rj = recons[:, j]
            rj_norm = rj / (np.linalg.norm(rj, axis=1, keepdims=True) + 1e-8)
            clc[i_idx, j_idx] = float(np.mean(np.sum(ri_norm * rj_norm, axis=1)))

    layer_labels = [f"L{l}" for l in layers_subset]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(clc, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(L))
    ax.set_yticks(range(L))
    ax.set_xticklabels(layer_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(layer_labels, fontsize=8)
    ax.set_title("Cross-Layer Consistency (CLC) Matrix — Normalized Arm", fontsize=12, fontweight="bold")

    # Mark L20
    if 20 in layers_subset:
        idx20 = layers_subset.index(20)
        ax.axhline(y=idx20, color="red", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axvline(x=idx20, color="red", linestyle="--", linewidth=1.0, alpha=0.6)

    plt.colorbar(im, ax=ax, fraction=0.03, label="Pairwise Cosine Similarity")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def fig4_pca_scatter(
    data_dir: Path,
    output_path: Path,
    probe_layers: list[int] | None = None,
    n_components: int = 2,
    max_n: int = 200,
) -> None:
    """
    PCA scatter: ground truth vs reconstructed activations for select layers.
    Colored by domain; shape distinguishes GT vs reconstructed.
    """
    probe_layers = probe_layers or [5, 10, 15, 20, 25]

    acts = np.load(data_dir / "activations.npy").astype(np.float32)
    recon_path = data_dir / "reconstructions_normalized.npy"
    if not recon_path.exists():
        logger.warning("Missing reconstructions_normalized.npy — skipping PCA scatter")
        return

    recons = np.load(recon_path).astype(np.float32)

    with open(data_dir / "texts.json") as f:
        texts_data = json.load(f)
    with open(data_dir / "meta.json") as f:
        meta = json.load(f)

    domain_labels = texts_data["domain_labels"]
    domains = meta["domains"]

    N = min(len(acts), max_n)
    rng = np.random.default_rng(42)
    idx = rng.choice(len(acts), N, replace=False)
    domain_sub = [domain_labels[i] for i in idx]

    n_plots = len(probe_layers)
    fig, axes = plt.subplots(2, (n_plots + 1) // 2, figsize=(5 * ((n_plots + 1) // 2), 9))
    axes = np.array(axes).flatten()

    from sklearn.decomposition import PCA

    for ax_idx, layer in enumerate(probe_layers):
        ax = axes[ax_idx]
        gt = acts[idx, layer]
        rc = recons[idx, layer]

        combined = np.concatenate([gt, rc], axis=0)
        pca = PCA(n_components=n_components, random_state=42)
        combined_2d = pca.fit_transform(combined)
        gt_2d = combined_2d[:N]
        rc_2d = combined_2d[N:]

        for domain in domains:
            d_idx = [i for i, d in enumerate(domain_sub) if d == domain]
            color = DOMAIN_COLORS.get(domain, "black")
            ax.scatter(gt_2d[d_idx, 0], gt_2d[d_idx, 1], c=color, marker="o", s=20,
                       alpha=0.6, label=f"{domain} (GT)" if ax_idx == 0 else None)
            ax.scatter(rc_2d[d_idx, 0], rc_2d[d_idx, 1], c=color, marker="x", s=20,
                       alpha=0.4, label=f"{domain} (RC)" if ax_idx == 0 else None)

        expl = pca.explained_variance_ratio_.sum()
        ax.set_title(f"Layer {layer} (PCA var={expl:.2f})", fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

    # Hide unused axes
    for ax_idx in range(len(probe_layers), len(axes)):
        axes[ax_idx].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(domains), fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("PCA: GT (circles) vs Reconstructed (×) Activations by Domain", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def fig5_norm_sensitivity(
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    """Overlay of raw vs normalized CS with error bands across all layers."""
    n_layers = metrics["meta"]["n_layers"]
    layers = list(range(n_layers))

    fig, ax = plt.subplots(figsize=(10, 5))

    for arm in ["raw", "normalized"]:
        means, stds = [], []
        for l in layers:
            layer_data = metrics["layers"].get(str(l), {})
            arm_data = layer_data.get(arm, {})
            means.append(arm_data.get("cs_mean", np.nan))
            stds.append(arm_data.get("cs_std", np.nan))

        means, stds = np.array(means), np.array(stds)
        color = LAYER_COLORS[arm]
        ax.plot(layers, means, color=color, label=arm, linewidth=2)
        ax.fill_between(layers, means - stds, means + stds, color=color, alpha=0.15)

    ax.axvline(x=20, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label="L20 (train)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity (mean ± std)")
    ax.set_title("Norm Sensitivity: Raw vs Normalized Input Activations", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(0, n_layers - 1)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def fig6_qualitative_grid(
    data_dir: Path,
    output_path: Path,
    probe_layers: list[int] | None = None,
    n_samples: int = 4,
) -> None:
    """
    Text grid: sample descriptions at select layers.
    Rows = text samples, Cols = layers.
    """
    probe_layers = probe_layers or [5, 10, 15, 20, 25]
    desc_path = data_dir / "descriptions_normalized.json"
    if not desc_path.exists():
        logger.warning("Missing descriptions_normalized.json — skipping qualitative grid")
        return

    with open(desc_path) as f:
        all_descs = json.load(f)
    with open(data_dir / "texts.json") as f:
        texts_data = json.load(f)

    desc_map: dict[tuple[int, int], str] = {}
    for item in all_descs:
        desc_map[(item["text_id"], item["layer"])] = item["description"]

    texts = texts_data["texts"]
    sample_ids = list(range(min(n_samples, len(texts))))

    n_cols = len(probe_layers)
    n_rows = n_samples
    fig = plt.figure(figsize=(4 * n_cols, 2.5 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols, hspace=0.4, wspace=0.3)

    for row, text_id in enumerate(sample_ids):
        input_snippet = texts[text_id][:80] + "..." if len(texts[text_id]) > 80 else texts[text_id]
        for col, layer in enumerate(probe_layers):
            ax = fig.add_subplot(gs[row, col])
            desc = desc_map.get((text_id, layer), "(no description)")
            # Truncate for display
            desc_short = desc[:200] + "..." if len(desc) > 200 else desc

            ax.text(0.05, 0.95, f"L{layer}", transform=ax.transAxes,
                    fontsize=9, fontweight="bold", va="top", color="#333333")
            ax.text(0.05, 0.80, desc_short, transform=ax.transAxes,
                    fontsize=7, va="top", wrap=True,
                    bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F5F5F5", "alpha": 0.8})

            if col == 0:
                ax.set_ylabel(f"Sample {text_id}\n{input_snippet[:40]}...",
                              fontsize=6.5, rotation=0, labelpad=60, va="center")
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)

    fig.suptitle("Qualitative Descriptions Across Layers (Normalized Arm)",
                 fontsize=12, fontweight="bold", y=1.01)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", output_path)


def plot_results(
    data_dir: str,
    results_dir: str,
    output_dir: str,
) -> None:
    """
    Generate all 6 figures.

    Args:
        data_dir: Directory with activations.npy, descriptions*.json, reconstructions*.npy.
        results_dir: Directory with metrics.json (from compute_metrics.py).
        output_dir: Where to save figures.
    """
    _set_style()

    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metrics_path = Path(results_dir) / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found at {metrics_path}. Run compute_metrics.py first.")

    with open(metrics_path) as f:
        metrics = json.load(f)

    n_layers = metrics["meta"]["n_layers"]

    logger.info("Generating fig1: layer profile")
    fig1_layer_profile(metrics, out_path / "fig1_layer_profile.png")

    logger.info("Generating fig2: domain heatmap")
    fig2_domain_heatmap(metrics, out_path / "fig2_domain_heatmap.png")

    logger.info("Generating fig3: CLC matrix")
    layers_for_clc = list(range(0, n_layers, 2))  # every 2nd layer to keep matrix legible
    fig3_clc_matrix(data_path, out_path / "fig3_clc_matrix.png", layers_subset=layers_for_clc)

    logger.info("Generating fig4: PCA scatter")
    try:
        fig4_pca_scatter(data_path, out_path / "fig4_pca_scatter.png",
                         probe_layers=[5, 10, 15, 20, 25])
    except ImportError:
        logger.warning("scikit-learn not installed — skipping PCA scatter. pip install scikit-learn")

    logger.info("Generating fig5: norm sensitivity")
    fig5_norm_sensitivity(metrics, out_path / "fig5_norm_sensitivity.png")

    logger.info("Generating fig6: qualitative grid")
    fig6_qualitative_grid(data_path, out_path / "fig6_qualitative_grid.png",
                          probe_layers=[5, 10, 15, 20, 25])

    logger.info("All figures saved to %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate cross-layer generalization figures")
    p.add_argument("--data-dir", default="experiments/data")
    p.add_argument("--results-dir", default="experiments/results")
    p.add_argument("--output-dir", default="experiments/figures")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    plot_results(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

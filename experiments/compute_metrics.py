"""
Step 4: Compute cross-layer generalization metrics.

Loads ground-truth activations, reconstructions (raw + normalized), and baselines,
then computes per-layer metrics with domain stratification, statistical tests,
and diversity analysis.

Metrics computed:
  - Cosine Similarity (CS)
  - Fraction of Variance Explained (FVE)
  - Rank Retrieval Recall@1, Recall@10 (description → activation)
  - nRMSE (normalized RMSE = RMSE / mean(||h||) per layer)
  - DIAS (Description-Invariant Activation Similarity) — intra-description diversity

Statistical tests (per metric × arm vs baselines):
  - Wilcoxon signed-rank test
  - Benjamini-Hochberg FDR correction across all layer × metric × arm comparisons

Output files (in --output-dir):
  metrics.json           — all metrics, all layers, all arms/baselines, domain breakdown
  wilcoxon_results.json  — p-values, BH-corrected q-values, significance flags

Usage:
    python experiments/compute_metrics.py \
        --data-dir experiments/data \
        --output-dir experiments/results
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _cosine_similarity(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Per-row cosine similarity. A, B: (N, d). Returns (N,)."""
    norm_a = np.linalg.norm(A, axis=1, keepdims=True)
    norm_b = np.linalg.norm(B, axis=1, keepdims=True)
    denom = norm_a * norm_b + 1e-8
    return np.sum(A * B, axis=1) / denom.squeeze()


def _fve(true: np.ndarray, pred: np.ndarray) -> float:
    """
    Fraction of Variance Explained across N samples.
    FVE = 1 - Var(true - pred) / Var(true)
    """
    residual_var = np.var(true - pred, axis=0).sum()
    total_var = np.var(true, axis=0).sum()
    return float(1.0 - residual_var / (total_var + 1e-12))


def _nrmse(true: np.ndarray, pred: np.ndarray) -> float:
    """nRMSE = RMSE / mean(||true||)."""
    rmse = float(np.sqrt(np.mean((true - pred) ** 2)))
    mean_norm = float(np.mean(np.linalg.norm(true, axis=1)))
    return rmse / (mean_norm + 1e-12)


def _rank_retrieval(
    descriptions_recons: np.ndarray,
    true_acts: np.ndarray,
) -> tuple[float, float]:
    """
    Recall@1 and Recall@10 for retrieval task:
    given a reconstructed activation from description i, find rank of true activation i
    among all N true activations.

    descriptions_recons: (N, d) — AR(description_i)
    true_acts: (N, d) — ground truth activations
    Returns: (recall@1, recall@10)
    """
    N = len(descriptions_recons)
    if N == 0:
        return 0.0, 0.0

    # Normalize for cosine similarity
    desc_norm = descriptions_recons / (np.linalg.norm(descriptions_recons, axis=1, keepdims=True) + 1e-8)
    act_norm = true_acts / (np.linalg.norm(true_acts, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity matrix (N, N)
    sim_matrix = desc_norm @ act_norm.T  # (N, N)

    # Rank of diagonal element (correct pair) — lower is better
    ranks_at1 = 0
    ranks_at10 = 0
    for i in range(N):
        row = sim_matrix[i]
        correct_sim = row[i]
        rank = int(np.sum(row > correct_sim))  # 0-indexed rank
        if rank == 0:
            ranks_at1 += 1
        if rank < 10:
            ranks_at10 += 1

    return ranks_at1 / N, ranks_at10 / N


def _dias(descriptions_recons: np.ndarray) -> float:
    """
    Description-Invariant Activation Similarity (DIAS):
    mean pairwise cosine similarity within a batch of reconstructed activations.
    High DIAS → AV is mode-collapsing (all descriptions map to similar vectors).
    Low DIAS → diverse, discriminative reconstructions.
    """
    N = len(descriptions_recons)
    if N < 2:
        return 0.0
    norms = np.linalg.norm(descriptions_recons, axis=1, keepdims=True)
    normed = descriptions_recons / (norms + 1e-8)
    sim_matrix = normed @ normed.T  # (N, N)
    # Upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(N, k=1)
    return float(np.mean(sim_matrix[triu_idx]))


def _bh_correction(p_values: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg FDR correction. Returns (q_values, is_significant)."""
    m = len(p_values)
    if m == 0:
        return [], []
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    ranks = np.arange(1, m + 1)
    bh_threshold = sorted_p * m / ranks
    q_values = np.zeros(m)
    for i in range(m - 1, -1, -1):
        q_values[sorted_idx[i]] = min(
            sorted_p[i] * m / ranks[i],
            q_values[sorted_idx[i + 1]] if i < m - 1 else 1.0,
        )
    q_values = np.minimum(q_values, 1.0)
    is_significant = [float(q) < alpha for q in q_values]
    return q_values.tolist(), is_significant


def compute_layer_metrics(
    true: np.ndarray,
    pred: np.ndarray,
    domain_labels: list[str],
    domains: list[str],
) -> dict[str, Any]:
    """
    Compute all metrics for one (layer, arm) pair.

    Args:
        true: (N, d) ground truth activations.
        pred: (N, d) reconstructed activations.
        domain_labels: list of domain names per sample.
        domains: full list of domain names.

    Returns:
        Dict with overall and per-domain metrics.
    """
    N = len(true)
    cs_per_sample = _cosine_similarity(true, pred)

    result: dict[str, Any] = {
        "n": N,
        "cs_mean": float(np.mean(cs_per_sample)),
        "cs_std": float(np.std(cs_per_sample)),
        "fve": _fve(true, pred),
        "nrmse": _nrmse(true, pred),
        "recall@1": _rank_retrieval(pred, true)[0],
        "recall@10": _rank_retrieval(pred, true)[1],
        "dias": _dias(pred),
        "cs_per_sample": cs_per_sample.tolist(),
        "by_domain": {},
    }

    for domain in domains:
        idx = [i for i, d in enumerate(domain_labels) if d == domain]
        if not idx:
            continue
        t_d = true[idx]
        p_d = pred[idx]
        cs_d = _cosine_similarity(t_d, p_d)
        result["by_domain"][domain] = {
            "n": len(idx),
            "cs_mean": float(np.mean(cs_d)),
            "cs_std": float(np.std(cs_d)),
            "fve": _fve(t_d, p_d),
            "nrmse": _nrmse(t_d, p_d),
            "recall@1": _rank_retrieval(p_d, t_d)[0],
            "recall@10": _rank_retrieval(p_d, t_d)[1],
        }

    return result


def compute_metrics(
    data_dir: str,
    output_dir: str,
    rank_retrieval_max_n: int = 500,
) -> None:
    """
    Load all arrays and compute the full metrics suite.

    Args:
        data_dir: Directory with activations.npy, reconstructions_*.npy, baselines, meta.json.
        output_dir: Where to write metrics.json and wilcoxon_results.json.
        rank_retrieval_max_n: Max N for O(N^2) rank retrieval (subsample if larger).
    """
    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    with open(data_path / "meta.json") as f:
        meta = json.load(f)
    with open(data_path / "texts.json") as f:
        texts_data = json.load(f)

    N: int = meta["n_samples"]
    n_layers: int = meta["n_layers"]
    domains: list[str] = meta["domains"]
    domain_labels: list[str] = texts_data["domain_labels"]

    logger.info("Loading activations (%d×%d×%d)", N, n_layers, meta["d_model"])
    activations = np.load(data_path / "activations.npy").astype(np.float32)

    arms = {
        "raw": data_path / "reconstructions_raw.npy",
        "normalized": data_path / "reconstructions_normalized.npy",
    }
    baselines = {
        "random": data_path / "baseline_random.npy",
        "shuffled": data_path / "baseline_shuffled.npy",
    }

    loaded_arrays: dict[str, np.ndarray] = {}
    for name, path in {**arms, **baselines}.items():
        if path.exists():
            loaded_arrays[name] = np.load(path).astype(np.float32)
            logger.info("Loaded %s %s", name, loaded_arrays[name].shape)
        else:
            logger.warning("Missing %s — skipping", path)

    mean_baseline = np.load(data_path / "baseline_mean.npy").astype(np.float32) if (data_path / "baseline_mean.npy").exists() else None

    # Subsample index for rank retrieval (O(N^2))
    rng = np.random.default_rng(42)
    if N > rank_retrieval_max_n:
        rr_idx = rng.choice(N, rank_retrieval_max_n, replace=False)
    else:
        rr_idx = np.arange(N)

    all_metrics: dict[str, Any] = {
        "meta": {
            "n_samples": N,
            "n_layers": n_layers,
            "domains": domains,
            "rank_retrieval_n": len(rr_idx),
        },
        "layers": {},
    }

    # Wilcoxon collection: (arm, layer, metric) → p-value for BH correction
    wilcoxon_tests: list[dict[str, Any]] = []

    for layer_idx in tqdm(range(n_layers), desc="Computing metrics per layer"):
        true_layer = activations[:, layer_idx]
        layer_results: dict[str, Any] = {"layer": layer_idx}

        for arm_name, arr in loaded_arrays.items():
            if arr.ndim == 3:
                pred_layer = arr[:, layer_idx]
            else:
                pred_layer = arr  # baseline_mean has shape (n_layers, d)

            metrics = compute_layer_metrics(true_layer, pred_layer, domain_labels, domains)

            # Add rank retrieval on subsampled set
            rr_true = true_layer[rr_idx]
            rr_pred = pred_layer[rr_idx]
            metrics["recall@1_rr"] = _rank_retrieval(rr_pred, rr_true)[0]
            metrics["recall@10_rr"] = _rank_retrieval(rr_pred, rr_true)[1]

            layer_results[arm_name] = metrics

            # Wilcoxon test: arm vs baseline_random (if available)
            if arm_name in arms and "random" in loaded_arrays:
                cs_arm = np.array(metrics["cs_per_sample"])
                cs_rand = _cosine_similarity(true_layer, loaded_arrays["random"][:, layer_idx])
                stat, pval = stats.wilcoxon(cs_arm - cs_rand, alternative="greater", zero_method="wilcox")
                wilcoxon_tests.append({
                    "arm": arm_name,
                    "layer": layer_idx,
                    "metric": "cs",
                    "vs": "random",
                    "statistic": float(stat),
                    "p_value": float(pval),
                })

            # Wilcoxon: arm vs shuffled baseline
            if arm_name in arms and "shuffled" in loaded_arrays:
                cs_arm = np.array(metrics["cs_per_sample"])
                cs_shuf = _cosine_similarity(true_layer, loaded_arrays["shuffled"][:, layer_idx])
                stat, pval = stats.wilcoxon(cs_arm - cs_shuf, alternative="greater", zero_method="wilcox")
                wilcoxon_tests.append({
                    "arm": arm_name,
                    "layer": layer_idx,
                    "metric": "cs",
                    "vs": "shuffled",
                    "statistic": float(stat),
                    "p_value": float(pval),
                })

        # Mean vector baseline (shape: d_model only, broadcast)
        if mean_baseline is not None:
            mean_vec = mean_baseline[layer_idx][None]  # (1, d_model)
            mean_pred = np.tile(mean_vec, (N, 1))
            layer_results["mean_baseline"] = compute_layer_metrics(
                true_layer, mean_pred, domain_labels, domains
            )

        all_metrics["layers"][str(layer_idx)] = layer_results

    # Apply BH correction to all Wilcoxon tests
    all_p = [t["p_value"] for t in wilcoxon_tests]
    q_values, is_sig = _bh_correction(all_p)
    for i, test in enumerate(wilcoxon_tests):
        test["q_value"] = q_values[i]
        test["significant"] = is_sig[i]

    # Summary: count significant layers per arm
    for arm_name in arms:
        sig_raw = [t for t in wilcoxon_tests if t["arm"] == arm_name and t["vs"] == "random" and t["significant"]]
        sig_shuf = [t for t in wilcoxon_tests if t["arm"] == arm_name and t["vs"] == "shuffled" and t["significant"]]
        logger.info(
            "Arm=%s: %d/%d layers significant vs random, %d/%d vs shuffled (BH q<0.05)",
            arm_name, len(sig_raw), n_layers, len(sig_shuf), n_layers,
        )

    # Persist
    metrics_path = out_path / "metrics.json"
    # Remove cs_per_sample from JSON (too large) — keep aggregates only
    for layer_data in all_metrics["layers"].values():
        for arm_data in layer_data.values():
            if isinstance(arm_data, dict):
                arm_data.pop("cs_per_sample", None)

    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info("Saved metrics → %s", metrics_path)

    wilcoxon_path = out_path / "wilcoxon_results.json"
    with open(wilcoxon_path, "w") as f:
        json.dump({"tests": wilcoxon_tests, "n_tests": len(wilcoxon_tests)}, f, indent=2)
    logger.info("Saved Wilcoxon results → %s", wilcoxon_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute cross-layer generalization metrics")
    p.add_argument("--data-dir", default="experiments/data")
    p.add_argument("--output-dir", default="experiments/results")
    p.add_argument("--rank-retrieval-max-n", type=int, default=500)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    compute_metrics(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        rank_retrieval_max_n=args.rank_retrieval_max_n,
    )


if __name__ == "__main__":
    main()

"""
Step 3: Run AR reconstruction on all descriptions from Step 2.

Loads descriptions_{raw,normalized}.json and runs the AR model on each
description to produce a reconstructed activation vector. Also computes
baselines:
  - B1: random Gaussian vectors (per-layer norm-matched)
  - B2: shuffled activations (real vectors, wrong text)
  - B5: mean activation vector per layer

Output files (in --output-dir):
  reconstructions_raw.npy        — float32, shape (N, 28, 3584)
  reconstructions_normalized.npy — float32, shape (N, 28, 3584)
  baseline_random.npy            — float32, shape (N, 28, 3584)
  baseline_shuffled.npy          — float32, shape (N, 28, 3584) — shuffled activations
  baseline_mean.npy              — float32, shape (28, 3584)    — per-layer mean vector

Usage:
    python experiments/run_ar_sweep.py \
        --config configs/qwen7b_layer20.yaml \
        --data-dir experiments/data \
        --ar-checkpoint checkpoints/ar_sft/final \
        --output-dir experiments/data \
        [--batch-size 32]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla_train.ar_sft import TruncatedARModel
from nla_train.injection import AR_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


@torch.no_grad()
def _run_ar_batch(
    ar_model: TruncatedARModel,
    tokenizer: AutoTokenizer,
    descriptions: list[str],
    device: torch.device,
    max_length: int = 256,
) -> np.ndarray:
    """Run AR on a batch of descriptions, return (B, d_model) float32."""
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": AR_PROMPT_TEMPLATE.format(explanation=d)}],
            tokenize=False,
            add_generation_prompt=False,
        )
        for d in descriptions
    ]
    enc = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    ).to(device)
    preds = ar_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    return preds.float().cpu().numpy()


def run_ar_sweep(
    config_path: str,
    data_dir: str,
    ar_checkpoint: str,
    output_dir: str,
    batch_size: int = 32,
    seed: int = 42,
) -> None:
    """
    Reconstruct activations from all descriptions and compute baselines.

    Args:
        config_path: Path to qwen7b_layer20.yaml.
        data_dir: Directory with activations.npy, meta.json, descriptions_*.json.
        ar_checkpoint: Path to AR SFT checkpoint.
        output_dir: Where to write reconstruction arrays.
        batch_size: AR inference batch size.
        seed: Random seed for baseline sampling.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(Path(data_dir) / "meta.json") as f:
        meta = json.load(f)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    N: int = meta["n_samples"]
    n_layers: int = meta["n_layers"]
    d_model: int = meta["d_model"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    np.random.seed(seed)

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading AR model from %s", ar_checkpoint)
    ar_base = AutoModelForCausalLM.from_pretrained(
        cfg["target_model"],
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    ar_truncated = TruncatedARModel(
        base_model=ar_base,
        target_layer=cfg["target_layer"],
        d_model=d_model,
    )
    del ar_base
    gc.collect()
    ar_truncated = PeftModel.from_pretrained(ar_truncated, ar_checkpoint, is_trainable=False)
    ar_truncated.to(device).eval()
    for p in ar_truncated.parameters():
        p.requires_grad_(False)

    activations = np.load(Path(data_dir) / "activations.npy").astype(np.float32)
    layer_norms = np.load(Path(data_dir) / "layer_norms.npy")

    # ---- Reconstruct from descriptions ----
    for arm in ["raw", "normalized"]:
        out_path = out_dir / f"reconstructions_{arm}.npy"
        if out_path.exists():
            logger.info("Skipping %s — already exists", out_path)
            continue

        desc_path = Path(data_dir) / f"descriptions_{arm}.json"
        if not desc_path.exists():
            logger.warning("Missing %s — skipping arm %s", desc_path, arm)
            continue

        with open(desc_path) as f:
            all_descs = json.load(f)

        # Reorganize by (text_id, layer) for ordered access
        desc_map: dict[tuple[int, int], str] = {}
        for item in all_descs:
            desc_map[(item["text_id"], item["layer"])] = item["description"]

        recons = np.zeros((N, n_layers, d_model), dtype=np.float32)

        for layer_idx in tqdm(range(n_layers), desc=f"AR reconstruct [{arm}]"):
            descs = [desc_map.get((i, layer_idx), "") for i in range(N)]
            for start in range(0, N, batch_size):
                batch = descs[start : start + batch_size]
                preds = _run_ar_batch(ar_truncated, tokenizer, batch, device)
                recons[start : start + len(batch), layer_idx] = preds

            gc.collect()

        np.save(out_path, recons)
        logger.info("Saved %s %s", out_path, recons.shape)

    # ---- Baseline B1: Random Gaussian (norm-matched per layer) ----
    b1_path = out_dir / "baseline_random.npy"
    if not b1_path.exists():
        logger.info("Computing baseline B1: random Gaussian")
        b1 = np.zeros((N, n_layers, d_model), dtype=np.float32)
        for layer_idx in range(n_layers):
            median_norm = float(meta["norm_stats"][str(layer_idx)]["median"])
            rand_vecs = np.random.randn(N, d_model).astype(np.float32)
            norms = np.linalg.norm(rand_vecs, axis=1, keepdims=True)
            b1[:, layer_idx] = rand_vecs / norms * median_norm
        np.save(b1_path, b1)
        logger.info("Saved baseline_random.npy")

    # ---- Baseline B2: Shuffled activations ----
    b2_path = out_dir / "baseline_shuffled.npy"
    if not b2_path.exists():
        logger.info("Computing baseline B2: shuffled activations")
        b2 = np.zeros_like(activations)
        for layer_idx in range(n_layers):
            perm = np.random.permutation(N)
            b2[:, layer_idx] = activations[perm, layer_idx]
        np.save(b2_path, b2)
        logger.info("Saved baseline_shuffled.npy")

    # ---- Baseline B5: Mean activation vector per layer ----
    b5_path = out_dir / "baseline_mean.npy"
    if not b5_path.exists():
        logger.info("Computing baseline B5: per-layer mean")
        b5 = activations.mean(axis=0)  # (n_layers, d_model)
        np.save(b5_path, b5)
        logger.info("Saved baseline_mean.npy")

    del ar_truncated
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("AR sweep complete")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AR sweep — reconstruct all descriptions")
    p.add_argument("--config", default="configs/qwen7b_layer20.yaml")
    p.add_argument("--data-dir", default="experiments/data")
    p.add_argument("--ar-checkpoint", default="checkpoints/ar_sft/final")
    p.add_argument("--output-dir", default="experiments/data")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    run_ar_sweep(
        config_path=args.config,
        data_dir=args.data_dir,
        ar_checkpoint=args.ar_checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

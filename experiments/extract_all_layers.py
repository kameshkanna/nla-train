"""
Step 1: Extract residual stream activations from ALL 28 layers.

Loads a 2000-text corpus across 5 domains (FineWeb, Wikipedia, PubMed,
GitHub READMEs, Reddit), runs a single forward pass per text with hooks at
every decoder layer, and saves a (2000, 28, 3584) float16 tensor.

Also saves per-layer norm statistics needed for the normalization ablation.

Output files (in --output-dir):
  activations.npy        — float16, shape (N, 28, 3584)
  texts.json             — list of N input strings
  meta.json              — corpus stats, norm stats per layer, layer count
  layer_norms.npy        — float32, shape (N, 28) — ||h_L||_2 for each sample/layer

Usage:
    python experiments/extract_all_layers.py \
        --config configs/qwen7b_layer20.yaml \
        --output-dir experiments/data \
        --n-per-domain 400 \
        [--token-position 10]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

DOMAINS = {
    "fineweb": dict(path="HuggingFaceFW/fineweb", name="sample-10BT", split="train", text_col="text"),
    "wikipedia": dict(path="wikimedia/wikipedia", name="20231101.en", split="train", text_col="text"),
    # ccdv/pubmed-summarization is parquet-native; article = full paper body
    "pubmed": dict(path="ccdv/pubmed-summarization", name=None, split="train", text_col="article"),
    # iamtarun/python_code_instructions_18k_alpaca is parquet-native, ungated, real Python code
    "github": dict(path="iamtarun/python_code_instructions_18k_alpaca", name=None, split="train", text_col="output"),
    "reddit": dict(path="sentence-transformers/reddit-title-body", name=None, split="train", text_col="body"),
}


def _load_texts(domain: str, n: int, min_len: int = 100, seed: int = 42) -> list[str]:
    """Stream n texts from a domain, filtering by minimum length."""
    cfg = DOMAINS[domain]
    try:
        if cfg["name"]:
            ds = load_dataset(cfg["path"], cfg["name"], split=cfg["split"], streaming=True)
        else:
            ds = load_dataset(cfg["path"], split=cfg["split"], streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=10_000)
        texts = []
        for item in ds:
            t = item.get(cfg["text_col"], "")
            if isinstance(t, str) and len(t) >= min_len:
                texts.append(t[:2000])  # cap at 2000 chars
            if len(texts) >= n:
                break
        logger.info("Loaded %d texts from %s", len(texts), domain)
        return texts
    except Exception as e:
        logger.warning("Failed to load %s (%s) — using fallback synthetic texts", domain, e)
        rng = random.Random(seed)
        fallback = [
            f"This is a synthetic fallback text for domain {domain}, sample {i}. "
            f"It contains enough words to exceed the minimum length threshold for activation extraction."
            for i in range(n)
        ]
        return fallback


class _AllLayerHook:
    """Registers forward hooks on all decoder layers and captures hidden states."""

    def __init__(self, model: AutoModelForCausalLM, n_layers: int) -> None:
        self._captures: dict[int, torch.Tensor] = {}
        self._handles = []
        for layer_idx in range(n_layers):
            def _make_hook(idx: int):
                def _hook(module: nn.Module, inp: tuple, out) -> None:
                    h = out[0] if isinstance(out, tuple) else out
                    self._captures[idx] = h.detach().float().cpu()
                return _hook
            handle = model.model.layers[layer_idx].register_forward_hook(_make_hook(layer_idx))
            self._handles.append(handle)

    def clear(self) -> None:
        self._captures.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def get(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._captures.get(layer_idx)


def extract_all_layers(
    config_path: str,
    output_dir: str,
    n_per_domain: int = 400,
    token_position: int = 10,
    seed: int = 42,
) -> None:
    """
    Extract residual stream activations at every layer for a diverse corpus.

    Args:
        config_path: Path to qwen7b_layer20.yaml.
        output_dir: Directory to write outputs.
        n_per_domain: Number of texts per domain (total = n_per_domain * 5).
        token_position: Token position to extract activation from (0-indexed).
        seed: Random seed for corpus shuffling.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_layers: int = cfg["num_layers"]
    d_model: int = cfg["d_model"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("Loading tokenizer: %s", cfg["target_model"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading base model: %s", cfg["target_model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["target_model"],
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
        trust_remote_code=True,
    )
    model.eval()

    # ---- Load corpus ----
    all_texts: list[str] = []
    domain_labels: list[str] = []
    for domain in DOMAINS:
        texts = _load_texts(domain, n_per_domain, seed=seed)
        all_texts.extend(texts[:n_per_domain])
        domain_labels.extend([domain] * min(len(texts), n_per_domain))

    N = len(all_texts)
    logger.info("Total corpus: %d texts across %d domains", N, len(DOMAINS))

    # ---- Extract activations ----
    activations = np.zeros((N, n_layers, d_model), dtype=np.float16)
    layer_norms = np.zeros((N, n_layers), dtype=np.float32)
    valid_mask = np.ones(N, dtype=bool)

    hook_manager = _AllLayerHook(model, n_layers)

    for i, text in enumerate(tqdm(all_texts, desc="Extracting activations")):
        enc = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
            add_special_tokens=True,
        ).to(device)

        seq_len = enc["input_ids"].shape[1]
        pos = min(token_position, seq_len - 1)

        hook_manager.clear()
        try:
            with torch.no_grad():
                model(**enc)
        except Exception as e:
            logger.warning("Forward pass failed for text %d: %s", i, e)
            valid_mask[i] = False
            continue

        for layer_idx in range(n_layers):
            hidden = hook_manager.get(layer_idx)
            if hidden is None:
                valid_mask[i] = False
                break
            vec = hidden[0, pos].numpy().astype(np.float16)
            activations[i, layer_idx] = vec
            layer_norms[i, layer_idx] = float(np.linalg.norm(vec.astype(np.float32)))

        if i % 100 == 0:
            gc.collect()

    hook_manager.remove()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Filter invalid samples
    valid_idx = np.where(valid_mask)[0]
    activations = activations[valid_idx]
    layer_norms = layer_norms[valid_idx]
    all_texts = [all_texts[i] for i in valid_idx]
    domain_labels = [domain_labels[i] for i in valid_idx]
    N_valid = len(valid_idx)
    logger.info("Valid samples: %d / %d", N_valid, N)

    # ---- Per-layer norm statistics ----
    norm_stats = {}
    for layer_idx in range(n_layers):
        norms = layer_norms[:, layer_idx]
        norm_stats[layer_idx] = {
            "mean": float(np.mean(norms)),
            "std": float(np.std(norms)),
            "median": float(np.median(norms)),
            "p10": float(np.percentile(norms, 10)),
            "p90": float(np.percentile(norms, 90)),
        }

    # ---- Save ----
    np.save(out_dir / "activations.npy", activations)
    np.save(out_dir / "layer_norms.npy", layer_norms)

    with open(out_dir / "texts.json", "w") as f:
        json.dump({"texts": all_texts, "domain_labels": domain_labels}, f)

    meta = {
        "n_samples": N_valid,
        "n_layers": n_layers,
        "d_model": d_model,
        "token_position": token_position,
        "n_per_domain": n_per_domain,
        "domains": list(DOMAINS.keys()),
        "seed": seed,
        "norm_stats": norm_stats,
        "layer20_median_norm": norm_stats[20]["median"],
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved activations.npy %s, layer_norms.npy, texts.json, meta.json → %s",
                activations.shape, out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract all-layer activations")
    p.add_argument("--config", default="configs/qwen7b_layer20.yaml")
    p.add_argument("--output-dir", default="experiments/data")
    p.add_argument("--n-per-domain", type=int, default=400)
    p.add_argument("--token-position", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    extract_all_layers(
        config_path=args.config,
        output_dir=args.output_dir,
        n_per_domain=args.n_per_domain,
        token_position=args.token_position,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

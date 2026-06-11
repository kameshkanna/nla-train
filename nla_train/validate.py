"""
Validation: compare our trained AV checkpoint against kitft's reference.

Protocol:
  1. Load 100 held-out activations from data/train/rl_train.parquet (last 100 rows).
  2. Run both AV checkpoints (ours + kitft) on the same activations → descriptions.
  3. Run our AR model on both sets of descriptions → reconstructed activations.
  4. Report:
     - Reconstruction MSE (normalized): ours vs kitft
     - Mean description length
     - 5 side-by-side qualitative examples

Usage:
    python -m nla_train.validate \
        --config configs/qwen7b_layer20.yaml \
        --our-av-checkpoint checkpoints/grpo/final_av \
        --ar-checkpoint checkpoints/ar_sft/final \
        --nla-meta data/labeled/nla_meta_av.yaml \
        --data-dir data/train \
        [--n-samples 100]
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla_train.ar_sft import TruncatedARModel
from nla_train.injection import AV_PROMPT_TEMPLATE, AR_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


def _load_activations(parquet_path: Path, n_samples: int) -> np.ndarray:
    """Load the last n_samples rows from a parquet file as a float32 array."""
    table = pq.read_table(parquet_path)
    table = table.slice(max(0, len(table) - n_samples), n_samples)
    n = len(table)
    d = len(table.column("activation_vector")[0].as_py())
    acts = table.column("activation_vector").combine_chunks()
    return acts.values.to_numpy(zero_copy_only=False).reshape(n, d).astype(np.float32)


@torch.no_grad()
def _run_av(
    av_model: PeftModel,
    tokenizer: AutoTokenizer,
    activations: np.ndarray,
    injection_char: str,
    injection_token_id: int,
    left_neighbor_id: int,
    right_neighbor_id: int,
    injection_scale: float,
    max_new_tokens: int,
    device: torch.device,
    batch_size: int = 8,
) -> list[str]:
    """Generate AV descriptions for a batch of activations."""
    from nla_train.injection import inject_at_marked_positions

    descriptions: list[str] = []
    n = len(activations)

    for start in tqdm(range(0, n, batch_size), desc="AV generate", leave=False):
        batch_acts = activations[start : start + batch_size]
        B = len(batch_acts)

        prompt_content = AV_PROMPT_TEMPLATE.format(injection_char=injection_char)
        messages = [{"role": "user", "content": prompt_content}]
        prompt_str: str = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = tokenizer(
            [prompt_str] * B,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        embed_layer = av_model.get_input_embeddings()
        inputs_embeds = embed_layer(enc["input_ids"]).clone()
        act_tensor = torch.tensor(batch_acts, dtype=inputs_embeds.dtype, device=device)
        inputs_embeds = inject_at_marked_positions(
            input_ids=enc["input_ids"],
            embeddings=inputs_embeds,
            activation_vectors=act_tensor,
            injection_token_id=injection_token_id,
            left_neighbor_id=left_neighbor_id,
            right_neighbor_id=right_neighbor_id,
            injection_scale=injection_scale,
        )

        out_ids = av_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        prompt_len = enc["input_ids"].shape[1]
        for i in range(B):
            generated = out_ids[i, prompt_len:]
            descriptions.append(tokenizer.decode(generated, skip_special_tokens=True))

    return descriptions


@torch.no_grad()
def _run_ar(
    ar_model: TruncatedARModel,
    tokenizer: AutoTokenizer,
    descriptions: list[str],
    device: torch.device,
    max_length: int = 256,
    batch_size: int = 16,
) -> np.ndarray:
    """Reconstruct activations from descriptions using AR model."""
    all_preds: list[np.ndarray] = []
    n = len(descriptions)

    for start in tqdm(range(0, n, batch_size), desc="AR reconstruct", leave=False):
        batch_desc = descriptions[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": AR_PROMPT_TEMPLATE.format(explanation=d)}],
                tokenize=False,
                add_generation_prompt=False,
            )
            for d in batch_desc
        ]
        enc = tokenizer(
            prompts,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
        preds = ar_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        all_preds.append(preds.float().cpu().numpy())

    return np.concatenate(all_preds, axis=0)


def _mse_normalized(preds: np.ndarray, golds: np.ndarray) -> float:
    """Mean MSE of L2-normalized vectors."""
    p = preds / (np.linalg.norm(preds, axis=1, keepdims=True) + 1e-8)
    g = golds / (np.linalg.norm(golds, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum((p - g) ** 2, axis=1)))


def validate(
    config_path: str,
    our_av_checkpoint: str,
    ar_checkpoint: str,
    nla_meta_path: str,
    data_dir: str,
    n_samples: int = 100,
    kitft_av_checkpoint: str | None = None,
) -> None:
    """
    Run validation comparing our AV checkpoint vs kitft reference.

    Args:
        config_path: Path to qwen7b_layer20.yaml.
        our_av_checkpoint: Path to our trained AV LoRA checkpoint.
        ar_checkpoint: Path to our trained AR SFT checkpoint.
        nla_meta_path: Path to nla_meta_av.yaml.
        data_dir: Directory containing rl_train.parquet (held-out data).
        n_samples: Number of activations to validate on.
        kitft_av_checkpoint: Optional kitft reference AV checkpoint for comparison.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(nla_meta_path) as f:
        nla_meta = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    injection_char = nla_meta["tokens"]["injection_char"]
    injection_token_id = nla_meta["tokens"]["injection_token_id"]
    left_neighbor_id = nla_meta["tokens"]["injection_left_neighbor_id"]
    right_neighbor_id = nla_meta["tokens"]["injection_right_neighbor_id"]
    injection_scale = nla_meta["extraction"]["injection_scale"]

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(cfg["verbalizer_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading held-out activations (%d samples)", n_samples)
    activations = _load_activations(Path(data_dir) / "rl_train.parquet", n_samples)
    logger.info("Activation shape: %s", activations.shape)

    # ---- Load AR model (frozen, for reconstruction scoring) ----
    logger.info("Loading AR model from %s", ar_checkpoint)
    ar_base = AutoModelForCausalLM.from_pretrained(
        cfg["target_model"], torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    ar_truncated = TruncatedARModel(
        base_model=ar_base, target_layer=cfg["target_layer"], d_model=cfg["d_model"]
    )
    del ar_base
    gc.collect()
    ar_truncated = PeftModel.from_pretrained(ar_truncated, ar_checkpoint, is_trainable=False)
    ar_truncated.to(device).eval()
    for p in ar_truncated.parameters():
        p.requires_grad_(False)

    results: dict[str, dict] = {}

    def _evaluate_av(label: str, av_ckpt: str) -> None:
        logger.info("=== Evaluating: %s ===", label)
        av_base = AutoModelForCausalLM.from_pretrained(
            cfg["verbalizer_model"],
            torch_dtype=torch.bfloat16,
            device_map={"": str(device)},
            trust_remote_code=True,
        )
        av_model = PeftModel.from_pretrained(av_base, av_ckpt, is_trainable=False)
        av_model.eval()

        descriptions = _run_av(
            av_model=av_model,
            tokenizer=tokenizer,
            activations=activations,
            injection_char=injection_char,
            injection_token_id=injection_token_id,
            left_neighbor_id=left_neighbor_id,
            right_neighbor_id=right_neighbor_id,
            injection_scale=injection_scale,
            max_new_tokens=cfg["grpo"]["max_completion_length"],
            device=device,
        )

        del av_model, av_base
        gc.collect()
        torch.cuda.empty_cache()

        preds = _run_ar(ar_truncated, tokenizer, descriptions, device)
        mse = _mse_normalized(preds, activations)
        mean_len = np.mean([len(d.split()) for d in descriptions])

        results[label] = {"descriptions": descriptions, "mse": mse, "mean_len": mean_len}
        logger.info("%s | MSE=%.6f | mean_desc_len=%.1f words", label, mse, mean_len)

    _evaluate_av("ours", our_av_checkpoint)

    if kitft_av_checkpoint:
        _evaluate_av("kitft", kitft_av_checkpoint)

    # ---- Print report ----
    print("\n" + "=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)
    for label, r in results.items():
        print(f"\n[{label}]  MSE={r['mse']:.6f}  mean_desc_len={r['mean_len']:.1f} words")

    if "ours" in results and "kitft" in results:
        delta = results["ours"]["mse"] - results["kitft"]["mse"]
        pct = delta / results["kitft"]["mse"] * 100
        print(f"\nDelta MSE (ours - kitft): {delta:+.6f}  ({pct:+.1f}%)")

    print("\n--- 5 qualitative examples ---")
    for i in range(min(5, n_samples)):
        print(f"\n[Sample {i+1}]")
        for label in results:
            desc = results[label]["descriptions"][i]
            print(f"  [{label}]: {desc[:300]}")

    # ---- Save results ----
    out_dir = Path(cfg["validation"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    import json
    out = {
        label: {"mse": r["mse"], "mean_len": r["mean_len"], "descriptions": r["descriptions"]}
        for label, r in results.items()
    }
    with open(out_dir / "validation_results.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info("Results saved to %s/validation_results.json", out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NLA Validation")
    p.add_argument("--config", default="configs/qwen7b_layer20.yaml")
    p.add_argument("--our-av-checkpoint", default="checkpoints/grpo/final_av")
    p.add_argument("--ar-checkpoint", default="checkpoints/ar_sft/final")
    p.add_argument("--nla-meta", default="data/labeled/nla_meta_av.yaml")
    p.add_argument("--data-dir", default="data/train")
    p.add_argument("--n-samples", type=int, default=100)
    p.add_argument("--kitft-av-checkpoint", default=None,
                   help="Optional: kitft reference checkpoint for side-by-side comparison")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    validate(
        config_path=args.config,
        our_av_checkpoint=args.our_av_checkpoint,
        ar_checkpoint=args.ar_checkpoint,
        nla_meta_path=args.nla_meta,
        data_dir=args.data_dir,
        n_samples=args.n_samples,
        kitft_av_checkpoint=args.kitft_av_checkpoint,
    )


if __name__ == "__main__":
    main()

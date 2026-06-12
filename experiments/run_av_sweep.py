"""
Step 2: Run AV inference on all (sample, layer) pairs.

Loads pre-extracted activations from Step 1 and runs the AV model on each,
injecting the layer-L activation at the CJK injection position. Supports two
normalization arms:
  - raw: feed activations as-is (no norm scaling)
  - normalized: scale each activation to match layer-20 median norm

Output files (in --output-dir):
  descriptions_raw.json        — list of N*28 dicts {text_id, layer, description}
  descriptions_normalized.json — same with norm-scaled activations

Usage:
    python experiments/run_av_sweep.py \
        --config configs/qwen7b_layer20.yaml \
        --data-dir experiments/data \
        --av-checkpoint checkpoints/grpo/final_av \
        --nla-meta data/labeled/nla_meta_av.yaml \
        --output-dir experiments/data \
        [--batch-size 8] \
        [--max-new-tokens 80]
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

from nla_train.injection import AV_PROMPT_TEMPLATE, inject_at_marked_positions

logger = logging.getLogger(__name__)


@torch.no_grad()
def _run_av_batch(
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
) -> list[str]:
    """Run AV on a batch of activation vectors, return descriptions."""
    B = len(activations)
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
    act_tensor = torch.tensor(activations, dtype=inputs_embeds.dtype, device=device)
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
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        pad_token_id=tokenizer.eos_token_id,
    )
    return [tokenizer.decode(out_ids[i], skip_special_tokens=True) for i in range(B)]


def run_av_sweep(
    config_path: str,
    data_dir: str,
    av_checkpoint: str,
    nla_meta_path: str,
    output_dir: str,
    batch_size: int = 8,
    max_new_tokens: int = 80,
) -> None:
    """
    Run AV inference for all (sample, layer) pairs in both normalization arms.

    Args:
        config_path: Path to qwen7b_layer20.yaml.
        data_dir: Directory with activations.npy and meta.json from Step 1.
        av_checkpoint: Path to AV LoRA checkpoint.
        nla_meta_path: Path to nla_meta_av.yaml.
        output_dir: Where to write description JSON files.
        batch_size: AV inference batch size.
        max_new_tokens: Max tokens per description.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(nla_meta_path) as f:
        nla_meta = yaml.safe_load(f)
    with open(Path(data_dir) / "meta.json") as f:
        meta = json.load(f)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    injection_char = nla_meta["tokens"]["injection_char"]
    injection_token_id = nla_meta["tokens"]["injection_token_id"]
    left_neighbor_id = nla_meta["tokens"]["injection_left_neighbor_id"]
    right_neighbor_id = nla_meta["tokens"]["injection_right_neighbor_id"]
    injection_scale = nla_meta["extraction"]["injection_scale"]
    layer20_median_norm: float = meta["layer20_median_norm"]

    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(cfg["verbalizer_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading AV model from %s", av_checkpoint)
    is_peft = (Path(av_checkpoint) / "adapter_config.json").exists()
    if is_peft:
        av_base = AutoModelForCausalLM.from_pretrained(
            cfg["verbalizer_model"],
            torch_dtype=torch.bfloat16,
            device_map={"": str(device)},
            trust_remote_code=True,
        )
        av_model = PeftModel.from_pretrained(av_base, av_checkpoint, is_trainable=False)
    else:
        av_model = AutoModelForCausalLM.from_pretrained(
            av_checkpoint,
            torch_dtype=torch.bfloat16,
            device_map={"": str(device)},
            trust_remote_code=True,
        )
    av_model.eval()

    logger.info("Loading activations")
    activations = np.load(Path(data_dir) / "activations.npy").astype(np.float32)
    layer_norms = np.load(Path(data_dir) / "layer_norms.npy")
    N, n_layers, d_model = activations.shape
    logger.info("Activations shape: %s", activations.shape)

    for arm in ["raw", "normalized"]:
        out_path = out_dir / f"descriptions_{arm}.json"
        if out_path.exists():
            logger.info("Skipping %s — already exists", out_path)
            continue

        results: list[dict] = []
        logger.info("=== ARM: %s ===", arm)

        # Flatten to (N*n_layers, d_model) for batched processing
        # Iterate layer by layer to keep memory manageable
        for layer_idx in tqdm(range(n_layers), desc=f"Layers [{arm}]"):
            layer_acts = activations[:, layer_idx, :].copy()  # (N, d_model)

            if arm == "normalized":
                # Scale each vector so its norm matches layer-20 median norm
                current_norms = layer_norms[:, layer_idx]
                scale_factors = np.where(
                    current_norms > 1e-8,
                    layer20_median_norm / current_norms,
                    1.0,
                )
                layer_acts = layer_acts * scale_factors[:, None]

            # Process in batches
            for start in range(0, N, batch_size):
                batch_acts = layer_acts[start : start + batch_size]
                descs = _run_av_batch(
                    av_model=av_model,
                    tokenizer=tokenizer,
                    activations=batch_acts,
                    injection_char=injection_char,
                    injection_token_id=injection_token_id,
                    left_neighbor_id=left_neighbor_id,
                    right_neighbor_id=right_neighbor_id,
                    injection_scale=injection_scale,
                    max_new_tokens=max_new_tokens,
                    device=device,
                )
                for b_idx, desc in enumerate(descs):
                    results.append({
                        "text_id": start + b_idx,
                        "layer": layer_idx,
                        "description": desc,
                        "arm": arm,
                    })

            gc.collect()

        with open(out_path, "w") as f:
            json.dump(results, f)
        logger.info("Saved %d descriptions → %s", len(results), out_path)

    del av_model
    if is_peft:
        del av_base
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("AV sweep complete")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AV sweep — all layers, both norm arms")
    p.add_argument("--config", default="configs/qwen7b_layer20.yaml")
    p.add_argument("--data-dir", default="experiments/data")
    p.add_argument("--av-checkpoint", default="checkpoints/grpo/final_av")
    p.add_argument("--nla-meta", default="data/labeled/nla_meta_av.yaml")
    p.add_argument("--output-dir", default="experiments/data")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=80)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    run_av_sweep(
        config_path=args.config,
        data_dir=args.data_dir,
        av_checkpoint=args.av_checkpoint,
        nla_meta_path=args.nla_meta,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()

"""
Token-level NLA evaluation — Neuronpedia-style.

For each token in an input text:
  1. Extract the residual stream activation at that position (layer 20).
  2. Run the AV model → explanation string.
  3. Run the AR model on the explanation → reconstructed activation.
  4. Compute RMSE between normalized reconstruction and normalized gold.

Output: per-token table of (token, explanation, rmse) printed to stdout,
with the best tokens (lowest RMSE) highlighted.

Usage:
    python -m nla_train.token_eval \
        --config configs/qwen7b_layer20.yaml \
        --av-checkpoint checkpoints/grpo/final_av \
        --ar-checkpoint checkpoints/ar_sft/final \
        --nla-meta data/labeled/nla_meta_av.yaml \
        --text "The quick brown fox jumps over the lazy dog." \
        [--top-k 5]
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla_train.ar_sft import TruncatedARModel
from nla_train.injection import AV_PROMPT_TEMPLATE, AR_PROMPT_TEMPLATE, inject_at_marked_positions

logger = logging.getLogger(__name__)


class _StopForward(Exception):
    pass


def _extract_activations(
    base_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    text: str,
    target_layer: int,
    device: torch.device,
    min_position: int = 5,
) -> tuple[list[str], np.ndarray]:
    """
    Extract residual stream activations at every token position in `text`.

    Args:
        base_model: Full base model (no LoRA), used for activation extraction only.
        tokenizer: Tokenizer.
        text: Input text string.
        target_layer: Layer index to extract activations from.
        device: CUDA or CPU device.
        min_position: Skip first N token positions (insufficient left context).

    Returns:
        (tokens, activations) where tokens is a list of decoded token strings
        and activations is a float32 array of shape (n_tokens, d_model).
    """
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=True).to(device)
    input_ids = enc["input_ids"]
    seq_len = input_ids.shape[1]

    captured: dict[str, torch.Tensor] = {}

    def _hook(module: nn.Module, inp: tuple, out: tuple) -> None:
        if isinstance(out, tuple):
            captured["hidden"] = out[0].detach().float().cpu()
        else:
            captured["hidden"] = out.detach().float().cpu()
        raise _StopForward()

    handle = base_model.model.layers[target_layer].register_forward_hook(_hook)
    try:
        with torch.no_grad():
            base_model(input_ids=input_ids)
    except _StopForward:
        pass
    finally:
        handle.remove()

    hidden = captured["hidden"][0]  # (seq_len, d_model)
    tokens = [tokenizer.decode([t]) for t in input_ids[0].tolist()]

    valid = list(range(min_position, seq_len))
    return (
        [tokens[i] for i in valid],
        hidden[valid].numpy(),
    )


@torch.no_grad()
def _run_av_single(
    av_model: PeftModel,
    tokenizer: AutoTokenizer,
    activation: np.ndarray,
    injection_char: str,
    injection_token_id: int,
    left_neighbor_id: int,
    right_neighbor_id: int,
    injection_scale: float,
    max_new_tokens: int,
    device: torch.device,
) -> str:
    """Run AV model on a single activation vector, return explanation string."""
    prompt_content = AV_PROMPT_TEMPLATE.format(injection_char=injection_char)
    messages = [{"role": "user", "content": prompt_content}]
    prompt_str: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    enc = tokenizer(prompt_str, return_tensors="pt", truncation=True, max_length=512).to(device)

    embed_layer = av_model.get_input_embeddings()
    inputs_embeds = embed_layer(enc["input_ids"]).clone()
    act_tensor = torch.tensor(activation[None], dtype=inputs_embeds.dtype, device=device)
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
    return tokenizer.decode(out_ids[0], skip_special_tokens=True)


@torch.no_grad()
def _run_ar_single(
    ar_model: TruncatedARModel,
    tokenizer: AutoTokenizer,
    description: str,
    device: torch.device,
    max_length: int = 256,
) -> np.ndarray:
    """Run AR model on a description, return reconstructed activation as float32 array."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": AR_PROMPT_TEMPLATE.format(explanation=description)}],
        tokenize=False,
        add_generation_prompt=False,
    )
    enc = tokenizer(
        prompt, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt"
    ).to(device)
    pred = ar_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    return pred[0].float().cpu().numpy()


def _rmse_normalized(pred: np.ndarray, gold: np.ndarray) -> float:
    """RMSE between L2-normalized vectors."""
    p = pred / (np.linalg.norm(pred) + 1e-8)
    g = gold / (np.linalg.norm(gold) + 1e-8)
    return float(np.sqrt(np.mean((p - g) ** 2)))


def token_eval(
    config_path: str,
    av_checkpoint: str,
    ar_checkpoint: str,
    nla_meta_path: str,
    text: str,
    top_k: int = 5,
    max_new_tokens: int = 100,
    min_position: int = 5,
    focus_token: str | None = None,
) -> None:
    """
    Run per-token NLA evaluation on input text.

    Args:
        config_path: Path to qwen7b_layer20.yaml.
        av_checkpoint: Path to AV checkpoint (PEFT LoRA or full merged).
        ar_checkpoint: Path to AR SFT checkpoint (PEFT LoRA).
        nla_meta_path: Path to nla_meta_av.yaml.
        text: Input text to evaluate on.
        top_k: Number of best (lowest RMSE) tokens to display in detail.
        max_new_tokens: Max tokens for AV generation.
        min_position: Skip first N token positions.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(nla_meta_path) as f:
        nla_meta = yaml.safe_load(f)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    injection_char = nla_meta["tokens"]["injection_char"]
    injection_token_id = nla_meta["tokens"]["injection_token_id"]
    left_neighbor_id = nla_meta["tokens"]["injection_left_neighbor_id"]
    right_neighbor_id = nla_meta["tokens"]["injection_right_neighbor_id"]
    injection_scale = nla_meta["extraction"]["injection_scale"]

    tokenizer = AutoTokenizer.from_pretrained(cfg["verbalizer_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load base model for activation extraction
    logger.info("Loading base model for activation extraction")
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["target_model"],
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
        trust_remote_code=True,
    )
    base_model.eval()

    logger.info("Extracting activations from layer %d", cfg["target_layer"])
    tokens, activations = _extract_activations(
        base_model=base_model,
        tokenizer=tokenizer,
        text=text,
        target_layer=cfg["target_layer"],
        device=device,
        min_position=min_position,
    )
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Extracted %d token activations", len(tokens))

    # If focus_token set, only evaluate that single position (last occurrence)
    if focus_token is not None:
        matches = [i for i, t in enumerate(tokens) if focus_token in t]
        if not matches:
            raise ValueError(f"focus_token {focus_token!r} not found in tokens: {tokens}")
        idx = matches[-1]
        tokens = [tokens[idx]]
        activations = activations[idx : idx + 1]
        logger.info("Focus mode: evaluating only token %r at position %d", tokens[0], idx)

    # Load AV model
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

    # Load AR model
    logger.info("Loading AR model from %s", ar_checkpoint)
    ar_base = AutoModelForCausalLM.from_pretrained(
        cfg["target_model"],
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
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

    # Per-token evaluation
    results: list[dict] = []
    for i, (token, activation) in enumerate(zip(tokens, activations)):
        logger.info("Token %d/%d: %r", i + 1, len(tokens), token)
        explanation = _run_av_single(
            av_model, tokenizer, activation,
            injection_char, injection_token_id,
            left_neighbor_id, right_neighbor_id, injection_scale,
            max_new_tokens, device,
        )
        reconstruction = _run_ar_single(ar_truncated, tokenizer, explanation, device)
        rmse = _rmse_normalized(reconstruction, activation)
        results.append({"token": token, "explanation": explanation, "rmse": rmse})

    # Sort by RMSE for summary
    ranked = sorted(results, key=lambda r: r["rmse"])
    top_k = min(top_k, len(ranked))

    print("\n" + "=" * 70)
    print(f"TOKEN-LEVEL NLA EVALUATION  |  {len(tokens)} tokens  |  layer {cfg['target_layer']}")
    print("=" * 70)
    print(f"\nInput text: {text[:120]}")

    print(f"\n{'TOKEN':<20} {'RMSE':>8}")
    print("-" * 30)
    for r in results:
        marker = " ◀" if r["rmse"] <= ranked[top_k - 1]["rmse"] else ""
        print(f"  {repr(r['token']):<18} {r['rmse']:>8.4f}{marker}")

    print(f"\n--- Top {top_k} best-reconstructed tokens ---")
    for r in ranked[:top_k]:
        print(f"\nToken: {repr(r['token'])}  RMSE={r['rmse']:.4f}")
        print(f"  {r['explanation'][:400]}")

    # Save
    import json
    out_dir = Path(cfg["validation"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "token_eval.json", "w") as f:
        json.dump({"text": text, "results": results}, f, indent=2)
    logger.info("Saved to %s/token_eval.json", out_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Token-level NLA evaluation")
    p.add_argument("--config", default="configs/qwen7b_layer20.yaml")
    p.add_argument("--av-checkpoint", default="checkpoints/grpo/final_av")
    p.add_argument("--ar-checkpoint", default="checkpoints/ar_sft/final")
    p.add_argument("--nla-meta", default="data/labeled/nla_meta_av.yaml")
    p.add_argument("--text", required=True, help="Input text to evaluate on")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--min-position", type=int, default=5)
    p.add_argument("--focus-token", default=None,
                   help="Only evaluate the last occurrence of this token string (e.g. ',')")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    token_eval(
        config_path=args.config,
        av_checkpoint=args.av_checkpoint,
        ar_checkpoint=args.ar_checkpoint,
        nla_meta_path=args.nla_meta,
        text=args.text,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        min_position=args.min_position,
        focus_token=args.focus_token,
    )


if __name__ == "__main__":
    main()

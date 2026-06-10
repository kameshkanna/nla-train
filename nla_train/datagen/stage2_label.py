"""
Stage 2: Generate AV labels via direct batched HuggingFace inference.

Loads kitft/nla-qwen2.5-7b-L20-av directly and runs batched generation with
vectorized embedding injection — no SGLang HTTP server required.

Throughput rationale:
  - SGLang /generate with input_embeds: ~3-4 req/s ceiling. Unique per-request
    embeddings prevent SGLang from batching prefill; serializing 125×3584 floats
    to JSON per HTTP call dominates latency.
  - Direct HF batched inference: all prompts share the same template, so we clone
    base embeddings once and inject activations vectorized across the batch.
    64 sequences per GPU pass on H100 → ~15-20 seq/s → 250k rows ≈ 4h.

Usage:
    # No SGLang server needed. Kill it first to free GPU memory.
    python -m nla_train.datagen.stage2_label \\
        --config configs/qwen7b_layer20.yaml \\
        --input-dir data/split \\
        --output-dir data/labeled \\
        [--resume]
"""

from __future__ import annotations

import argparse
import gc
import logging
import re
from pathlib import Path

import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from nla_train.injection import (
    AV_PROMPT_TEMPLATE,
    select_injection_token,
    compute_injection_neighbors,
    compute_injection_scale,
    write_nla_meta,
)

logger = logging.getLogger(__name__)

LABELED_SCHEMA = pa.schema([
    pa.field("doc_id", pa.string()),
    pa.field("token_position", pa.int32()),
    pa.field("text_snippet", pa.string()),
    pa.field("activation_vector", pa.list_(pa.float32())),
    pa.field("layer", pa.int32()),
    pa.field("explanation", pa.string()),
])


def _find_injection_pos(
    ids: list[int],
    injection_token_id: int,
    left_neighbor_id: int,
    right_neighbor_id: int,
) -> int:
    """Return the position of the injection token in `ids`, or raise."""
    for i in range(1, len(ids) - 1):
        if (
            ids[i] == injection_token_id
            and ids[i - 1] == left_neighbor_id
            and ids[i + 1] == right_neighbor_id
        ):
            return i
    for i, tok in enumerate(ids):
        if tok == injection_token_id:
            return i
    raise RuntimeError(f"Injection token {injection_token_id} not found in prompt.")


def _build_prompt_ids(
    tokenizer: AutoTokenizer,
    injection_char: str,
) -> list[int]:
    """Tokenize the AV prompt template once; reused for every row."""
    prompt_content = AV_PROMPT_TEMPLATE.format(injection_char=injection_char)
    formatted: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return tokenizer.encode(formatted, add_special_tokens=False)


@torch.inference_mode()
def _generate_batch(
    activations: np.ndarray,
    base_ids: list[int],
    inject_pos: int,
    embed_weight: torch.Tensor,
    injection_scale: float,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    temperature: float,
    device: torch.device,
) -> list[str]:
    """
    Generate explanations for a batch of activation vectors.

    All prompts in the batch share the same template, so we build base embeddings
    once, clone, then inject each activation vectorized — a single GPU matmul
    instead of B sequential operations.

    Args:
        activations: (B, d_model) float32 array.
        base_ids: Prompt token IDs (same for all rows).
        inject_pos: Index of the injection token within base_ids.
        embed_weight: (vocab_size, d_model) CPU float tensor from the verbalizer.
        injection_scale: Scalar to normalize injected activation norms.
        model: Loaded kitft AV model.
        tokenizer: AV model tokenizer.
        max_new_tokens: Generation budget per sequence.
        temperature: Sampling temperature.
        device: GPU device the model lives on.

    Returns:
        List of B explanation strings (empty string on generation failure).
    """
    B = len(activations)

    ids_tensor = torch.tensor(base_ids, dtype=torch.long)
    base_embeds = embed_weight[ids_tensor].float()               # (seq_len, d_model)
    batch_embeds = base_embeds.unsqueeze(0).expand(B, -1, -1).clone()  # (B, seq_len, d_model)

    acts = torch.from_numpy(activations.astype(np.float32))     # (B, d_model)
    norms = acts.norm(dim=1, keepdim=True).clamp(min=1e-8)
    batch_embeds[:, inject_pos, :] = (acts / norms) * injection_scale

    seq_len = batch_embeds.shape[1]
    attention_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)

    output_ids = model.generate(
        inputs_embeds=batch_embeds.to(device=device, dtype=torch.bfloat16),
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else None,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.eos_token_id,
    )

    results: list[str] = []
    for ids in output_ids:
        text = tokenizer.decode(ids, skip_special_tokens=True)
        m = re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL)
        results.append(m.group(1).strip() if m else text.strip())
    return results


def _load_checkpoint(path: Path, expected_len: int) -> list[str] | None:
    """Load checkpoint if it exists and matches expected row count."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    if len(data) != expected_len:
        logger.warning("Checkpoint length %d ≠ %d — starting fresh", len(data), expected_len)
        return None
    labeled = sum(bool(e) for e in data)
    logger.info("Resumed checkpoint: %d/%d already labeled", labeled, expected_len)
    return data


def _save_checkpoint(explanations: list[str], path: Path) -> None:
    """Atomically write explanations list to checkpoint file."""
    tmp = path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(orjson.dumps(explanations))
    tmp.rename(path)


def generate_labels(
    input_parquet: str | Path,
    output_parquet: str | Path,
    av_model_name: str,
    verbalizer_model_name: str,
    d_model: int,
    layer_idx: int,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    injection_cache_path: str | Path,
    nla_meta_output_path: str | Path,
    resume: bool = False,
    seed: int = 42,
) -> int:
    """
    Run the AV labeling pipeline on the av_sft split using batched HF inference.

    Loads the kitft AV checkpoint directly — no SGLang server required. Processes
    rows in batches for checkpoint granularity and GPU efficiency.

    Args:
        input_parquet: Path to av_sft.parquet from Stage 1.
        output_parquet: Output path for the labeled parquet.
        av_model_name: kitft AV checkpoint path or HF model ID.
        verbalizer_model_name: Target model for tokenizer + embedding table.
        d_model: Target model hidden dimension.
        layer_idx: Extraction layer.
        batch_size: GPU batch size for generation (default 64 on H100).
        max_new_tokens: Max tokens per verbalization.
        temperature: Sampling temperature.
        injection_cache_path: Cache path for injection token selection.
        nla_meta_output_path: Where to write the nla_meta.yaml sidecar.
        resume: Skip already-labeled rows via checkpoint file.
        seed: Random seed.

    Returns:
        Number of labeled rows written.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("Loading tokenizer from verbalizer model: %s", verbalizer_model_name)
    tokenizer = AutoTokenizer.from_pretrained(verbalizer_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading embedding table (CPU, then freed)")
    base_model = AutoModelForCausalLM.from_pretrained(
        verbalizer_model_name, dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    embed_weight = base_model.get_input_embeddings().weight.detach().float().cpu()
    del base_model
    gc.collect()

    logger.info("Loading kitft AV model: %s", av_model_name)
    av_model = AutoModelForCausalLM.from_pretrained(
        av_model_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    av_model.eval()
    for p in av_model.parameters():
        p.requires_grad_(False)
    device = next(av_model.parameters()).device
    logger.info("AV model on device: %s", device)

    injection_char, injection_token_id = select_injection_token(
        tokenizer, cache_path=injection_cache_path
    )
    left_neighbor_id, right_neighbor_id = compute_injection_neighbors(tokenizer, injection_char)
    base_ids = _build_prompt_ids(tokenizer, injection_char)
    inject_pos = _find_injection_pos(base_ids, injection_token_id, left_neighbor_id, right_neighbor_id)
    logger.info(
        "Prompt: %d tokens, injection at position %d (token id %d)",
        len(base_ids), inject_pos, injection_token_id,
    )

    logger.info("Reading input parquet: %s", input_parquet)
    table = pq.read_table(input_parquet)
    n_rows = len(table)
    logger.info("Rows to label: %d", n_rows)

    act_col = table.column("activation_vector").combine_chunks()
    activations = act_col.values.to_numpy(zero_copy_only=False).reshape(n_rows, d_model)
    injection_scale = compute_injection_scale(activations)
    logger.info("Injection scale: %.4f", injection_scale)

    write_nla_meta(
        output_path=nla_meta_output_path,
        role="av",
        d_model=d_model,
        layer_idx=layer_idx,
        injection_char=injection_char,
        injection_token_id=injection_token_id,
        left_neighbor_id=left_neighbor_id,
        right_neighbor_id=right_neighbor_id,
        injection_scale=injection_scale,
    )

    doc_ids = table.column("doc_id").to_pylist()
    positions = table.column("token_position").to_pylist()
    snippets = table.column("text_snippet").to_pylist()
    layers = table.column("layer").to_pylist()

    checkpoint_path = Path(output_parquet).with_suffix(".checkpoint.json")
    if resume:
        existing = _load_checkpoint(checkpoint_path, n_rows)
        explanations: list[str] = existing if existing is not None else [""] * n_rows
    else:
        explanations = [""] * n_rows

    n_already = sum(bool(e) for e in explanations)
    logger.info("Starting: %d to label, %d already done", n_rows - n_already, n_already)

    pbar = tqdm(total=n_rows, initial=n_already, desc="Labeling", unit="seq", dynamic_ncols=True)
    checkpoint_every = max(1, 4096 // batch_size)  # checkpoint ~every 4096 rows

    batch_count = 0
    for batch_start in range(0, n_rows, batch_size):
        batch_end = min(batch_start + batch_size, n_rows)
        pending = [i for i in range(batch_start, batch_end) if not explanations[i]]

        if not pending:
            continue

        batch_acts = activations[pending]
        try:
            texts = _generate_batch(
                activations=batch_acts,
                base_ids=base_ids,
                inject_pos=inject_pos,
                embed_weight=embed_weight,
                injection_scale=injection_scale,
                model=av_model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                device=device,
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            logger.warning("Batch %d failed: %s — skipping", batch_start, e)
            torch.cuda.empty_cache()
            gc.collect()
            continue

        for idx, expl in zip(pending, texts):
            explanations[idx] = expl

        pbar.update(len(pending))
        batch_count += 1

        if batch_count % checkpoint_every == 0:
            _save_checkpoint(explanations, checkpoint_path)
            n_done = sum(bool(e) for e in explanations)
            logger.info("Checkpoint: %d/%d (%.1f%%)", n_done, n_rows, 100 * n_done / n_rows)

    pbar.close()

    _save_checkpoint(explanations, checkpoint_path)
    n_valid = sum(bool(e) for e in explanations)
    logger.info("Labeled: %d/%d (%.1f%%)", n_valid, n_rows, 100 * n_valid / n_rows)

    valid_indices = [i for i, e in enumerate(explanations) if e]
    out_table = pa.table(
        {
            "doc_id": pa.array([doc_ids[i] for i in valid_indices]),
            "token_position": pa.array([positions[i] for i in valid_indices], pa.int32()),
            "text_snippet": pa.array([snippets[i] for i in valid_indices]),
            "activation_vector": pa.array(
                [activations[i].tolist() for i in valid_indices],
                pa.list_(pa.float32()),
            ),
            "layer": pa.array([layers[i] for i in valid_indices], pa.int32()),
            "explanation": pa.array([explanations[i] for i in valid_indices]),
        },
        schema=LABELED_SCHEMA,
    )

    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, output_parquet, compression="zstd")
    logger.info("Wrote %d labeled rows → %s", n_valid, output_parquet)

    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return n_valid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: AV labeling via batched HF inference")
    p.add_argument("--config", default="configs/qwen7b_layer20.yaml")
    p.add_argument("--input-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--batch-size", type=int, default=None, help="Override config batch_size")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dg = cfg["datagen"]
    lab = cfg["labeling"]
    inj = cfg["injection"]

    input_dir = Path(args.input_dir or str(Path(dg["output_dir"]).parent / "split"))
    output_dir = Path(args.output_dir or lab["output_dir"])

    batch_size = args.batch_size or lab.get("hf_batch_size", 64)

    generate_labels(
        input_parquet=input_dir / "av_sft.parquet",
        output_parquet=output_dir / "av_sft_labeled.parquet",
        av_model_name=cfg.get("av_labeler_model", cfg["reference_av_checkpoint"]),
        verbalizer_model_name=cfg["verbalizer_model"],
        d_model=cfg["d_model"],
        layer_idx=cfg["target_layer"],
        batch_size=batch_size,
        max_new_tokens=lab["max_new_tokens"],
        temperature=lab["temperature"],
        injection_cache_path=inj["cache_file"],
        nla_meta_output_path=output_dir / "nla_meta_av.yaml",
        resume=args.resume,
        seed=cfg["seed"],
    )


if __name__ == "__main__":
    main()

"""
Stage 2: Generate AV labels via the kitft SGLang checkpoint.

Calls the running SGLang server (kitft/nla-qwen2.5-7b-L20-av) on every
activation in the av_sft split to produce gold descriptions. These descriptions
become the training targets for our AV SFT stage and, via cross-join with the
ar_sft split, the inputs for our AR SFT stage.

Throughput design:
  - Single event loop + persistent httpx.AsyncClient for the entire run.
    (Old design recreated both every 1024 rows — ~250 TCP handshake storms.)
  - asyncio.to_thread() for _build_embeds so CPU-bound embedding construction
    overlaps with in-flight SGLang HTTP requests.
  - asyncio.Semaphore(concurrency) gates only the HTTP call, not embed building,
    so SGLang is never idle waiting for the next payload.
  - Checkpoint written every batch to a .json sidecar so --resume skips
    already-labeled rows after a crash.

Usage:
    # 1. Start SGLang server first:
    #    python -m sglang.launch_server --model-path kitft/nla-qwen2.5-7b-L20-av \\
    #        --port 30000 --mem-fraction-static 0.45
    #
    # 2. Then run:
    python -m nla_train.datagen.stage2_label \\
        --config configs/qwen7b_layer20.yaml \\
        --input-dir data/split \\
        --output-dir data/labeled \\
        [--resume]
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import logging
import re
import time
from pathlib import Path
from typing import Optional

import httpx
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


def _build_embeds(
    activation: np.ndarray,
    tokenizer: AutoTokenizer,
    embed_weight: torch.Tensor,
    injection_token_id: int,
    left_neighbor_id: int,
    right_neighbor_id: int,
    injection_scale: float,
    injection_char: str,
) -> np.ndarray:
    """
    Build the input_embeds array for a single activation vector.

    Returns float32 numpy array of shape (seq_len, d_model).
    CPU-bound — call via asyncio.to_thread to overlap with HTTP I/O.
    """
    prompt_content = AV_PROMPT_TEMPLATE.format(injection_char=injection_char)
    formatted: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids: list[int] = tokenizer.encode(formatted, add_special_tokens=False)

    inject_pos: Optional[int] = None
    for i in range(1, len(ids) - 1):
        if (
            ids[i] == injection_token_id
            and ids[i - 1] == left_neighbor_id
            and ids[i + 1] == right_neighbor_id
        ):
            inject_pos = i
            break

    if inject_pos is None:
        for i, tok in enumerate(ids):
            if tok == injection_token_id:
                inject_pos = i
                break

    if inject_pos is None:
        raise RuntimeError(f"Injection token {injection_token_id} not found in tokenized prompt.")

    ids_tensor = torch.tensor(ids, dtype=torch.long)
    embeds = embed_weight[ids_tensor].clone().float()

    act = torch.from_numpy(activation.astype(np.float32))
    norm = act.norm()
    if norm > 1e-8:
        act = act / norm
    act = act * injection_scale
    embeds[inject_pos] = act

    return embeds.numpy().astype(np.float32)


async def _process_row(
    idx: int,
    activation: np.ndarray,
    tokenizer: AutoTokenizer,
    embed_weight: torch.Tensor,
    injection_token_id: int,
    left_neighbor_id: int,
    right_neighbor_id: int,
    injection_scale: float,
    injection_char: str,
    sglang_url: str,
    max_new_tokens: int,
    temperature: float,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> tuple[int, str]:
    """
    Build embeds (in thread pool) then POST to SGLang (under semaphore).

    Embed building runs in asyncio.to_thread so it overlaps with other tasks'
    in-flight HTTP requests — SGLang is never starved waiting for the next payload.

    Returns (row_idx, explanation_text).
    """
    try:
        embeds: np.ndarray = await asyncio.to_thread(
            _build_embeds,
            activation,
            tokenizer,
            embed_weight,
            injection_token_id,
            left_neighbor_id,
            right_neighbor_id,
            injection_scale,
            injection_char,
        )
    except RuntimeError as e:
        logger.warning("embed build failed for row %d: %s", idx, e)
        return idx, ""

    payload = {
        "input_embeds": embeds.tolist(),
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        },
    }
    encoded = orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY)

    async with semaphore:
        try:
            resp = await client.post(
                sglang_url.rstrip("/") + "/generate",
                content=encoded,
                headers={"Content-Type": "application/json"},
                timeout=180.0,
            )
            resp.raise_for_status()
            text: str = resp.json()["text"]
            m = re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL)
            explanation = m.group(1).strip() if m else text.strip()
        except (httpx.HTTPError, KeyError, RuntimeError) as e:
            logger.warning("SGLang call failed for row %d: %s", idx, e)
            explanation = ""

    return idx, explanation


def _load_checkpoint(path: Path) -> list[str] | None:
    """Load existing explanations from checkpoint file. Returns None if not found."""
    if not path.exists():
        return None
    with open(path, "rb") as f:
        data = orjson.loads(f.read())
    logger.info("Loaded checkpoint with %d rows (%d labeled)", len(data), sum(bool(e) for e in data))
    return data


def _save_checkpoint(explanations: list[str], path: Path) -> None:
    """Write current explanations list to checkpoint file (atomic via tmp rename)."""
    tmp = path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        f.write(orjson.dumps(explanations))
    tmp.rename(path)


async def _label_all_async(
    n_rows: int,
    activations: np.ndarray,
    tokenizer: AutoTokenizer,
    embed_weight: torch.Tensor,
    injection_token_id: int,
    left_neighbor_id: int,
    right_neighbor_id: int,
    injection_scale: float,
    injection_char: str,
    sglang_url: str,
    max_new_tokens: int,
    temperature: float,
    concurrency: int,
    batch_size: int,
    explanations: list[str],
    checkpoint_path: Path,
    pbar: tqdm,
) -> None:
    """
    Single-event-loop, single-client labeling loop.

    Processes rows in batches of batch_size for checkpoint granularity.
    Within each batch, fires all pending tasks concurrently (semaphore-gated).
    Embed building overlaps with in-flight HTTP calls via asyncio.to_thread.
    """
    semaphore = asyncio.Semaphore(concurrency)
    limits = httpx.Limits(
        max_connections=concurrency + 8,
        max_keepalive_connections=concurrency,
    )

    async with httpx.AsyncClient(limits=limits, timeout=httpx.Timeout(180.0)) as client:
        for batch_start in range(0, n_rows, batch_size):
            batch_end = min(batch_start + batch_size, n_rows)
            pending = [i for i in range(batch_start, batch_end) if not explanations[i]]
            already_done = (batch_end - batch_start) - len(pending)
            pbar.update(already_done)

            if not pending:
                continue

            tasks = [
                asyncio.create_task(
                    _process_row(
                        idx=i,
                        activation=activations[i],
                        tokenizer=tokenizer,
                        embed_weight=embed_weight,
                        injection_token_id=injection_token_id,
                        left_neighbor_id=left_neighbor_id,
                        right_neighbor_id=right_neighbor_id,
                        injection_scale=injection_scale,
                        injection_char=injection_char,
                        sglang_url=sglang_url,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        client=client,
                        semaphore=semaphore,
                    )
                )
                for i in pending
            ]

            for coro in asyncio.as_completed(tasks):
                try:
                    idx, expl = await coro
                    explanations[idx] = expl
                except Exception as e:
                    logger.warning("Unexpected error in task: %s", e)
                pbar.update(1)

            _save_checkpoint(explanations, checkpoint_path)
            logger.info(
                "Checkpoint: %d/%d labeled (%.1f%%)",
                sum(bool(e) for e in explanations), n_rows,
                100 * sum(bool(e) for e in explanations) / n_rows,
            )


def generate_labels(
    input_parquet: str | Path,
    output_parquet: str | Path,
    model_name: str,
    d_model: int,
    layer_idx: int,
    sglang_url: str,
    batch_size: int,
    concurrency: int,
    max_new_tokens: int,
    temperature: float,
    injection_cache_path: str | Path,
    nla_meta_output_path: str | Path,
    resume: bool = False,
    seed: int = 42,
) -> int:
    """
    Run the AV labeling pipeline on the av_sft split.

    Single asyncio.run() wraps the entire dataset so the event loop and HTTP
    connection pool live for the full run — no per-batch teardown/setup overhead.

    Args:
        input_parquet: Path to av_sft.parquet from Stage 1.
        output_parquet: Output path for the labeled parquet.
        model_name: Target model name (for tokenizer and embedding table).
        d_model: Target model hidden dimension.
        layer_idx: Extraction layer.
        sglang_url: Running SGLang AV server URL.
        batch_size: Rows per checkpoint batch (not per HTTP burst).
        concurrency: Max simultaneous SGLang requests.
        max_new_tokens: Max tokens per verbalization.
        temperature: Sampling temperature.
        injection_cache_path: Cache path for injection token selection.
        nla_meta_output_path: Where to write the nla_meta.yaml sidecar.
        resume: If True, load existing checkpoint and skip already-labeled rows.
        seed: Random seed.

    Returns:
        Number of labeled rows written.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    logger.info("Loading embedding table (CPU only — then freed)")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    embed_weight = model.get_input_embeddings().weight.detach().float().cpu()
    del model
    gc.collect()

    injection_char, injection_token_id = select_injection_token(
        tokenizer, cache_path=injection_cache_path
    )
    left_neighbor_id, right_neighbor_id = compute_injection_neighbors(tokenizer, injection_char)

    logger.info("Reading input parquet: %s", input_parquet)
    table = pq.read_table(input_parquet)
    n_rows = len(table)
    logger.info("Rows to label: %d", n_rows)

    activations = np.array(table.column("activation_vector").to_pylist(), dtype=np.float32)
    injection_scale = compute_injection_scale(activations)

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
    explanations: list[str] = []
    if resume:
        existing = _load_checkpoint(checkpoint_path)
        if existing and len(existing) == n_rows:
            explanations = existing
        else:
            logger.warning("Checkpoint mismatch or missing — starting fresh")
            explanations = [""] * n_rows
    else:
        explanations = [""] * n_rows

    n_already = sum(bool(e) for e in explanations)
    logger.info("Starting labeling: %d to label, %d already done", n_rows - n_already, n_already)

    pbar = tqdm(total=n_rows, initial=0, desc="Labeling", unit="row", dynamic_ncols=True)
    t0 = time.monotonic()

    asyncio.run(
        _label_all_async(
            n_rows=n_rows,
            activations=activations,
            tokenizer=tokenizer,
            embed_weight=embed_weight,
            injection_token_id=injection_token_id,
            left_neighbor_id=left_neighbor_id,
            right_neighbor_id=right_neighbor_id,
            injection_scale=injection_scale,
            injection_char=injection_char,
            sglang_url=sglang_url,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            concurrency=concurrency,
            batch_size=batch_size,
            explanations=explanations,
            checkpoint_path=checkpoint_path,
            pbar=pbar,
        )
    )

    pbar.close()
    elapsed = time.monotonic() - t0
    n_valid = sum(bool(e) for e in explanations)
    logger.info(
        "Labeling complete: %d/%d valid (%.1f%%) in %.1fmin",
        n_valid, n_rows, 100 * n_valid / n_rows, elapsed / 60,
    )

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
        logger.info("Removed checkpoint file")

    return n_valid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: Generate AV labels via SGLang (async)")
    p.add_argument("--config", default="configs/qwen7b_layer20.yaml")
    p.add_argument("--input-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint, skip labeled rows")
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

    generate_labels(
        input_parquet=input_dir / "av_sft.parquet",
        output_parquet=output_dir / "av_sft_labeled.parquet",
        model_name=cfg["target_model"],
        d_model=cfg["d_model"],
        layer_idx=cfg["target_layer"],
        sglang_url=lab["sglang_url"],
        batch_size=lab.get("batch_size", 1024),
        concurrency=lab.get("concurrency", 128),
        max_new_tokens=lab["max_new_tokens"],
        temperature=lab["temperature"],
        injection_cache_path=inj["cache_file"],
        nla_meta_output_path=output_dir / "nla_meta_av.yaml",
        resume=args.resume,
        seed=cfg["seed"],
    )


if __name__ == "__main__":
    main()

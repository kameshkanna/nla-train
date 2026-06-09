"""
Stage 0: Activation Extraction via batched HuggingFace forward pass.

Loads the target model (Qwen2.5-7B-Instruct) and runs the FineWeb corpus through
in large batches, capturing the residual stream at the target layer via a forward hook.
We then select specific token positions per document and write activation vectors to
Parquet files.

We use a plain HF model (not vLLM) because:
  - We need activations at arbitrary token positions within a sequence, not just the
    last token of the full sequence.
  - vLLM's batching is designed for generation; it exposes per-request last-token
    logits, not mid-sequence hidden states across a batch.
  - With left-padding + batch size 512 on an H100, throughput is equivalent to vLLM
    for this pure-forward-pass workload.

Design:
  - Left-pad all sequences in a batch to a common length.
  - Single forward pass captures the full (batch, seq, d_model) hidden state.
  - Extract activations at the N requested token positions per document from that
    one cached hidden state — no per-position re-forward.
  - StopForward exception aborts execution after the target layer to avoid wasting
    time on later layers + the LM head.
  - Write to Parquet in chunk_size-doc chunks for resumability (--resume flag).

Throughput target: 512 docs × 10 positions = 5120 activations per batch.
Expected: ~50k activations/minute on H100 → 1M vectors in ~20 minutes.

Usage:
    python -m nla_train.datagen.stage0_extract \
        --config configs/qwen7b_layer20.yaml \
        --output-dir data/raw \
        [--n-docs 100000] \
        [--resume]
"""

from __future__ import annotations

import argparse
import gc
import logging
import uuid
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

ACTIVATION_SCHEMA = pa.schema([
    pa.field("doc_id", pa.string()),
    pa.field("token_position", pa.int32()),
    pa.field("text_snippet", pa.string()),
    pa.field("activation_vector", pa.list_(pa.float32())),
    pa.field("layer", pa.int32()),
])

_ACT_STORE: dict[str, torch.Tensor] = {}


class _StopForward(Exception):
    """Raised by the layer hook to abort the forward pass after target layer."""


def _make_layer_hook(key: str) -> callable:
    """
    Return a forward hook that captures hidden states and stops execution.

    Captures (batch, seq, d_model) and raises _StopForward so PyTorch unwinds
    the forward pass immediately — no subsequent layers run.
    """
    def _hook(module: nn.Module, inp: tuple, out: object) -> None:
        hidden = out[0] if isinstance(out, tuple) else out
        _ACT_STORE[key] = hidden.detach().float().cpu()
        raise _StopForward
    return _hook


def _stream_fineweb(
    corpus_name: str,
    n_docs: int,
    text_column: str = "text",
) -> Generator[tuple[str, str], None, None]:
    """Stream FineWeb documents, yielding (doc_id, text) pairs."""
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name=corpus_name,
        split="train",
        streaming=True,
        trust_remote_code=True,
    )
    count = 0
    for row in ds:
        if count >= n_docs:
            break
        doc_id = row.get("id") or str(uuid.uuid4())
        text = row.get(text_column, "")
        if text.strip():
            yield str(doc_id), text
            count += 1


def _select_positions(
    seq_len: int,
    positions_per_doc: int,
    min_position: int,
) -> list[int]:
    """Select up to `positions_per_doc` token positions uniformly from [min_position, seq_len-1]."""
    available = list(range(min_position, seq_len))
    if not available:
        return []
    if len(available) <= positions_per_doc:
        return sorted(available)
    stride = len(available) // positions_per_doc
    return sorted(available[i * stride] for i in range(positions_per_doc))


class ActivationExtractor:
    """
    Batched activation extractor using a frozen HF model with a StopForward hook.

    The model is loaded in bfloat16, frozen (no gradients), and run with
    torch.inference_mode(). A forward hook is installed on the target decoder
    layer that:
      1. Saves the full (batch, seq, d_model) hidden state.
      2. Raises _StopForward to abort the rest of the forward pass.

    All sequences in a batch are left-padded to the same length using the
    tokenizer's pad token on the left side. After the forward pass, we index
    into the saved hidden state using the actual (unpadded) token positions.
    """

    def __init__(
        self,
        model_name: str,
        layer_idx: int,
        max_seq_len: int = 512,
    ) -> None:
        self._layer_idx = layer_idx
        self._hook_key = f"layer_{layer_idx}"

        logger.info("Loading model: %s (bfloat16, frozen)", model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

        target_layer = self.model.model.layers[layer_idx]
        target_layer.register_forward_hook(_make_layer_hook(self._hook_key))
        logger.info("StopForward hook installed at layer %d", layer_idx)

        self._device = next(self.model.parameters()).device

    @torch.inference_mode()
    def extract_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run a left-padded batch through the model and return hidden states at the target layer.

        Args:
            input_ids: (batch, seq_len) long tensor, left-padded.
            attention_mask: (batch, seq_len) long tensor.

        Returns:
            Float32 CPU tensor of shape (batch, seq_len, d_model).
        """
        _ACT_STORE.clear()
        try:
            self.model(
                input_ids=input_ids.to(self._device),
                attention_mask=attention_mask.to(self._device),
            )
        except _StopForward:
            pass

        hidden = _ACT_STORE.get(self._hook_key)
        if hidden is None:
            raise RuntimeError(f"Hook at layer {self._layer_idx} did not fire.")
        return hidden  # (batch, seq_len, d_model) on CPU


def run_extraction(
    model_name: str,
    layer_idx: int,
    corpus_name: str,
    n_docs: int,
    positions_per_doc: int,
    min_position: int,
    max_seq_len: int,
    output_dir: str | Path,
    chunk_size: int = 512,
    batch_size: int = 512,
    resume: bool = False,
    seed: int = 42,
) -> Path:
    """
    Full Stage 0 pipeline: extract activations for all documents and save to Parquet.

    One Parquet chunk = chunk_size documents. Within each chunk, documents are
    batched in groups of batch_size for the GPU forward pass.

    The key design: we tokenize each document once (truncated to max_seq_len),
    select N positions, then run a single left-padded batch forward pass over all
    documents in the batch. We index into the returned (batch, seq, d_model) tensor
    at each document's requested positions using the left-padding offset.

    Args:
        model_name: HuggingFace model ID.
        layer_idx: Target decoder layer index.
        corpus_name: FineWeb subset name (e.g. "sample-10BT").
        n_docs: Total documents to process.
        positions_per_doc: Activation samples per document.
        min_position: Minimum token position (skip initial tokens with little context).
        max_seq_len: Maximum sequence length per document.
        output_dir: Directory to write Parquet chunk files.
        chunk_size: Documents per Parquet chunk (for resume granularity).
        batch_size: Documents per GPU forward pass (max throughput on H100: 512).
        resume: If True, skip already-written chunk files.
        seed: Random seed.

    Returns:
        Path to the output directory containing Parquet files.
    """
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    extractor = ActivationExtractor(
        model_name=model_name,
        layer_idx=layer_idx,
        max_seq_len=max_seq_len,
    )

    doc_stream = _stream_fineweb(corpus_name, n_docs)
    rows_written = 0
    chunk_idx = 0
    chunk_docs: list[tuple[str, list[int]]] = []  # (doc_id, token_ids)

    pbar = tqdm(total=n_docs, desc="Extracting", unit="doc", dynamic_ncols=True)

    def _flush_chunk(docs: list[tuple[str, list[int]]], c_idx: int) -> int:
        """Process one chunk and return number of rows written."""
        chunk_path = output_dir / f"chunk_{c_idx:05d}.parquet"
        if resume and chunk_path.exists():
            logger.info("Skipping existing chunk %d", c_idx)
            return 0

        rows = _process_chunk(
            docs=docs,
            extractor=extractor,
            tokenizer=tokenizer,
            layer_idx=layer_idx,
            positions_per_doc=positions_per_doc,
            min_position=min_position,
            batch_size=batch_size,
        )
        _write_parquet(rows, chunk_path)
        logger.info(
            "Chunk %05d: %d docs → %d activations (total so far: %d)",
            c_idx, len(docs), len(rows), rows_written + len(rows),
        )
        gc.collect()
        torch.cuda.empty_cache()
        return len(rows)

    for doc_id, text in doc_stream:
        token_ids = tokenizer.encode(text, add_special_tokens=False)[:max_seq_len]
        if len(token_ids) <= min_position:
            pbar.update(1)
            continue
        chunk_docs.append((doc_id, token_ids))
        pbar.update(1)

        if len(chunk_docs) >= chunk_size:
            rows_written += _flush_chunk(chunk_docs, chunk_idx)
            chunk_idx += 1
            chunk_docs = []

    # Final partial chunk
    if chunk_docs:
        rows_written += _flush_chunk(chunk_docs, chunk_idx)

    pbar.close()
    logger.info(
        "Stage 0 complete: %d activations in %d chunks → %s",
        rows_written, chunk_idx + 1, output_dir,
    )
    return output_dir


def _process_chunk(
    docs: list[tuple[str, list[int]]],
    extractor: ActivationExtractor,
    tokenizer: AutoTokenizer,
    layer_idx: int,
    positions_per_doc: int,
    min_position: int,
    batch_size: int,
) -> list[dict]:
    """
    Process one chunk of (doc_id, token_ids) pairs in GPU batches.

    For each mini-batch:
      1. Left-pad all sequences to the same length.
      2. Run one GPU forward pass.
      3. Recover the per-document hidden states from the padded tensor using offsets.
      4. Extract activations at each requested token position (in original, unpadded coords).
    """
    rows: list[dict] = []
    n = len(docs)

    for batch_start in range(0, n, batch_size):
        batch = docs[batch_start : batch_start + batch_size]
        ids_list = [ids for _, ids in batch]
        doc_ids = [did for did, _ in batch]

        # Left-pad
        enc = tokenizer.pad(
            {"input_ids": ids_list},
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"]          # (B, L)
        attention_mask = enc["attention_mask"] # (B, L)
        padded_len = input_ids.shape[1]

        hidden = extractor.extract_batch(input_ids, attention_mask)
        # hidden: (B, L, d_model)

        for b_idx, (doc_id, token_ids) in enumerate(zip(doc_ids, ids_list)):
            seq_len = len(token_ids)
            # Left-padding offset: last seq_len positions correspond to real tokens
            pad_offset = padded_len - seq_len

            positions = _select_positions(seq_len, positions_per_doc, min_position)
            for pos in positions:
                padded_pos = pad_offset + pos
                act = hidden[b_idx, padded_pos, :].numpy().astype(np.float32)

                start = max(0, pos - 5)
                snippet = tokenizer.decode(token_ids[start : pos + 6])[:200]

                rows.append({
                    "doc_id": doc_id,
                    "token_position": pos,
                    "text_snippet": snippet,
                    "activation_vector": act.tolist(),
                    "layer": layer_idx,
                })

    return rows


def _write_parquet(rows: list[dict], path: Path) -> None:
    """Write a list of row dicts to a zstd-compressed Parquet file."""
    if not rows:
        logger.warning("No rows to write for %s — skipping", path)
        return
    table = pa.table(
        {
            "doc_id": pa.array([r["doc_id"] for r in rows], pa.string()),
            "token_position": pa.array([r["token_position"] for r in rows], pa.int32()),
            "text_snippet": pa.array([r["text_snippet"] for r in rows], pa.string()),
            "activation_vector": pa.array(
                [r["activation_vector"] for r in rows], pa.list_(pa.float32())
            ),
            "layer": pa.array([r["layer"] for r in rows], pa.int32()),
        },
        schema=ACTIVATION_SCHEMA,
    )
    pq.write_table(table, path, compression="zstd")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 0: Extract activations (batched HF)")
    p.add_argument("--config", default="configs/qwen7b_layer20.yaml")
    p.add_argument("--output-dir", default=None, help="Override config output dir")
    p.add_argument("--n-docs", type=int, default=None, help="Override config n_docs")
    p.add_argument("--resume", action="store_true", help="Skip existing chunks")
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
    output_dir = args.output_dir or dg["output_dir"]
    n_docs = args.n_docs or dg["n_docs"]

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    run_extraction(
        model_name=cfg["target_model"],
        layer_idx=cfg["target_layer"],
        corpus_name=dg["corpus_name"],
        n_docs=n_docs,
        positions_per_doc=dg["positions_per_doc"],
        min_position=dg["min_token_position"],
        max_seq_len=dg["max_seq_len"],
        output_dir=output_dir,
        chunk_size=dg["chunk_size"],
        batch_size=dg.get("batch_size", 512),
        resume=args.resume,
        seed=cfg["seed"],
    )


if __name__ == "__main__":
    main()

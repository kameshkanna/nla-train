"""
Stage 2: Generate AV labels via the kitft SGLang checkpoint.

Calls the running SGLang server (kitft/nla-qwen2.5-7b-L20-av) on every
activation in the av_sft split to produce gold descriptions. These descriptions
become the training targets for our AV SFT stage and, via cross-join with the
ar_sft split, the inputs for our AR SFT stage.

The kitft AV checkpoint is our bootstrap oracle — it replaces the Claude API
calls used in the original pipeline. This is cheaper, offline-compatible, and
provides consistent labels aligned with the exact injection protocol we use.

Usage:
    # 1. Start SGLang server first:
    #    sglang serve --model-path kitft/nla-qwen2.5-7b-L20-av --port 30000 \
    #        --disable-radix-cache --mem-fraction-static 0.45
    #
    # 2. Then run:
    python -m nla_train.datagen.stage2_label \
        --config configs/qwen7b_layer20.yaml \
        --input-dir data/split \
        --output-dir data/labeled
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from pathlib import Path

import httpx
import numpy as np
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer

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
    Build the input_embeds array for a single activation vector, following
    the kitft injection protocol exactly.

    Returns:
        Float32 numpy array of shape (seq_len, d_model).
    """
    prompt_content = AV_PROMPT_TEMPLATE.format(injection_char=injection_char)
    formatted: str = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_content}],
        tokenize=False,
        add_generation_prompt=True,
    )
    ids: list[int] = tokenizer.encode(formatted, add_special_tokens=False)

    inject_pos: int | None = None
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
        raise RuntimeError(
            f"Injection token {injection_token_id} not found in tokenized prompt."
        )

    ids_tensor = torch.tensor(ids, dtype=torch.long)
    embeds = embed_weight[ids_tensor].clone().float()

    act = torch.from_numpy(activation.astype(np.float32))
    norm = act.norm()
    if norm > 1e-8:
        act = act / norm
    act = act * injection_scale
    embeds[inject_pos] = act

    return embeds.numpy().astype(np.float32)


def _call_sglang(
    embeds: np.ndarray,
    sglang_url: str,
    max_new_tokens: int,
    temperature: float,
    client: httpx.Client,
) -> str:
    """POST input_embeds to SGLang and extract the explanation text."""
    payload = {
        "input_embeds": embeds.tolist(),
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        },
    }
    resp = client.post(
        sglang_url.rstrip("/") + "/generate",
        content=orjson.dumps(payload, option=orjson.OPT_SERIALIZE_NUMPY),
        headers={"Content-Type": "application/json"},
        timeout=60.0,
    )
    resp.raise_for_status()
    text: str = resp.json()["text"]
    m = re.search(r"<explanation>(.*?)</explanation>", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def generate_labels(
    input_parquet: str | Path,
    output_parquet: str | Path,
    model_name: str,
    d_model: int,
    layer_idx: int,
    sglang_url: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    injection_cache_path: str | Path,
    nla_meta_output_path: str | Path,
    seed: int = 42,
) -> int:
    """
    Run the AV labeling pipeline on the av_sft split.

    Args:
        input_parquet: Path to av_sft.parquet from Stage 1.
        output_parquet: Output path for the labeled parquet.
        model_name: Target model name (for tokenizer and embedding table).
        d_model: Target model hidden dimension.
        layer_idx: Extraction layer.
        sglang_url: Running SGLang AV server URL.
        batch_size: Number of activations per SGLang batch.
        max_new_tokens: Max tokens per verbalization.
        temperature: Sampling temperature.
        injection_cache_path: Cache path for injection token selection.
        nla_meta_output_path: Where to write the nla_meta.yaml sidecar.
        seed: Random seed.

    Returns:
        Number of labeled rows written.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    logger.info("Loading tokenizer and embedding table: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map="cpu"
    )
    embed_weight = model.get_input_embeddings().weight.detach().float().cpu()
    del model
    import gc
    gc.collect()

    injection_char, injection_token_id = select_injection_token(
        tokenizer, cache_path=injection_cache_path
    )
    left_neighbor_id, right_neighbor_id = compute_injection_neighbors(
        tokenizer, injection_char
    )

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

    explanations: list[str] = []

    with httpx.Client() as http_client:
        for i in tqdm(range(n_rows), desc="Labeling activations", dynamic_ncols=True):
            act = activations[i]
            try:
                embeds = _build_embeds(
                    act, tokenizer, embed_weight,
                    injection_token_id, left_neighbor_id, right_neighbor_id,
                    injection_scale, injection_char,
                )
                explanation = _call_sglang(
                    embeds, sglang_url, max_new_tokens, temperature, http_client
                )
            except (httpx.HTTPError, RuntimeError) as e:
                logger.warning("Labeling failed for row %d: %s", i, e)
                explanation = ""
            explanations.append(explanation)

            if (i + 1) % 500 == 0:
                logger.info("Labeled %d/%d", i + 1, n_rows)

    # Filter out rows where labeling failed
    valid_mask = [bool(e) for e in explanations]
    n_valid = sum(valid_mask)
    logger.info("Valid labels: %d/%d (%.1f%%)", n_valid, n_rows, 100 * n_valid / n_rows)

    out_table = pa.table(
        {
            "doc_id": pa.array([doc_ids[i] for i, v in enumerate(valid_mask) if v]),
            "token_position": pa.array([positions[i] for i, v in enumerate(valid_mask) if v]),
            "text_snippet": pa.array([snippets[i] for i, v in enumerate(valid_mask) if v]),
            "activation_vector": pa.array(
                [activations[i].tolist() for i, v in enumerate(valid_mask) if v],
                pa.list_(pa.float32()),
            ),
            "layer": pa.array([layers[i] for i, v in enumerate(valid_mask) if v]),
            "explanation": pa.array([explanations[i] for i, v in enumerate(valid_mask) if v]),
        },
        schema=LABELED_SCHEMA,
    )

    Path(output_parquet).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, output_parquet, compression="zstd")
    logger.info("Wrote %d labeled rows → %s", n_valid, output_parquet)
    return n_valid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: Generate AV labels via SGLang")
    p.add_argument("--config", default="configs/qwen7b_layer20.yaml")
    p.add_argument("--input-dir", default=None)
    p.add_argument("--output-dir", default=None)
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
        batch_size=lab["batch_size"],
        max_new_tokens=lab["max_new_tokens"],
        temperature=lab["temperature"],
        injection_cache_path=inj["cache_file"],
        nla_meta_output_path=output_dir / "nla_meta_av.yaml",
        seed=cfg["seed"],
    )


if __name__ == "__main__":
    main()

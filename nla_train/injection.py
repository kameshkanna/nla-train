"""
CJK injection token protocol for NLA training.

The NLA training pipeline communicates activation vectors to the AV/AR models by
replacing the embedding of a designated "injection token" with the raw activation
vector. The injection token is a single CJK character (U+3200–U+33FF) chosen to:

  1. Tokenize to exactly one token in the target tokenizer.
  2. Be effectively absent from natural English text (no collision risk).
  3. Have identifiable left/right neighbor tokens so embedding injection can be
     validated at runtime (prevents false positives).

This module handles:
  - Selecting and caching the injection token.
  - Computing injection scale (so injected vector has same norm as real embeddings).
  - Injecting activation vectors into a batch of embeddings at runtime.
  - Generating and parsing the nla_meta.yaml sidecar file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from transformers import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

# AV prompt: describes the activation vector to the model.
AV_PROMPT_TEMPLATE = (
    "You are a meticulous AI researcher conducting an important investigation "
    "into activation vectors from a language model. Your overall task is to "
    "describe the semantic content of that activation vector.\n\n"
    "We will pass the vector enclosed in <concept> tags into your context. You "
    "must then produce an explanation for the vector, enclosed within <explanation> "
    "tags. The explanation consists of 2-3 text snippets describing that vector.\n\n"
    "Here is the vector:\n\n"
    "<concept>{injection_char}</concept>\n\n"
    "Please provide an explanation."
)

# AR prompt: given an explanation, reconstruct the activation vector.
AR_PROMPT_TEMPLATE = (
    "Summary of the following text: "
    "<text>{explanation}</text> <summary>"
)


def select_injection_token(
    tokenizer: PreTrainedTokenizerBase,
    cjk_start: int = 0x3200,
    cjk_end: int = 0x33FF,
    cache_path: Optional[str | Path] = None,
) -> tuple[str, int]:
    """
    Select a CJK character that tokenizes to exactly one token and cache it.

    Iterates through U+3200–U+33FF and returns the first character that the
    tokenizer encodes as a single token id. The result is cached to avoid
    re-running this on every startup.

    Args:
        tokenizer: Target model tokenizer.
        cjk_start: Unicode range start (inclusive).
        cjk_end: Unicode range end (inclusive).
        cache_path: Optional path to a YAML cache file. If the cache exists and
            contains a valid entry, it is returned immediately.

    Returns:
        (injection_char, injection_token_id) — the chosen character and its token id.

    Raises:
        RuntimeError: If no suitable single-token CJK character is found in range.
    """
    if cache_path is not None:
        p = Path(cache_path)
        if p.exists():
            with open(p) as f:
                cached = yaml.safe_load(f)
            char = cached.get("injection_char")
            tok_id = cached.get("injection_token_id")
            if char and tok_id is not None:
                logger.info("Loaded injection token from cache: %r (id=%d)", char, tok_id)
                return char, tok_id

    for codepoint in range(cjk_start, cjk_end + 1):
        char = chr(codepoint)
        ids = tokenizer.encode(char, add_special_tokens=False)
        if len(ids) == 1:
            logger.info("Selected injection token: %r (U+%04X, id=%d)", char, codepoint, ids[0])
            if cache_path is not None:
                p = Path(cache_path)
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, "w") as f:
                    yaml.dump({"injection_char": char, "injection_token_id": ids[0]}, f)
            return char, ids[0]

    raise RuntimeError(
        f"No single-token CJK character found in range U+{cjk_start:04X}–U+{cjk_end:04X} "
        f"for tokenizer {type(tokenizer).__name__}."
    )


def compute_injection_neighbors(
    tokenizer: PreTrainedTokenizerBase,
    injection_char: str,
    av_template: str = AV_PROMPT_TEMPLATE,
) -> tuple[int, int]:
    """
    Compute the canonical left and right neighbor token IDs surrounding the
    injection character in the AV prompt template.

    These neighbor IDs are stored in nla_meta.yaml and used at inference/training
    time to validate the injection position (prevents false positives if the
    injection character appears naturally elsewhere in the sequence).

    Args:
        tokenizer: Target tokenizer.
        injection_char: The selected CJK injection character.
        av_template: The AV prompt template string with {injection_char} placeholder.

    Returns:
        (left_neighbor_id, right_neighbor_id)

    Raises:
        RuntimeError: If the injection character cannot be located in the tokenized prompt.
    """
    prompt = av_template.format(injection_char=injection_char)
    messages = [{"role": "user", "content": prompt}]
    formatted: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    ids: list[int] = tokenizer.encode(formatted, add_special_tokens=False)

    inject_id = tokenizer.encode(injection_char, add_special_tokens=False)[0]
    positions = [i for i, t in enumerate(ids) if t == inject_id]

    if not positions:
        raise RuntimeError(
            f"Injection char {injection_char!r} (id={inject_id}) not found in "
            f"tokenized AV prompt. ids={ids[:30]}..."
        )

    pos = positions[0]
    if pos == 0 or pos == len(ids) - 1:
        raise RuntimeError(
            f"Injection token at boundary position {pos} — cannot extract neighbors."
        )

    left_id = ids[pos - 1]
    right_id = ids[pos + 1]
    logger.info(
        "Injection neighbors: left=%d right=%d (position %d in prompt)",
        left_id, right_id, pos,
    )
    return left_id, right_id


def compute_injection_scale(activation_vectors: np.ndarray) -> float:
    """
    Compute the injection scale factor from a sample of activation vectors.

    The injected vector is L2-normalized and then multiplied by this scale so
    the resulting embedding has approximately the same norm as real token embeddings.

    Args:
        activation_vectors: Float32 array of shape (N, d_model).

    Returns:
        Scalar injection scale = mean(||h_i||) over the sample.
    """
    norms = np.linalg.norm(activation_vectors, axis=1)
    scale = float(np.mean(norms))
    logger.info("Injection scale computed: %.4f (from %d vectors)", scale, len(norms))
    return scale


def inject_at_marked_positions(
    input_ids: torch.Tensor,
    embeddings: torch.Tensor,
    activation_vectors: torch.Tensor,
    injection_token_id: int,
    left_neighbor_id: int,
    right_neighbor_id: int,
    injection_scale: float,
) -> torch.Tensor:
    """
    Replace embeddings at injection positions with scaled activation vectors.

    Scans each sequence in the batch for the three-token pattern
    [left_neighbor_id, injection_token_id, right_neighbor_id] and overwrites
    the embedding at the injection position with the corresponding (normalized,
    scaled) activation vector from `activation_vectors`.

    Args:
        input_ids: Long tensor of shape (batch, seq_len).
        embeddings: Float tensor of shape (batch, seq_len, d_model). Modified in-place.
        activation_vectors: Float tensor of shape (batch, d_model) — one per sequence.
        injection_token_id: Token id of the CJK injection character.
        left_neighbor_id: Expected token id immediately left of injection position.
        right_neighbor_id: Expected token id immediately right of injection position.
        injection_scale: Scale applied after L2-normalization.

    Returns:
        Modified embeddings tensor (same object, modified in-place).

    Raises:
        RuntimeError: If the injection position cannot be found for any sequence
            in the batch, which indicates a template or tokenization mismatch.
    """
    batch_size, seq_len = input_ids.shape
    ids_np = input_ids.cpu().numpy()

    for b in range(batch_size):
        inject_pos: Optional[int] = None
        for i in range(1, seq_len - 1):
            if (
                ids_np[b, i] == injection_token_id
                and ids_np[b, i - 1] == left_neighbor_id
                and ids_np[b, i + 1] == right_neighbor_id
            ):
                inject_pos = i
                break

        if inject_pos is None:
            # Fallback: find by token id alone (without neighbor validation)
            candidates = np.where(ids_np[b] == injection_token_id)[0]
            if len(candidates) == 0:
                raise RuntimeError(
                    f"Batch item {b}: injection token id {injection_token_id} not found. "
                    f"ids={ids_np[b, :30].tolist()}..."
                )
            inject_pos = int(candidates[0])
            logger.warning(
                "Batch item %d: injection found via fallback (no neighbor match) at pos %d",
                b, inject_pos,
            )

        act = activation_vectors[b].float()
        norm = act.norm()
        if norm > 1e-8:
            act = act / norm
        act = act * injection_scale
        embeddings[b, inject_pos] = act.to(embeddings.device, embeddings.dtype)

    return embeddings


def write_nla_meta(
    output_path: str | Path,
    role: str,
    d_model: int,
    layer_idx: int,
    injection_char: str,
    injection_token_id: int,
    left_neighbor_id: int,
    right_neighbor_id: int,
    injection_scale: float,
    av_prompt_template: str = AV_PROMPT_TEMPLATE,
    ar_prompt_template: str = AR_PROMPT_TEMPLATE,
) -> None:
    """
    Write an nla_meta.yaml sidecar file compatible with the kitft schema_version=2 format.

    Args:
        output_path: Full path to write the YAML file.
        role: "av" or "ar".
        d_model: Model hidden dimension.
        layer_idx: Extraction layer index.
        injection_char: CJK injection character.
        injection_token_id: Its token id.
        left_neighbor_id: Left neighbor token id.
        right_neighbor_id: Right neighbor token id.
        injection_scale: Computed injection scale.
        av_prompt_template: AV prompt template string.
        ar_prompt_template: AR prompt template string.
    """
    meta = {
        "schema_version": 2,
        "role": role,
        "d_model": d_model,
        "extraction_layer_index": layer_idx,
        "extraction": {
            "injection_scale": injection_scale,
        },
        "tokens": {
            "injection_char": injection_char,
            "injection_token_id": injection_token_id,
            "injection_left_neighbor_id": left_neighbor_id,
            "injection_right_neighbor_id": right_neighbor_id,
        },
        "prompt_templates": {
            "av": av_prompt_template,
            "ar": ar_prompt_template,
        },
    }
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        yaml.dump(meta, f, allow_unicode=True, default_flow_style=False)
    logger.info("Wrote nla_meta.yaml to %s", p)

"""
Derive French-language CAA steering vectors for Qwen2.5-7B-Instruct.

Uses Contrastive Activation Addition (CAA): extract last-token hidden states
for French vs English completions of the same prompts, compute mean difference
per layer, L2-normalise. Calibrated K values are read from the actbak norm
profile so the scale is consistent with the safety vectors.

Target layers: 11–24 (40–90% depth, matching actbak Qwen convention).

Output: experiments/data/french_vectors.npz  — same schema as actbak safety.npz
  layer_indices     (14,)      int32
  k_values          (14,)      float32
  mean_directions   (14, 3584) float32  unit-norm
  pca_directions    (14, 3584) float32  unit-norm
  pca_variance_ratios (14,)   float32

Usage:
    python experiments/derive_french_vectors.py \
        --model Qwen/Qwen2.5-7B-Instruct \
        --norm-profile /path/to/activation-baking/results/norm_profiles/qwen2.5-7b-instruct.csv \
        --output-dir experiments/data \
        [--n-pairs 120] [--batch-size 4]
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Contrastive pairs for deriving a "respond in French" CAA direction.
#
# Design principle: the system/instruction differs between positive and
# negative — positive says "respond in French", negative says "respond in
# the user's language." The user turn and the assistant completion cover
# the SAME semantic content. This makes the CAA direction purely
# "output mode = French" with zero topic confound.
#
# Format: {
#   "french_prompt":  full text shown to model under "respond in French" mode
#   "english_prompt": full text shown to model under "respond normally" mode
# }
# We extract the last-token hidden state of each prompt, then
# direction = mean(h_french_prompt) - mean(h_english_prompt).
# The user question (topic-neutral, no French content) is the same in both.
# Only the system instruction differs: "respond in French" vs "respond in the user's language".
# This makes the CAA direction purely encode the output-language mode switch.
_USER_QUESTIONS = [
    "What is two plus two?",
    "How are you today?",
    "What is the tallest mountain on Earth?",
    "Explain how a computer works.",
    "What causes earthquakes?",
    "How do vaccines work?",
    "What is photosynthesis?",
    "Describe how the internet works.",
    "What is the boiling point of water?",
    "Why do leaves change colour in autumn?",
    "How do airplanes stay in the air?",
    "What is the speed of light?",
    "Explain what gravity is.",
    "What is a black hole?",
    "How do humans digest food?",
    "What is climate change?",
    "How does memory work in the brain?",
    "What is a prime number?",
    "Explain the water cycle.",
    "What causes thunder and lightning?",
    "How do fish breathe underwater?",
    "What is DNA?",
    "How do bees make honey?",
    "What is the theory of evolution?",
    "How does a battery work?",
    "What is machine learning?",
    "Explain supply and demand.",
    "What is the difference between a virus and a bacterium?",
    "How do solar panels generate electricity?",
    "What is the stock market?",
    "How do telescopes work?",
    "What is an algorithm?",
    "How does the moon affect tides?",
    "What is quantum mechanics?",
    "How is glass made?",
    "Explain what inflation means.",
    "What is the mitochondria?",
    "How do birds navigate during migration?",
    "What is the ozone layer?",
    "How does a combustion engine work?",
    "What is the meaning of entropy?",
    "Explain the Pythagorean theorem.",
    "How does Wi-Fi work?",
    "What is a neural network?",
    "How is steel produced?",
    "What causes ocean currents?",
    "How do plants reproduce?",
    "What is radioactivity?",
    "Explain how a microchip is made.",
    "What is the difference between weather and climate?",
]

def _make_chat(system: str, user: str, tokenizer) -> str:
    """Apply chat template with a given system prompt and user message."""
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": system},
         {"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )

SYSTEM_FRENCH = "You must respond exclusively in French, regardless of the language the user writes in."
SYSTEM_DEFAULT = "You are a helpful assistant. Respond in the same language the user writes in."

# Built lazily in derive_french_vectors() after tokenizer is loaded
FRENCH_PAIRS: list[dict] = [
    {"question": q} for q in _USER_QUESTIONS
]


TARGET_LAYERS = list(range(11, 25))  # 11–24, matching actbak Qwen convention


class _LayerHook:
    """Captures last-token hidden states at multiple layers in one forward pass."""

    def __init__(self, model: AutoModelForCausalLM, layer_indices: list[int]) -> None:
        self._captures: dict[int, torch.Tensor] = {}
        self._handles = []
        for idx in layer_indices:
            def _make(i: int):
                def _hook(module, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    self._captures[i] = h.detach().float().cpu()
                return _hook
            self._handles.append(model.model.layers[idx].register_forward_hook(_make(idx)))

    def clear(self) -> None:
        self._captures.clear()

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def last_token(self, layer_idx: int) -> torch.Tensor | None:
        h = self._captures.get(layer_idx)
        return h[0, -1] if h is not None else None


@torch.no_grad()
def _extract_activations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    layer_indices: list[int],
    batch_size: int,
    device: torch.device,
    completion_tokens: int = 10,
) -> dict[int, np.ndarray]:
    """Generate a short completion per text, then extract the activation at the
    first generated token position.

    Both the French-mode and English-mode prompts end with the same
    ``<|im_start|>assistant\\n`` token, so extracting from the prompt last-token
    gives a near-zero diff. Extracting from the first *generated* token captures
    the activations at the point where the model has already committed to a
    language (French ``"La"``/``"Le"`` vs English ``"The"``/``"I"``).

    Returns:
        {layer_idx: np.ndarray (N, d_model)}
    """
    hook = _LayerHook(model, layer_indices)
    out: dict[int, list[np.ndarray]] = {l: [] for l in layer_indices}

    for start in tqdm(range(0, len(texts), batch_size), desc="Extracting", leave=False):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=256).to(device)
        prompt_len = enc["input_ids"].shape[1]

        # Generate a short completion — we only need the first token's activation
        hook.clear()
        gen_ids = model.generate(
            **enc,
            max_new_tokens=completion_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        # gen_ids: (B, prompt_len + completion_tokens)
        # Re-run a full forward pass over the generated sequence to get clean hook captures
        hook.clear()
        full_enc = {
            "input_ids": gen_ids,
            "attention_mask": torch.ones_like(gen_ids),
        }
        model(**full_enc)

        for l in layer_indices:
            h = hook._captures.get(l)
            if h is not None:
                # Capture at the position of the first new token (prompt_len)
                for b_idx in range(len(batch)):
                    out[l].append(h[b_idx, prompt_len].float().numpy())

        del gen_ids, full_enc
        gc.collect()

    hook.remove()
    return {l: np.stack(v).astype(np.float32) for l, v in out.items() if v}


def derive_french_vectors(
    model_name: str,
    norm_profile_path: str,
    output_dir: str,
    batch_size: int = 4,
) -> None:
    """
    Derive French CAA steering vectors for Qwen2.5-7B-Instruct.

    Positive: system="respond in French" + question
    Negative: system="respond in user's language" + same question
    Direction = mean(h_positive) - mean(h_negative), unit-normalised per layer.

    Args:
        model_name: HF model ID.
        norm_profile_path: Path to actbak norm profile CSV.
        output_dir: Where to write french_vectors.npz.
        batch_size: Batch size for activation extraction.
    """
    out_path = Path(output_dir) / "french_vectors.npz"

    logger.info("Using %d contrastive question pairs", len(FRENCH_PAIRS))

    norm_df = pd.read_csv(norm_profile_path)
    k_lookup = dict(zip(norm_df["layer_idx"].astype(int), norm_df["k_value"].astype(float)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Build chat-formatted prompts — same question, only system instruction differs
    french_texts = [_make_chat(SYSTEM_FRENCH, p["question"], tokenizer) for p in FRENCH_PAIRS]
    english_texts = [_make_chat(SYSTEM_DEFAULT, p["question"], tokenizer) for p in FRENCH_PAIRS]
    logger.info("Example positive:\n%s", french_texts[0][:300])
    logger.info("Example negative:\n%s", english_texts[0][:300])

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": str(device)}
    )
    model.eval()

    logger.info("Extracting French-mode activations")
    french_acts = _extract_activations(model, tokenizer, french_texts, TARGET_LAYERS, batch_size, device)
    logger.info("Extracting default-mode activations")
    english_acts = _extract_activations(model, tokenizer, english_texts, TARGET_LAYERS, batch_size, device)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    layer_indices, k_values, mean_dirs, pca_dirs, pca_vars = [], [], [], [], []

    for l in TARGET_LAYERS:
        if l not in french_acts or l not in english_acts:
            continue

        diff = french_acts[l] - english_acts[l]  # (N, d)
        mean_diff = diff.mean(axis=0)
        norm = np.linalg.norm(mean_diff)
        mean_dir = (mean_diff / norm).astype(np.float32)

        pca = PCA(n_components=1, random_state=42)
        pca.fit(diff)
        pca_dir = pca.components_[0].astype(np.float32)
        # sign: align with mean direction
        if np.dot(pca_dir, mean_dir) < 0:
            pca_dir = -pca_dir

        layer_indices.append(l)
        k_values.append(float(k_lookup.get(l, 5.0)))
        mean_dirs.append(mean_dir)
        pca_dirs.append(pca_dir)
        pca_vars.append(float(pca.explained_variance_ratio_[0]))
        logger.info("Layer %d: mean_norm=%.3f  PCA_var=%.3f  K=%.4f",
                    l, float(norm), pca_vars[-1], k_values[-1])

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.savez(
        out_path,
        layer_indices=np.array(layer_indices, dtype=np.int32),
        k_values=np.array(k_values, dtype=np.float32),
        mean_directions=np.stack(mean_dirs).astype(np.float32),
        pca_directions=np.stack(pca_dirs).astype(np.float32),
        pca_variance_ratios=np.array(pca_vars, dtype=np.float32),
    )
    logger.info("Saved %s", out_path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Derive French CAA steering vectors")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--norm-profile",
                   default="experiments/steering_data/qwen2.5-7b-norm-profile.csv",
                   help="Path to actbak norm profile CSV")
    p.add_argument("--output-dir", default="experiments/steering_data")
    p.add_argument("--batch-size", type=int, default=4)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s — %(message)s",
                        datefmt="%H:%M:%S")
    args = parse_args()
    derive_french_vectors(
        model_name=args.model,
        norm_profile_path=args.norm_profile,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

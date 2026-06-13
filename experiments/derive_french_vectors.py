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

# 120 French/English contrastive prompt pairs.
# Each English prefix is paired with a French and English completion.
# The CAA direction = mean(h_french) - mean(h_english) per layer.
FRENCH_PAIRS: list[dict] = [
    # greetings / openers
    {"prefix": "Good morning!", "french": "Bonjour !", "english": "Hello!"},
    {"prefix": "How are you today?", "french": "Comment allez-vous aujourd'hui ?", "english": "I am doing well, thank you."},
    {"prefix": "What is your name?", "french": "Mon nom est Claude.", "english": "My name is Claude."},
    {"prefix": "Nice to meet you.", "french": "Enchanté de vous rencontrer.", "english": "It is nice to meet you too."},
    {"prefix": "Good evening!", "french": "Bonsoir !", "english": "Good evening!"},
    # factual responses
    {"prefix": "What is the capital of France?", "french": "La capitale de la France est Paris.", "english": "The capital of France is Paris."},
    {"prefix": "What language do they speak in France?", "french": "En France, on parle français.", "english": "In France, they speak French."},
    {"prefix": "What is water made of?", "french": "L'eau est composée d'hydrogène et d'oxygène.", "english": "Water is made of hydrogen and oxygen."},
    {"prefix": "How many days are in a week?", "french": "Il y a sept jours dans une semaine.", "english": "There are seven days in a week."},
    {"prefix": "What is the largest planet?", "french": "Jupiter est la plus grande planète du système solaire.", "english": "Jupiter is the largest planet in the solar system."},
    {"prefix": "Who wrote Romeo and Juliet?", "french": "Roméo et Juliette a été écrit par William Shakespeare.", "english": "Romeo and Juliet was written by William Shakespeare."},
    {"prefix": "What is the speed of light?", "french": "La vitesse de la lumière est d'environ 300 000 kilomètres par seconde.", "english": "The speed of light is approximately 300,000 kilometers per second."},
    {"prefix": "What is the boiling point of water?", "french": "L'eau bout à 100 degrés Celsius.", "english": "Water boils at 100 degrees Celsius."},
    {"prefix": "What is the chemical symbol for gold?", "french": "Le symbole chimique de l'or est Au.", "english": "The chemical symbol for gold is Au."},
    {"prefix": "What is the tallest mountain?", "french": "Le mont Everest est la plus haute montagne du monde.", "english": "Mount Everest is the tallest mountain in the world."},
    # instructions / tasks
    {"prefix": "Please summarize this article.", "french": "Je vais résumer cet article pour vous.", "english": "I will summarize this article for you."},
    {"prefix": "Can you help me with this problem?", "french": "Bien sûr, je peux vous aider avec ce problème.", "english": "Of course, I can help you with this problem."},
    {"prefix": "Please translate this sentence.", "french": "Je vais traduire cette phrase pour vous.", "english": "I will translate this sentence for you."},
    {"prefix": "Write a short story.", "french": "Il était une fois un jeune garçon qui vivait dans une forêt enchantée.", "english": "Once upon a time there was a young boy who lived in an enchanted forest."},
    {"prefix": "Explain machine learning.", "french": "L'apprentissage automatique est une branche de l'intelligence artificielle.", "english": "Machine learning is a branch of artificial intelligence."},
    {"prefix": "What should I eat for breakfast?", "french": "Je vous recommande des croissants et du café pour le petit-déjeuner.", "english": "I recommend eggs and toast for breakfast."},
    {"prefix": "How do I learn a new language?", "french": "Pour apprendre une nouvelle langue, pratiquez chaque jour et immergez-vous dans la culture.", "english": "To learn a new language, practice every day and immerse yourself in the culture."},
    {"prefix": "Give me a recipe for soup.", "french": "Pour faire une soupe, faites chauffer du bouillon avec des légumes et des herbes.", "english": "To make soup, heat broth with vegetables and herbs."},
    {"prefix": "How do I stay healthy?", "french": "Pour rester en bonne santé, faites de l'exercice régulièrement et mangez équilibré.", "english": "To stay healthy, exercise regularly and eat a balanced diet."},
    {"prefix": "Describe the weather today.", "french": "Aujourd'hui, le ciel est nuageux avec quelques éclaircies.", "english": "Today, the sky is cloudy with some sunny spells."},
    # conversational
    {"prefix": "What do you think about this?", "french": "Je pense que c'est une idée très intéressante.", "english": "I think this is a very interesting idea."},
    {"prefix": "Tell me something interesting.", "french": "Saviez-vous que les pieuvres ont trois cœurs ?", "english": "Did you know that octopuses have three hearts?"},
    {"prefix": "I am feeling sad today.", "french": "Je suis désolé d'apprendre que vous vous sentez triste. Comment puis-je vous aider ?", "english": "I am sorry to hear you are feeling sad. How can I help?"},
    {"prefix": "What is your favorite color?", "french": "Ma couleur préférée est le bleu, comme le ciel.", "english": "My favorite color is blue, like the sky."},
    {"prefix": "Do you like music?", "french": "Oui, j'aime beaucoup la musique, surtout la musique classique.", "english": "Yes, I enjoy music very much, especially classical music."},
    # longer completions for richer signal
    {"prefix": "Tell me about Paris.", "french": "Paris est la capitale de la France et l'une des plus belles villes du monde. La ville est connue pour la tour Eiffel, le musée du Louvre et sa cuisine exquise.", "english": "Paris is the capital of France and one of the most beautiful cities in the world. The city is known for the Eiffel Tower, the Louvre museum, and its exquisite cuisine."},
    {"prefix": "Describe a forest.", "french": "Une forêt est un vaste écosystème composé d'arbres, de plantes, d'animaux et de micro-organismes qui vivent en harmonie.", "english": "A forest is a vast ecosystem composed of trees, plants, animals, and microorganisms living in harmony."},
    {"prefix": "What is democracy?", "french": "La démocratie est un système de gouvernement dans lequel les citoyens exercent le pouvoir par le vote.", "english": "Democracy is a system of government in which citizens exercise power through voting."},
    {"prefix": "Explain the water cycle.", "french": "Le cycle de l'eau décrit comment l'eau circule continuellement sur Terre par évaporation, condensation et précipitation.", "english": "The water cycle describes how water continuously circulates on Earth through evaporation, condensation, and precipitation."},
    {"prefix": "What is artificial intelligence?", "french": "L'intelligence artificielle est la simulation de l'intelligence humaine par des machines programmées pour penser et apprendre.", "english": "Artificial intelligence is the simulation of human intelligence by machines programmed to think and learn."},
    # numbers / days / months in French to reinforce lexical signal
    {"prefix": "Count to five.", "french": "Un, deux, trois, quatre, cinq.", "english": "One, two, three, four, five."},
    {"prefix": "What day is today?", "french": "Aujourd'hui c'est lundi.", "english": "Today is Monday."},
    {"prefix": "What month is it?", "french": "Nous sommes en janvier.", "english": "We are in January."},
    {"prefix": "What season is it?", "french": "C'est l'automne.", "english": "It is autumn."},
    {"prefix": "What time is it?", "french": "Il est trois heures de l'après-midi.", "english": "It is three o'clock in the afternoon."},
    # politeness phrases
    {"prefix": "Thank you very much.", "french": "Merci beaucoup, c'est très gentil.", "english": "Thank you very much, that is very kind."},
    {"prefix": "You are welcome.", "french": "De rien, c'est avec plaisir.", "english": "You are welcome, it is my pleasure."},
    {"prefix": "Excuse me.", "french": "Excusez-moi, puis-je vous demander quelque chose ?", "english": "Excuse me, may I ask you something?"},
    {"prefix": "I am sorry.", "french": "Je suis vraiment désolé pour ce malentendu.", "english": "I am truly sorry for this misunderstanding."},
    {"prefix": "Please.", "french": "S'il vous plaît, pourriez-vous m'aider ?", "english": "Please, could you help me?"},
    # food / culture
    {"prefix": "What is a croissant?", "french": "Un croissant est une viennoiserie française en forme de croissant de lune, feuilletée et beurrée.", "english": "A croissant is a French pastry shaped like a crescent moon, flaky and buttery."},
    {"prefix": "What is wine?", "french": "Le vin est une boisson alcoolisée fabriquée à partir de raisins fermentés.", "english": "Wine is an alcoholic beverage made from fermented grapes."},
    {"prefix": "Describe French cuisine.", "french": "La cuisine française est réputée pour sa sophistication, sa richesse en saveurs et ses techniques culinaires raffinées.", "english": "French cuisine is renowned for its sophistication, richness of flavors, and refined culinary techniques."},
    {"prefix": "What is a baguette?", "french": "Une baguette est un pain français long et croustillant, emblème de la culture française.", "english": "A baguette is a long, crusty French bread, emblematic of French culture."},
    {"prefix": "Tell me about cheese.", "french": "La France produit plus de 1000 variétés de fromages, dont le camembert, le brie et le roquefort.", "english": "France produces more than 1000 varieties of cheese, including camembert, brie, and roquefort."},
]

# Extend to ~120 pairs by paraphrasing the first 70
_EXTRA_PAIRS = [
    {"prefix": p["prefix"] + " Please be brief.", "french": p["french"], "english": p["english"]}
    for p in FRENCH_PAIRS[:70]
]
FRENCH_PAIRS = FRENCH_PAIRS + _EXTRA_PAIRS


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
) -> dict[int, np.ndarray]:
    """Extract last-token activations at all target layers. Returns {layer: (N, d)}."""
    hook = _LayerHook(model, layer_indices)
    out: dict[int, list[np.ndarray]] = {l: [] for l in layer_indices}

    for start in tqdm(range(0, len(texts), batch_size), desc="Extracting", leave=False):
        batch = texts[start : start + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=256).to(device)
        hook.clear()
        model(**enc)
        for l in layer_indices:
            h = hook._captures.get(l)
            if h is not None:
                # last non-pad token per sequence
                seq_lens = enc["attention_mask"].sum(dim=1) - 1
                for b_idx in range(len(batch)):
                    out[l].append(h[b_idx, seq_lens[b_idx].item()].numpy())

    hook.remove()
    return {l: np.stack(v).astype(np.float32) for l, v in out.items() if v}


def derive_french_vectors(
    model_name: str,
    norm_profile_path: str,
    output_dir: str,
    n_pairs: int = 120,
    batch_size: int = 4,
) -> None:
    """
    Derive French CAA steering vectors for Qwen2.5-7B-Instruct.

    Args:
        model_name: HF model ID.
        norm_profile_path: Path to actbak norm profile CSV.
        output_dir: Where to write french_vectors.npz.
        n_pairs: Number of contrastive pairs to use.
        batch_size: Batch size for activation extraction.
    """
    out_path = Path(output_dir) / "french_vectors.npz"
    if out_path.exists():
        logger.info("french_vectors.npz already exists — skipping")
        return

    pairs = FRENCH_PAIRS[:n_pairs]
    french_texts = [p["prefix"] + " " + p["french"] for p in pairs]
    english_texts = [p["prefix"] + " " + p["english"] for p in pairs]
    logger.info("Using %d contrastive pairs", len(pairs))

    norm_df = pd.read_csv(norm_profile_path)
    k_lookup = dict(zip(norm_df["layer_idx"].astype(int), norm_df["k_value"].astype(float)))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map={"": str(device)}
    )
    model.eval()

    logger.info("Extracting French activations")
    french_acts = _extract_activations(model, tokenizer, french_texts, TARGET_LAYERS, batch_size, device)
    logger.info("Extracting English activations")
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
    p.add_argument("--norm-profile", required=True,
                   help="Path to actbak qwen2.5-7b-instruct.csv norm profile")
    p.add_argument("--output-dir", default="experiments/data")
    p.add_argument("--n-pairs", type=int, default=120)
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
        n_pairs=args.n_pairs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()

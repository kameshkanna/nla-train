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
# Design principle: topic-neutral prefixes (no French topics, no France, no
# French culture). The ONLY difference between french and english completions
# is the output language. This ensures the CAA direction captures
# "output mode = French" not "thinking about French things."
#
# Pairs span diverse topics (science, math, animals, weather, technology,
# history, everyday tasks) so the direction generalises across domains.
FRENCH_PAIRS: list[dict] = [
    # science / nature
    {"prefix": "What is photosynthesis?",
     "french": "La photosynthèse est le processus par lequel les plantes convertissent la lumière solaire en énergie.",
     "english": "Photosynthesis is the process by which plants convert sunlight into energy."},
    {"prefix": "How do volcanoes form?",
     "french": "Les volcans se forment lorsque le magma remonte à travers les fissures de la croûte terrestre.",
     "english": "Volcanoes form when magma rises through cracks in the Earth's crust."},
    {"prefix": "What is gravity?",
     "french": "La gravité est la force qui attire les objets les uns vers les autres.",
     "english": "Gravity is the force that attracts objects toward each other."},
    {"prefix": "How do stars form?",
     "french": "Les étoiles se forment à partir de nuages de gaz et de poussière qui s'effondrent sous leur propre gravité.",
     "english": "Stars form from clouds of gas and dust that collapse under their own gravity."},
    {"prefix": "What is the water cycle?",
     "french": "Le cycle de l'eau décrit comment l'eau circule entre les océans, l'atmosphère et la terre.",
     "english": "The water cycle describes how water circulates between the oceans, atmosphere, and land."},
    {"prefix": "Why is the sky blue?",
     "french": "Le ciel est bleu parce que l'atmosphère diffuse la lumière bleue plus que les autres couleurs.",
     "english": "The sky is blue because the atmosphere scatters blue light more than other colours."},
    {"prefix": "What is DNA?",
     "french": "L'ADN est une molécule qui contient les instructions génétiques pour le développement et le fonctionnement de tous les êtres vivants.",
     "english": "DNA is a molecule that contains the genetic instructions for the development and functioning of all living organisms."},
    {"prefix": "How does the immune system work?",
     "french": "Le système immunitaire protège le corps en détectant et en détruisant les agents pathogènes.",
     "english": "The immune system protects the body by detecting and destroying pathogens."},
    # mathematics
    {"prefix": "What is a prime number?",
     "french": "Un nombre premier est un nombre naturel supérieur à 1 qui n'a pas d'autres diviseurs que 1 et lui-même.",
     "english": "A prime number is a natural number greater than 1 that has no divisors other than 1 and itself."},
    {"prefix": "What is the Pythagorean theorem?",
     "french": "Le théorème de Pythagore stipule que dans un triangle rectangle, le carré de l'hypoténuse est égal à la somme des carrés des deux autres côtés.",
     "english": "The Pythagorean theorem states that in a right triangle, the square of the hypotenuse equals the sum of the squares of the other two sides."},
    {"prefix": "What is a fraction?",
     "french": "Une fraction représente une partie d'un tout, exprimée sous la forme d'un numérateur divisé par un dénominateur.",
     "english": "A fraction represents a part of a whole, expressed as a numerator divided by a denominator."},
    {"prefix": "What is calculus?",
     "french": "Le calcul infinitésimal est une branche des mathématiques qui étudie les taux de variation et l'accumulation.",
     "english": "Calculus is a branch of mathematics that studies rates of change and accumulation."},
    # animals
    {"prefix": "How do bees make honey?",
     "french": "Les abeilles fabriquent le miel en collectant le nectar des fleurs et en le transformant dans la ruche.",
     "english": "Bees make honey by collecting nectar from flowers and processing it in the hive."},
    {"prefix": "Why do birds migrate?",
     "french": "Les oiseaux migrent pour trouver de la nourriture et des conditions climatiques plus favorables.",
     "english": "Birds migrate to find food and more favourable weather conditions."},
    {"prefix": "How do spiders spin webs?",
     "french": "Les araignées filent leurs toiles en produisant de la soie à partir de glandes spécialisées dans leur abdomen.",
     "english": "Spiders spin webs by producing silk from specialised glands in their abdomen."},
    {"prefix": "What do elephants eat?",
     "french": "Les éléphants mangent des herbes, des feuilles, des fruits et des écorces d'arbres.",
     "english": "Elephants eat grasses, leaves, fruits, and tree bark."},
    {"prefix": "How do fish breathe?",
     "french": "Les poissons respirent en extrayant l'oxygène dissous dans l'eau à travers leurs branchies.",
     "english": "Fish breathe by extracting dissolved oxygen from water through their gills."},
    # weather / environment
    {"prefix": "What causes thunder?",
     "french": "Le tonnerre est causé par l'expansion rapide de l'air chauffé par la foudre.",
     "english": "Thunder is caused by the rapid expansion of air heated by lightning."},
    {"prefix": "What is climate change?",
     "french": "Le changement climatique désigne les modifications à long terme des températures et des conditions météorologiques mondiales.",
     "english": "Climate change refers to long-term shifts in global temperatures and weather patterns."},
    {"prefix": "How do rainbows form?",
     "french": "Les arcs-en-ciel se forment lorsque la lumière solaire est réfractée et réfléchie dans les gouttes de pluie.",
     "english": "Rainbows form when sunlight is refracted and reflected inside raindrops."},
    # technology
    {"prefix": "What is the internet?",
     "french": "Internet est un réseau mondial d'ordinateurs interconnectés qui communiquent via des protocoles standardisés.",
     "english": "The internet is a global network of interconnected computers that communicate via standardised protocols."},
    {"prefix": "How does a computer processor work?",
     "french": "Un processeur exécute des instructions en effectuant des opérations arithmétiques et logiques sur des données.",
     "english": "A processor executes instructions by performing arithmetic and logical operations on data."},
    {"prefix": "What is machine learning?",
     "french": "L'apprentissage automatique est une méthode permettant aux ordinateurs d'apprendre à partir de données sans être explicitement programmés.",
     "english": "Machine learning is a method that allows computers to learn from data without being explicitly programmed."},
    {"prefix": "What is a database?",
     "french": "Une base de données est un système organisé pour stocker, gérer et récupérer des informations.",
     "english": "A database is an organised system for storing, managing, and retrieving information."},
    {"prefix": "How does encryption work?",
     "french": "Le chiffrement transforme les données lisibles en un format illisible à l'aide d'un algorithme et d'une clé.",
     "english": "Encryption transforms readable data into an unreadable format using an algorithm and a key."},
    # history / society (non-French topics)
    {"prefix": "What caused the First World War?",
     "french": "La Première Guerre mondiale a été déclenchée par l'assassinat de l'archiduc François-Ferdinand et par un réseau d'alliances entre les nations.",
     "english": "The First World War was triggered by the assassination of Archduke Franz Ferdinand and a network of alliances between nations."},
    {"prefix": "What is democracy?",
     "french": "La démocratie est un système de gouvernement dans lequel les citoyens exercent le pouvoir en votant.",
     "english": "Democracy is a system of government in which citizens exercise power by voting."},
    {"prefix": "What was the Industrial Revolution?",
     "french": "La révolution industrielle fut une période de transition vers de nouveaux procédés de fabrication en Europe et aux États-Unis.",
     "english": "The Industrial Revolution was a period of transition to new manufacturing processes in Europe and the United States."},
    # everyday tasks / instructions
    {"prefix": "How do you bake bread?",
     "french": "Pour faire du pain, mélangez de la farine, de l'eau, de la levure et du sel, puis laissez lever avant de cuire.",
     "english": "To bake bread, mix flour, water, yeast, and salt, then let it rise before baking."},
    {"prefix": "How do you change a tyre?",
     "french": "Pour changer un pneu, soulevez le véhicule avec un cric, retirez les boulons, remplacez le pneu et resserrez les boulons.",
     "english": "To change a tyre, lift the vehicle with a jack, remove the bolts, replace the tyre, and retighten the bolts."},
    {"prefix": "How do you plant a tree?",
     "french": "Pour planter un arbre, creusez un trou, placez l'arbre à la verticale, remplissez de terre et arrosez abondamment.",
     "english": "To plant a tree, dig a hole, place the tree upright, fill with soil, and water thoroughly."},
    {"prefix": "How do you write a good email?",
     "french": "Pour écrire un bon e-mail, utilisez un objet clair, un ton approprié, et soyez concis.",
     "english": "To write a good email, use a clear subject line, an appropriate tone, and be concise."},
    # short single-sentence pairs — highest signal-to-noise for CAA
    {"prefix": "The sun rises in the east.", "french": "Le soleil se lève à l'est.", "english": "The sun rises in the east."},
    {"prefix": "Dogs are loyal animals.", "french": "Les chiens sont des animaux loyaux.", "english": "Dogs are loyal animals."},
    {"prefix": "Water boils at 100 degrees.", "french": "L'eau bout à 100 degrés.", "english": "Water boils at 100 degrees."},
    {"prefix": "Exercise is good for health.", "french": "L'exercice est bon pour la santé.", "english": "Exercise is good for health."},
    {"prefix": "Reading improves vocabulary.", "french": "La lecture améliore le vocabulaire.", "english": "Reading improves vocabulary."},
    {"prefix": "The moon orbits the Earth.", "french": "La lune tourne autour de la Terre.", "english": "The moon orbits the Earth."},
    {"prefix": "Sleep is essential for recovery.", "french": "Le sommeil est essentiel pour la récupération.", "english": "Sleep is essential for recovery."},
    {"prefix": "Mathematics is the language of science.", "french": "Les mathématiques sont le langage de la science.", "english": "Mathematics is the language of science."},
    {"prefix": "Trees absorb carbon dioxide.", "french": "Les arbres absorbent le dioxyde de carbone.", "english": "Trees absorb carbon dioxide."},
    {"prefix": "Light travels faster than sound.", "french": "La lumière voyage plus vite que le son.", "english": "Light travels faster than sound."},
    # assistant-style response openers (highest value for steering output mode)
    {"prefix": "User: What is two plus two?\nAssistant:", "french": " Deux plus deux égale quatre.", "english": " Two plus two equals four."},
    {"prefix": "User: How are you?\nAssistant:", "french": " Je vais bien, merci de demander.", "english": " I am doing well, thank you for asking."},
    {"prefix": "User: What time is it?\nAssistant:", "french": " Je ne peux pas voir l'heure, mais je peux vous aider autrement.", "english": " I cannot see the time, but I can help you in other ways."},
    {"prefix": "User: Explain gravity briefly.\nAssistant:", "french": " La gravité est la force qui attire les masses les unes vers les autres.", "english": " Gravity is the force that attracts masses toward each other."},
    {"prefix": "User: What is the tallest mountain?\nAssistant:", "french": " Le mont Everest est la plus haute montagne du monde.", "english": " Mount Everest is the tallest mountain in the world."},
    {"prefix": "User: Give me a fun fact.\nAssistant:", "french": " Les pieuvres ont trois cœurs et du sang bleu.", "english": " Octopuses have three hearts and blue blood."},
    {"prefix": "User: How does a rainbow form?\nAssistant:", "french": " Un arc-en-ciel se forme quand la lumière est réfractée dans les gouttes d'eau.", "english": " A rainbow forms when light is refracted through water droplets."},
    {"prefix": "User: What should I eat for breakfast?\nAssistant:", "french": " Je vous suggère des œufs, des fruits et du pain complet.", "english": " I suggest eggs, fruit, and whole-grain bread."},
    {"prefix": "User: How do I stay focused?\nAssistant:", "french": " Éliminez les distractions, faites des pauses régulières et fixez-vous des objectifs clairs.", "english": " Eliminate distractions, take regular breaks, and set clear goals."},
    {"prefix": "User: What is the capital of Japan?\nAssistant:", "french": " La capitale du Japon est Tokyo.", "english": " The capital of Japan is Tokyo."},
    {"prefix": "User: How do plants grow?\nAssistant:", "french": " Les plantes poussent en absorbant l'eau, les nutriments du sol et la lumière du soleil.", "english": " Plants grow by absorbing water, nutrients from the soil, and sunlight."},
    {"prefix": "User: Tell me a joke.\nAssistant:", "french": " Pourquoi les plongeurs plongent-ils toujours en arrière ? Parce que sinon ils tomberaient dans le bateau.", "english": " Why do scuba divers always fall backwards off the boat? Because if they fell forward they'd still be in the boat."},
]

# Double the dataset by repeating with a "Please answer:" prefix variant
# so the direction averages over multiple prompt formats
_EXTRA_PAIRS = [
    {"prefix": "Please answer: " + p["prefix"], "french": p["french"], "english": p["english"]}
    for p in FRENCH_PAIRS
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
    p.add_argument("--norm-profile",
                   default="experiments/steering_data/qwen2.5-7b-norm-profile.csv",
                   help="Path to actbak norm profile CSV")
    p.add_argument("--output-dir", default="experiments/steering_data")
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

---
license: apache-2.0
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
  - mechanistic-interpretability
  - activation-steering
  - natural-language-autoencoders
  - lora
language:
  - en
---

# NLA — Qwen2.5-7B Layer 20 Activation Verbalizer

A LoRA adapter that converts a residual-stream activation vector from Qwen2.5-7B into a natural-language description of what that vector represents.

Built on the **Natural Language Autoencoder (NLA)** framework ([Fraser-Taliente et al., 2026](https://transformer-circuits.pub/2026/nla/index.html)). Trained on layer 20 via a 3-stage pipeline: AR SFT → AV SFT → RL (GRPO). Runs on a single H100, ~$35 total.

---

## How it works

The NLA framework has two components trained jointly:

- **AV (Activation Verbalizer)** — this model. Takes a raw activation vector, injects it into the token embedding stream at a special placeholder position, and generates a free-text description of its semantic content.
- **AR (Activation Reconstructor)** — a separate model that reads the AV's description and reconstructs the original activation vector. Used during RL training as the reward signal (reward = −MSE between reconstruction and true activation).

The injection mechanism replaces the embedding of a reserved single-token placeholder (`㈀`, U+3200) with the raw activation vector, so the model "reads" the activation directly without any projection head.

---

## Quick Start

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, "Kameshr/nla-qwen2.5-7b-L20-av")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model.eval()
```

### Extract a layer-20 activation

```python
def extract_layer20(text: str, model, tokenizer) -> torch.Tensor:
    captured = {}
    def hook(mod, inp, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()
        raise StopIteration
    enc = tokenizer(text, return_tensors="pt").to(model.device)
    handle = model.model.layers[20].register_forward_hook(hook)
    try:
        with torch.no_grad(): model(**enc)
    except StopIteration:
        pass
    finally:
        handle.remove()
    return captured["h"][0, -1]  # last-token activation, shape (3584,)
```

### Verbalize it

```python
INJECTION_CHAR = "㈀"  # U+3200 — single token in Qwen tokenizer

@torch.no_grad()
def verbalize(activation: torch.Tensor, model, tokenizer, max_new_tokens=80) -> str:
    prompt = (
        "You are a meticulous AI researcher investigating activation vectors from a language model. "
        "Describe the semantic content of the vector enclosed in <concept> tags.\n\n"
        f"<concept>{INJECTION_CHAR}</concept>\n\nPlease provide an explanation."
    )
    enc = tokenizer(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        ),
        return_tensors="pt",
    ).to(model.device)

    embeds = model.get_input_embeddings()(enc["input_ids"]).clone()
    inj_pos = (enc["input_ids"][0] == tokenizer.encode(INJECTION_CHAR, add_special_tokens=False)[0]).nonzero()[0, 0]
    embeds[0, inj_pos] = activation.to(embeds.dtype).to(embeds.device)

    out = model.generate(
        inputs_embeds=embeds,
        attention_mask=enc["attention_mask"],
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)
```

### Example

```python
text = "Photo caption: A golden retriever puppy sitting in a field of sunflowers,"
act = extract_layer20(text, base, tokenizer)
print(verbalize(act, model, tokenizer))
# → "A happy puppy sitting in a field of flowers, dog with colorful flowers around it"
```

The AV reads the trailing comma and anticipates what comes next — the description includes content not yet present in the input, showing the model is reading the *activation state* rather than just the surface text.

---

## Generalization: Does Training on One Layer Transfer to Others?

Training a separate NLA verbalizer per layer is expensive — 28 independent training runs for Qwen2.5-7B. We investigated whether a single L20-trained AV can be applied to other layers without retraining, and how performance degrades as a function of layer distance.

### Experimental Setup

We constructed a 2,000-text evaluation corpus sampled from five domains — FineWeb (web text), Wikipedia, PubMed (biomedical), GitHub (code), and Reddit — with 400 texts per domain. For each text, we extracted residual-stream activations from all 28 decoder layers in a single forward pass, yielding a (2,000 × 28 × 3,584) activation tensor. The L20-trained AV was then applied to every (text, layer) pair — 56,000 inferences in total — without any layer-specific adaptation. Each generated description was passed through the AR model to reconstruct an activation vector, which was compared against the ground-truth activation.

### Metrics

**Cosine Similarity (CS)** measures directional agreement between the AR-reconstructed activation and the true activation, ignoring vector magnitude. It is the appropriate metric here because activation norms vary substantially across layers (early layers have systematically lower norms than late layers), and a magnitude-sensitive metric like MSE would conflate norm differences with semantic fidelity. CS = 1 means perfect directional recovery; CS ≈ 0 is the expected value for random vectors.

**Recall@10** is a retrieval metric that evaluates semantic discriminability. For each text at each layer, we ask: given only the AV's description of that text, can the AR model recover an activation vector that ranks in the top 10 (out of 500 randomly sampled layer-matched activations) by cosine similarity to the true activation? This tests whether descriptions are specific enough to identify a particular sample in a pool — a much stricter criterion than average similarity. A random baseline achieves Recall@10 ≈ 0.02 (1/500 × 10).

### Statistical Testing

For each layer, we tested whether the AV's per-sample cosine similarities were significantly greater than the random-vector baseline using a **Wilcoxon signed-rank test** — a non-parametric test chosen because cosine similarity distributions are bounded, asymmetric, and not guaranteed to be normal. To control the false discovery rate across 56 simultaneous comparisons (28 layers × 2 baselines), we applied **Benjamini–Hochberg correction** at α = 0.05. All 28 layers were significant against both the random Gaussian baseline and the shuffled-activation baseline after correction.

### Results

Performance peaks at L20 (the training layer) and decays smoothly in both directions. The decay is gradual rather than abrupt — layers 10–25 all achieve CS > 0.50 and Recall@10 > 0.40, indicating that the AV's learned representation space is broadly compatible with the residual stream geometry across the middle portion of the network. The two failure modes are structurally distinct:

**Early layers (L0–L9):** Low but non-zero performance. Early layers encode surface-level token features — character n-grams, part-of-speech patterns — that are geometrically distant from the mid-network semantic representations the AV was trained on. Transfer is weak but statistically above chance.

**Final layer (L27):** Near-complete failure (CS = 0.215, Recall@10 = 0.005). The pre-unembedding layer is dominated by vocabulary-projection geometry — activations are pulled strongly toward logit directions — which is categorically different from the semantic subspace the AV operates in. This failure is structural, not a matter of distance from the training layer.

**Norm-scaling ablation:** We tested whether the performance decay was an artifact of inter-layer norm differences. Rescaling all input activations to match the layer-20 median norm before AV inference produced no meaningful change (< 0.002 cosine difference at every layer). The AV operates on activation *direction*, not magnitude — norm scaling is irrelevant.

### Practical Implication

The smooth decay profile suggests that full-network coverage does not require 28 independent models. A small number of strategically placed verbalizers — trained at early, mid, and late anchor layers — can cover the network with a controlled accuracy tradeoff. For Qwen2.5-7B, 2–3 AVs appear sufficient to maintain Recall@10 ≥ 0.40 across all layers except L27, reducing training cost by roughly 10× relative to per-layer training.

Full experiment code, figures, and per-layer results: [github.com/kameshkanna/nla-train/tree/main/experiments](https://github.com/kameshkanna/nla-train/tree/main/experiments)

---

## Training details

| Stage | Config | Time |
|---|---|---|
| AR SFT | Truncated Qwen2.5-7B (layers 0–20), LoRA r=64, MSE loss | ~1h |
| AV SFT | Full Qwen2.5-7B, LoRA r=32, CE loss, kitft AV as label oracle | ~1.5h |
| RL (GRPO) | TRL GRPOTrainer, reward = −MSE(AR(description), activation), 1,250 steps | ~7h |
| **Total** | **1× H100 80GB, Lambda Labs** | **~$35** |

RL was run for fewer steps than the original NLA paper due to compute budget — descriptions are structurally correct but less specific than a full RL run. The SFT baseline is strong; additional RL steps would close the gap to kitft.

Cost reductions vs original: single GPU (no Megatron), AR frozen during RL, vLLM colocate mode for generation, LoRA throughout.

---

## Citation

If you use this model, please cite the original NLA paper:

```bibtex
@misc{fraser-taliente2026nla,
  title={Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations},
  author={Fraser-Taliente and Kantamneni and Ong et al.},
  year={2026},
  url={https://transformer-circuits.pub/2026/nla/index.html}
}
```

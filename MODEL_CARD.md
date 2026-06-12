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

A LoRA adapter that turns a residual stream activation vector into a natural language explanation of what it represents.

Built using the **Natural Language Autoencoder (NLA)** framework from [Fraser-Taliente et al., 2026](https://transformer-circuits.pub/2026/nla/index.html). Trained on Qwen2.5-7B-Instruct, layer 20, via 3-stage pipeline: AR SFT → AV SFT → RL (GRPO). Single H100, ~$25 total.

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
    return captured["h"][0, -1]  # last token, shape (3584,)
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

The AV reads the comma and plans ahead — the description includes content not yet in the input.

---

## Generalization

Trained on layer 20, evaluated across all 28 layers (2000 texts, 5 domains):

| Layer range | Cosine Similarity | Recall@10 |
|---|---|---|
| L10–L14 | 0.52–0.57 | 0.40–0.55 |
| L15–L19 | 0.60–0.67 | 0.59–0.65 |
| **L20 (train)** | **0.692** | **0.682** |
| L21–L25 | 0.56–0.67 | 0.42–0.67 |
| L27 | 0.215 | 0.005 |

28/28 layers statistically significant vs random baseline (Wilcoxon, BH-corrected). Norm scaling has no effect — the AV learned directional, not magnitude-based, representations.

Full experiment: [github.com/kameshkanna/nla-train/experiments](https://github.com/kameshkanna/nla-train)

---

## Training

| Stage | Details |
|---|---|
| AR SFT | Truncated Qwen2.5-7B (layers 0–20), LoRA r=64, MSE loss, 1h |
| AV SFT | Full Qwen2.5-7B, LoRA r=32, CE loss on kitft-labeled descriptions, 1.5h |
| RL GRPO | TRL GRPOTrainer, reward = −MSE(AR(description), activation), 1250 steps, 7h |
| Compute | 1× H100 80GB, ~$25 total |

---

## Citation

```bibtex
@misc{fraser-taliente2026nla,
  title={Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations},
  author={Fraser-Taliente and Kantamneni and Ong et al.},
  year={2026},
  url={https://transformer-circuits.pub/2026/nla/index.html}
}
```

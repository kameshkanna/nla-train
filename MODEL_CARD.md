# NLA Qwen2.5-7B Layer 20 — AV Checkpoint

**Natural Language Autoencoder (NLA)** for Qwen2.5-7B-Instruct, layer 20.  
Trained to verbalize residual stream activations as natural language explanations.

---

## What This Is

An **Activation Verbalizer (AV)** — a LoRA-adapted Qwen2.5-7B-Instruct model that takes a residual stream activation vector from layer 20 of Qwen2.5-7B-Instruct and generates a natural language explanation of what that activation represents.

This is a reproduction of the NLA methodology from:
> *Fraser-Taliente, Kantamneni, Ong et al. "Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations." Transformer Circuits, 2026.*  
> [transformer-circuits.pub/2026/nla](https://transformer-circuits.pub/2026/nla/index.html)

---

## Quick Start

```python
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)
av_model = PeftModel.from_pretrained(base, "kameshkanna/nla-qwen2.5-7b-L20-av")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Inject activation and generate explanation
def verbalize(activation: torch.Tensor, injection_char="㈀", max_new_tokens=100):
    """
    activation: float tensor of shape (3584,) — layer 20 residual stream.
    Returns: explanation string.
    """
    prompt = (
        "You are a meticulous AI researcher conducting an important investigation "
        "into activation vectors from a language model. Your overall task is to "
        "describe the semantic content of that activation vector.\n\n"
        "We will pass the vector enclosed in <concept> tags into your context. You "
        "must then produce an explanation for the vector, enclosed within <explanation> "
        "tags. The explanation consists of 2-3 text snippets describing that vector.\n\n"
        f"Here is the vector:\n\n<concept>{injection_char}</concept>\n\nPlease provide an explanation."
    )
    messages = [{"role": "user", "content": prompt}]
    prompt_str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt_str, return_tensors="pt").to(av_model.device)

    # Inject activation at the injection token position
    embed = av_model.get_input_embeddings()
    embeds = embed(enc["input_ids"]).clone()
    inj_id = tokenizer.encode(injection_char, add_special_tokens=False)[0]
    pos = (enc["input_ids"][0] == inj_id).nonzero(as_tuple=True)[0][0]
    act = activation.float()
    act = act / (act.norm() + 1e-8)
    act = act * activation.norm()  # restore scale
    embeds[0, pos] = act.to(embeds.dtype).to(embeds.device)

    out = av_model.generate(
        inputs_embeds=embeds, attention_mask=enc["attention_mask"],
        max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)
```

---

## Extracting Activations from Qwen2.5-7B

```python
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

class StopForward(Exception): pass

def extract_layer20(text: str) -> torch.Tensor:
    """Returns activations of shape (seq_len, 3584) at layer 20."""
    captured = {}
    def hook(mod, inp, out):
        captured["h"] = (out[0] if isinstance(out, tuple) else out).detach().float().cpu()
        raise StopForward()

    enc = tokenizer(text, return_tensors="pt").to(model.device)
    handle = model.model.layers[20].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**enc)
    except StopForward:
        pass
    finally:
        handle.remove()
    return captured["h"][0]  # (seq_len, 3584)

# Get activation at last token position
text = "Photo caption: A golden retriever puppy sitting in a field of sunflowers,"
acts = extract_layer20(text)
last_token_act = acts[-1]  # (3584,)
explanation = verbalize(last_token_act)
print(explanation)
# → "<explanation> ... A happy puppy sitting in a field of flowers, dog with colorful flowers ... </explanation>"
```

---

## Injection Token

The CJK character `㈀` (U+3200) is used as the injection slot. It tokenizes to a single token in Qwen2.5-7B-Instruct and is absent from natural English text. The full `nla_meta.yaml` sidecar is included in this repo with all injection parameters.

---

## Training Details

### Architecture
- **Base model**: Qwen2.5-7B-Instruct
- **Target layer**: 20 (of 28 total decoder layers, ~71% depth)
- **d_model**: 3584
- **AV adapter**: LoRA rank=32, alpha=64, attention modules only
- **AR model**: Truncated Qwen2.5-7B (layers 0–20), no final layernorm, linear value head initialized to identity

### Pipeline

We reproduced the NLA training pipeline from scratch in three stages:

**Stage 1 — AR SFT** (`nla_train/ar_sft.py`)  
Trained the Activation Reconstructor to map (description → activation). 100k FineWeb documents, 25k used for AR SFT. Loss converged to 0.0001 in ~1h on 1× H100 80GB.

**Stage 2 — AV SFT** (`nla_train/av_sft.py`)  
Supervised fine-tuning of the AV on (activation → description) pairs, with gold labels generated by the kitft reference AV model (used as oracle instead of Claude API — same principle, zero cost). Converged in ~1.5h, final loss 1.74.

**Stage 3 — RL GRPO** (`nla_train/rl_grpo.py`)  
Joint optimization using TRL's GRPOTrainer. Reward = −MSE(L2_norm(AR(description)), L2_norm(gold_activation)). 50k samples × 4 generations, 1250 steps, ~7h.

### Compute Budget & Cost Optimizations

We trained on **1× H100 80GB** (Lambda Labs) at ~$2.49/hr. Total training cost: **~$25**.

Original kitft pipeline uses multi-GPU Megatron + Claude API calls. Key changes we made to fit single-GPU:

| Decision | Reason |
|---|---|
| LoRA rank 32 (AV) / 64 (AR) instead of full fine-tune | ~4× memory savings |
| `gradient_checkpointing=True` on AV | Full 28-layer model fits in 80GB |
| `CUDA_VISIBLE_DEVICES=0` — single GPU, no DataParallel | `nn.DataParallel` with 2 GPUs caused `ReduceAddCoalesced` OOM at backward |
| vLLM colocate mode (`vllm_mode="colocate"`) | 28s/it → 8.5s/it — removed the key speed bottleneck |
| AR frozen during RL (not jointly updated) | AR backward on single GPU OOMed; AR SFT at loss=0.0001 is a reliable oracle |
| kitft AV as label oracle (not Claude API) | Zero API cost for AV SFT labels |
| `max_completion_length=100` | Reduced from 150 to fit GPU budget with vLLM colocate |

### Results

| Metric | Ours | kitft reference |
|---|---|---|
| Reconstruction MSE (normalized) | 0.463 | 0.348 |
| Mean description length | 75.2 words | 76.9 words |
| Planning signal (comma probe) | ✓ detected | ✓ detected |

The +33% MSE gap vs kitft is attributable to frozen AR during RL. Joint AR training would recover most of this gap at the cost of ~2× GPU memory during RL.

### Training Code

Full training code: [github.com/kameshkanna/nla-train](https://github.com/kameshkanna/nla-train)

---

## Evaluation: Planning Detection

NLAs can surface **planning** — representations of future tokens before they are generated.

```bash
# Token-level eval — Neuronpedia style
python -m nla_train.token_eval \
    --config configs/qwen7b_layer20.yaml \
    --av-checkpoint checkpoints/grpo/final_av \
    --ar-checkpoint checkpoints/ar_sft/final \
    --nla-meta data/labeled/nla_meta_av.yaml \
    --text "Photo caption: A golden retriever puppy sitting in a field of sunflowers," \
    --focus-token ","
```

At the comma (end of first clause), the AV produces:
> *"A happy puppy sitting in a field of flowers, **dog with colorful flowers around it**"*

The bolded phrase was not in the input — it is the model's planned continuation, encoded in the layer-20 residual stream.

---

## Limitations

- Trained on FineWeb (English web text). Performance on code, math, or non-English text is untested.
- Layer 20 is ~71% depth. Planning signals for complex multi-step reasoning may be stronger at layers 24–27.
- AR model is frozen (not jointly updated during RL) — descriptions are somewhat less specific than kitft's reference.
- NLA explanations can be incorrect. Always treat them as hypotheses, not ground truth.

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

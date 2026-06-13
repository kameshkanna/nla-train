# nla-train

Training code for **Natural Language Autoencoders (NLA)** on Qwen2.5-7B-Instruct.  
Reproduces the [Fraser-Taliente et al., 2026](https://transformer-circuits.pub/2026/nla/index.html) pipeline on a single H100 for ~$35.

Trained checkpoint: [Kameshr/nla-qwen2.5-7b-L20-av](https://huggingface.co/Kameshr/nla-qwen2.5-7b-L20-av)

---

## What is an NLA?

An NLA has two components:

- **AV (Activation Verbalizer)** — takes a residual-stream activation vector, injects it into the token embedding stream at a reserved placeholder token, and generates a free-text description of its semantic content.
- **AR (Activation Reconstructor)** — reads the AV's description and reconstructs the original activation vector. Used during RL training as the reward signal.

Together they form an autoencoder in natural language: `activation → description → activation`.

---

## Repo Layout

```text
nla-train/
├── configs/
│   └── qwen7b_layer20.yaml     # all hyperparams
├── nla_train/
│   ├── injection.py            # CJK injection token + embedding injection
│   ├── ar_sft.py               # AR supervised fine-tuning
│   ├── av_sft.py               # AV supervised fine-tuning
│   ├── rl_grpo.py              # RL training with TRL GRPOTrainer
│   ├── validate.py             # validation + kitft comparison
│   └── token_eval.py           # per-token Neuronpedia-style evaluation
├── experiments/
│   ├── extract_all_layers.py   # step 1: extract (N, 28, 3584) activations
│   ├── run_av_sweep.py         # step 2: AV on all 56k (text, layer) pairs
│   ├── run_ar_sweep.py         # step 3: AR reconstruction + baselines
│   ├── compute_metrics.py      # step 4: CS, FVE, Recall@10, nRMSE, Wilcoxon+BH
│   ├── plot_results.py         # step 5: 6 paper-ready figures
│   └── run_generalization.sh   # end-to-end pipeline script
├── scripts/
│   ├── run_rl.sh               # launch RL training
│   ├── run_validation.sh       # run validation
│   └── run_token_eval.sh       # per-token eval
├── setup_rl_env.sh             # training environment
├── setup_val_env.sh            # inference / eval environment
└── accelerate_config.yaml      # single-GPU accelerate config
```

---

## Environments

Two isolated environments — keep them separate, they have conflicting torch/vLLM versions.

### Training env (RL + SFT)

```bash
bash setup_rl_env.sh
source nla-rl-env/bin/activate
```

Stack: `torch 2.9.1 · vLLM 0.16.0 · TRL 1.5.1 · transformers 4.57.6 · peft 0.19.1`

Use for: AR SFT, AV SFT, RL GRPO training.

### Inference env (validation, eval, experiments)

```bash
bash setup_val_env.sh
source nla-val-env/bin/activate
```

Stack: `torch 2.5.1 · transformers 4.47.0 · peft 0.14.0 · accelerate 1.2.1`

Use for: validation, token eval, generalization experiments.

> vLLM 0.16.0 requires torch 2.9.1 which conflicts with transformers 4.47.0 — that is why two envs are necessary.

---

## Training Pipeline

### 1. Data generation

```bash
source nla-rl-env/bin/activate
bash scripts/run_datagen.sh
```

Streams 100k FineWeb documents, extracts layer-20 activations, generates AV labels using the kitft reference checkpoint as oracle, packs train splits.

### 2. AR SFT

```bash
bash scripts/run_ar_sft.sh
```

Trains a truncated Qwen2.5-7B (layers 0–20 only) with an identity-init linear value head to reconstruct activations from descriptions. Target loss ≤ 0.001.

### 3. AV SFT

```bash
bash scripts/run_av_sft.sh
```

Supervised fine-tuning of the full Qwen2.5-7B on (activation → description) pairs using kitft-labeled gold descriptions.

### 4. RL GRPO

```bash
bash scripts/run_rl.sh
```

Joint optimization via TRL GRPOTrainer. Reward = −MSE between AR reconstruction and true activation. Runs on a single GPU with vLLM colocate mode.

```bash
# Key env vars set inside run_rl.sh
export CUDA_VISIBLE_DEVICES=0          # single GPU, prevents DataParallel OOM
export PYTORCH_ALLOC_CONF=expandable_segments:True
```

---

## Validation

```bash
source nla-val-env/bin/activate
bash scripts/run_validation.sh
```

Runs both your checkpoint and the kitft reference side-by-side and reports MSE, description length, and qualitative examples.

### Per-token evaluation (planning detection)

```bash
source nla-val-env/bin/activate
python -m nla_train.token_eval \
    --config configs/qwen7b_layer20.yaml \
    --av-checkpoint checkpoints/grpo/final_av \
    --ar-checkpoint checkpoints/ar_sft/final \
    --nla-meta data/labeled/nla_meta_av.yaml \
    --text "Photo caption: A golden retriever puppy sitting in a field of sunflowers," \
    --focus-token ","
```

---

## Generalization Experiment

Tests whether the L20 AV transfers to all 28 layers without retraining.

```bash
source nla-val-env/bin/activate
bash experiments/run_generalization.sh \
    --av-checkpoint checkpoints/grpo/final_av \
    --ar-checkpoint checkpoints/ar_sft/final
```

Produces 6 figures in `experiments/figures/` and full metrics in `experiments/results/`.  
Key result: layers 10–25 all achieve Recall@10 > 0.40 with zero retraining. See the [model card](https://huggingface.co/Kameshr/nla-qwen2.5-7b-L20-av) for the full writeup.

---

## Steering × AV Evaluation

Tests whether the AV can detect what a steering vector is doing to the residual stream.
Compares three conditions at layers 18–22: baseline (clean), broadcast injection, and
last-token injection. Two behaviors: safety vectors (from [actbak](https://github.com/kameshkanna/activation-baking))
and French-language CAA vectors derived from 120 contrastive pairs.

### Step 1 — Derive French vectors

```bash
source nla-val-env/bin/activate
python experiments/derive_french_vectors.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --norm-profile /path/to/activation-baking/results/norm_profiles/qwen2.5-7b-instruct.csv \
    --output-dir experiments/data
```

Saves `experiments/data/french_vectors.npz` — same schema as the actbak safety vectors.

### Step 2 — Run steering evaluation

```bash
python experiments/steering_av_eval.py \
    --config configs/qwen7b_layer20.yaml \
    --av-checkpoint checkpoints/grpo/final_av \
    --nla-meta data/labeled/nla_meta_av.yaml \
    --actbak-dir /path/to/activation-baking \
    --output-dir experiments/results \
    --k-scale 1.0 \
    --n-texts 40
```

Produces:

- `experiments/results/steering_eval.json` — per-sample descriptions + cosine shift metrics
- `experiments/figures/steering_eval_cosine_shift.png` — activation shift vs baseline per layer × mode × behavior
- `experiments/figures/steering_eval_detection_rate.png` — did the AV describe the steered concept?
- `experiments/figures/steering_eval_qualitative.png` — description comparison grid at L20

K values are read directly from the actbak norm profile (L18=4.933, L19=5.099, L20=5.355,
L21=5.734, L22=6.189) so scale is consistent with actbak's ramp evaluation. Use `--k-scale`
to explore conservative (0.5) or strong (2.0) intervention strengths.

---

## Using the Trained Checkpoint

See the [model card](https://huggingface.co/Kameshr/nla-qwen2.5-7b-L20-av) for a self-contained usage example with no dependency on this training repo.

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

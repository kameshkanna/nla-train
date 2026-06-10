"""
AR SFT: Activation Reconstructor Supervised Fine-Tuning.

Trains the AR model to reconstruct activation vectors from natural language
descriptions. The AR model is a truncated version of the target model (first
`target_layer` decoder layers only) with no final LayerNorm and a linear
value head initialized to the identity.

Architecture:
  - Backbone: Qwen2.5-7B-Instruct, layers 0..target_layer (inclusive), truncated
  - No final LayerNorm (we want raw residual stream at layer target_layer)
  - Value head: Linear(d_model, d_model), initialized to torch.eye(d_model)
  - LoRA adapters on all attention + MLP projection layers

Input format:
  "Summary of the following text: <text>{explanation}</text> <summary>"
  where the injection_char in the template is replaced by the actual CJK token,
  and at forward pass time the embedding at that position is replaced with the
  gold activation vector.

Loss:
  MSE(L2_norm(value_head(last_token_hidden)), L2_norm(gold_activation))

Usage:
    python -m nla_train.ar_sft \
        --config configs/qwen7b_layer20.yaml \
        --data-dir data/train \
        --nla-meta data/labeled/nla_meta_av.yaml
"""

from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

from nla_train.injection import (
    AR_PROMPT_TEMPLATE,
    inject_at_marked_positions,
    select_injection_token,
    compute_injection_neighbors,
)

logger = logging.getLogger(__name__)


class TruncatedARModel(nn.Module):
    """
    Qwen2.5-7B-Instruct truncated to the first `n_layers` decoder layers.

    Exposes only the embedding layer, the first n_layers transformer blocks,
    and a linear value head. No final LayerNorm — we want the raw residual
    stream at the truncation point to match what was extracted during datagen.

    Args:
        base_model: Full HuggingFace causal LM (only its layers are used).
        n_layers: Number of decoder layers to keep (= target_layer + 1).
        d_model: Hidden dimension.
    """

    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        n_layers: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.embed_tokens = base_model.model.embed_tokens
        self.layers = nn.ModuleList(list(base_model.model.layers[:n_layers]))
        self.d_model = d_model

        # Value head: d_model → d_model, initialized to identity.
        # Identity init yields ~17% better loss vs Kaiming on NLA (per kitft notes).
        self.value_head = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            self.value_head.weight.copy_(torch.eye(d_model))

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        activation_vectors: Optional[torch.Tensor] = None,
        injection_token_id: Optional[int] = None,
        left_neighbor_id: Optional[int] = None,
        right_neighbor_id: Optional[int] = None,
        injection_scale: Optional[float] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with optional activation injection.

        Args:
            input_ids: (batch, seq_len) long tensor.
            attention_mask: (batch, seq_len) float tensor.
            activation_vectors: (batch, d_model) float tensor of gold activations.
                If provided, the embedding at the injection position is replaced.
            injection_token_id / left_neighbor_id / right_neighbor_id / injection_scale:
                Injection protocol parameters. Required when activation_vectors is set.

        Returns:
            Tensor of shape (batch, d_model) — value head output at the last token.
        """
        if inputs_embeds is not None:
            hidden = inputs_embeds
        else:
            hidden = self.embed_tokens(input_ids)  # (batch, seq, d_model)

        if activation_vectors is not None:
            hidden = inject_at_marked_positions(
                input_ids=input_ids,
                embeddings=hidden,
                activation_vectors=activation_vectors,
                injection_token_id=injection_token_id,
                left_neighbor_id=left_neighbor_id,
                right_neighbor_id=right_neighbor_id,
                injection_scale=injection_scale,
            )

        for layer in self.layers:
            layer_out = layer(hidden, attention_mask=attention_mask)
            hidden = layer_out[0] if isinstance(layer_out, tuple) else layer_out

        # Extract last real token position per sequence
        if attention_mask is not None:
            lengths = attention_mask.sum(dim=1) - 1
            last_pos = lengths.clamp(min=0).long()
        else:
            last_pos = torch.full(
                (hidden.size(0),), hidden.size(1) - 1, dtype=torch.long, device=hidden.device
            )
        last_hidden = hidden[torch.arange(hidden.size(0), device=hidden.device), last_pos]

        return self.value_head(last_hidden)  # (batch, d_model)


class ARDataset(Dataset):
    """
    Dataset for AR SFT training.

    Each item: tokenized AR prompt with injection placeholder + gold activation.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer: AutoTokenizer,
        injection_char: str,
        max_length: int = 256,
    ) -> None:
        import numpy as np
        table = pq.read_table(parquet_path)
        self._explanations: list[str] = table.column("explanation").to_pylist()
        n = len(table)
        d = len(table.column("activation_vector")[0].as_py())
        act_col = table.column("activation_vector").combine_chunks()
        self._activations: np.ndarray = act_col.values.to_numpy(zero_copy_only=False).reshape(n, d)
        self._tokenizer = tokenizer
        self._injection_char = injection_char
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._explanations)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        explanation = self._explanations[idx]
        activation = torch.from_numpy(self._activations[idx].copy())

        prompt = AR_PROMPT_TEMPLATE.format(explanation=explanation)
        messages = [{"role": "user", "content": prompt}]
        formatted: str = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        enc = self._tokenizer(
            formatted,
            max_length=self._max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "activation_vector": activation,
        }


def mse_loss_normalized(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Normalized MSE loss: MSE(L2_norm(predictions), L2_norm(targets)).

    This is mathematically equivalent to 2*(1 - cosine_similarity), so
    minimizing this loss maximizes cosine alignment between prediction and target.

    Args:
        predictions: (batch, d_model)
        targets: (batch, d_model)

    Returns:
        Scalar loss.
    """
    pred_norm = nn.functional.normalize(predictions, dim=-1)
    tgt_norm = nn.functional.normalize(targets, dim=-1)
    return nn.functional.mse_loss(pred_norm, tgt_norm)


def train_ar_sft(
    config_path: str,
    data_dir: str | Path,
    nla_meta_path: str | Path,
) -> None:
    """
    Full AR SFT training loop.

    Args:
        config_path: Path to qwen7b_layer20.yaml.
        data_dir: Directory containing ar_sft_train.parquet.
        nla_meta_path: Path to nla_meta_av.yaml (contains injection token info).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    with open(nla_meta_path) as f:
        nla_meta = yaml.safe_load(f)

    ar_cfg = cfg["ar_sft"]
    inj_cfg = cfg["injection"]
    seed = cfg["seed"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    injection_char = nla_meta["tokens"]["injection_char"]
    injection_token_id = nla_meta["tokens"]["injection_token_id"]
    left_neighbor_id = nla_meta["tokens"]["injection_left_neighbor_id"]
    right_neighbor_id = nla_meta["tokens"]["injection_right_neighbor_id"]
    injection_scale = nla_meta["extraction"]["injection_scale"]

    logger.info("Loading tokenizer: %s", cfg["target_model"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["target_model"], trust_remote_code=True)

    logger.info("Loading base model: %s", cfg["target_model"])
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["target_model"],
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    n_layers = cfg["target_layer"] + 1
    ar_model = TruncatedARModel(
        base_model=base_model,
        n_layers=n_layers,
        d_model=cfg["d_model"],
    )
    del base_model
    gc.collect()

    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=ar_cfg["lora_rank"],
        lora_alpha=ar_cfg["lora_alpha"],
        lora_dropout=ar_cfg["lora_dropout"],
        target_modules=ar_cfg["target_modules"],
        bias="none",
    )
    ar_model = get_peft_model(ar_model, lora_config)
    ar_model.to(device)
    ar_model.train()
    ar_model.print_trainable_parameters()
    logger.info("Model on GPU — loading dataset")

    dataset = ARDataset(
        parquet_path=Path(data_dir) / "ar_sft_train.parquet",
        tokenizer=tokenizer,
        injection_char=injection_char,
    )
    loader = DataLoader(
        dataset,
        batch_size=ar_cfg["per_device_train_batch_size"],
        shuffle=True,
        num_workers=ar_cfg["dataloader_num_workers"],
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(ar_model.parameters(), lr=ar_cfg["learning_rate"])
    total_steps = len(loader) * ar_cfg["num_train_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=ar_cfg["warmup_steps"],
        num_training_steps=total_steps,
        num_cycles=0.5,
    )

    output_dir = Path(ar_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    accum_steps = ar_cfg["gradient_accumulation_steps"]
    best_loss = float("inf")

    for epoch in range(ar_cfg["num_train_epochs"]):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"AR SFT epoch {epoch+1}", dynamic_ncols=True)

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            activation_vectors = batch["activation_vector"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                predictions = ar_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    activation_vectors=activation_vectors,
                    injection_token_id=injection_token_id,
                    left_neighbor_id=left_neighbor_id,
                    right_neighbor_id=right_neighbor_id,
                    injection_scale=injection_scale,
                )
                loss = mse_loss_normalized(predictions, activation_vectors.float())
                loss = loss / accum_steps

            loss.backward()
            epoch_loss += loss.item() * accum_steps

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(ar_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = epoch_loss / (step + 1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

                if global_step % ar_cfg["logging_steps"] == 0:
                    logger.info(
                        "Step %d | loss=%.4f | lr=%.2e",
                        global_step, avg_loss, scheduler.get_last_lr()[0],
                    )

                if global_step % ar_cfg["save_steps"] == 0:
                    ckpt_path = output_dir / f"checkpoint-{global_step}"
                    ar_model.save_pretrained(ckpt_path)
                    logger.info("Saved checkpoint: %s", ckpt_path)

        avg_epoch_loss = epoch_loss / len(loader)
        logger.info("Epoch %d complete | avg_loss=%.4f", epoch + 1, avg_epoch_loss)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            ar_model.save_pretrained(output_dir / "best")
            logger.info("New best checkpoint saved (loss=%.4f)", best_loss)

    ar_model.save_pretrained(output_dir / "final")
    logger.info("AR SFT complete. Final model at %s/final", output_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AR SFT Training")
    p.add_argument("--config", default="configs/qwen7b_layer20.yaml")
    p.add_argument("--data-dir", default="data/train")
    p.add_argument("--nla-meta", default="data/labeled/nla_meta_av.yaml")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    train_ar_sft(
        config_path=args.config,
        data_dir=args.data_dir,
        nla_meta_path=args.nla_meta,
    )


if __name__ == "__main__":
    main()

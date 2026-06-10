"""
AV SFT: Activation Verbalizer Supervised Fine-Tuning.

Trains the AV model to generate natural language descriptions of activation
vectors. The AV model is the full target model (all 28 layers) fine-tuned
with LoRA. At training time, the embedding at the injection position is replaced
with the gold activation vector from the training corpus.

Loss: Standard cross-entropy on the response tokens only (masked elsewhere).

Usage:
    python -m nla_train.av_sft \
        --config configs/qwen7b_layer20.yaml \
        --data-dir data/train \
        --nla-meta data/labeled/nla_meta_av.yaml
"""

from __future__ import annotations

import argparse
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

from nla_train.injection import AV_PROMPT_TEMPLATE, inject_at_marked_positions

logger = logging.getLogger(__name__)


class AVDataset(Dataset):
    """
    Dataset for AV SFT training.

    Each item: tokenized (prompt + response) with loss mask covering only the
    response tokens, plus the gold activation vector for embedding injection.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer: AutoTokenizer,
        injection_char: str,
        max_length: int = 512,
        max_response_length: int = 150,
    ) -> None:
        table = pq.read_table(parquet_path)
        self._explanations: list[str] = table.column("explanation").to_pylist()
        n = len(table)
        d = len(table.column("activation_vector")[0].as_py())
        act_col = table.column("activation_vector").combine_chunks()
        self._activations: np.ndarray = act_col.values.to_numpy(zero_copy_only=False).reshape(n, d)
        self._tokenizer = tokenizer
        self._injection_char = injection_char
        self._max_length = max_length
        self._max_response_length = max_response_length

    def __len__(self) -> int:
        return len(self._explanations)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        explanation = self._explanations[idx]
        activation = torch.from_numpy(self._activations[idx].copy())

        prompt_content = AV_PROMPT_TEMPLATE.format(injection_char=self._injection_char)
        response = f"<explanation>{explanation}</explanation>"

        messages = [{"role": "user", "content": prompt_content}]
        formatted_prompt: str = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        full_text = formatted_prompt + response

        prompt_ids = self._tokenizer.encode(formatted_prompt, add_special_tokens=False)
        full_enc = self._tokenizer(
            full_text,
            max_length=self._max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = full_enc["input_ids"].squeeze(0)
        attention_mask = full_enc["attention_mask"].squeeze(0)

        # Build loss mask: -100 on prompt tokens, real ids on response tokens
        labels = input_ids.clone()
        prompt_len = min(len(prompt_ids), self._max_length)
        labels[:prompt_len] = -100
        # Also mask padding
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "activation_vector": activation,
        }


class AVModelWrapper(nn.Module):
    """
    Wraps a LoRA-patched AV model to perform embedding injection before
    each forward pass.

    Injection is done by computing inputs_embeds directly (bypassing the
    embed_tokens call inside the model), injecting activation vectors in-place
    on that fresh tensor, then passing inputs_embeds to the model. This avoids
    the hook + gradient-checkpointing recomputation conflict where a saved
    leaf tensor cannot be modified in-place.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        injection_token_id: int,
        left_neighbor_id: int,
        right_neighbor_id: int,
        injection_scale: float,
    ) -> None:
        super().__init__()
        self.model = model
        self._injection_token_id = injection_token_id
        self._left_neighbor_id = left_neighbor_id
        self._right_neighbor_id = right_neighbor_id
        self._injection_scale = injection_scale

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        activation_vectors: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> object:
        embed_layer = self.model.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids).clone()  # clone breaks view of weight matrix

        if activation_vectors is not None:
            inputs_embeds = inject_at_marked_positions(
                input_ids=input_ids,
                embeddings=inputs_embeds,
                activation_vectors=activation_vectors,
                injection_token_id=self._injection_token_id,
                left_neighbor_id=self._left_neighbor_id,
                right_neighbor_id=self._right_neighbor_id,
                injection_scale=self._injection_scale,
            )

        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

    def remove_hooks(self) -> None:
        pass  # no hooks registered


def train_av_sft(
    config_path: str,
    data_dir: str | Path,
    nla_meta_path: str | Path,
) -> None:
    """
    Full AV SFT training loop.

    Args:
        config_path: Path to qwen7b_layer20.yaml.
        data_dir: Directory containing av_sft_train.parquet.
        nla_meta_path: Path to nla_meta_av.yaml.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    with open(nla_meta_path) as f:
        nla_meta = yaml.safe_load(f)

    av_cfg = cfg["av_sft"]
    seed = cfg["seed"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    injection_char = nla_meta["tokens"]["injection_char"]
    injection_token_id = nla_meta["tokens"]["injection_token_id"]
    left_neighbor_id = nla_meta["tokens"]["injection_left_neighbor_id"]
    right_neighbor_id = nla_meta["tokens"]["injection_right_neighbor_id"]
    injection_scale = nla_meta["extraction"]["injection_scale"]

    logger.info("Loading tokenizer + model: %s", cfg["verbalizer_model"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["verbalizer_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["verbalizer_model"],
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    device = next(base_model.parameters()).device
    logger.info("Model loaded on device: %s", device)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=av_cfg["lora_rank"],
        lora_alpha=av_cfg["lora_alpha"],
        lora_dropout=av_cfg["lora_dropout"],
        target_modules=av_cfg["target_modules"],
        bias="none",
    )
    lora_model = get_peft_model(base_model, lora_config)
    lora_model.print_trainable_parameters()

    if av_cfg.get("gradient_checkpointing"):
        lora_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    av_model = AVModelWrapper(
        model=lora_model,
        injection_token_id=injection_token_id,
        left_neighbor_id=left_neighbor_id,
        right_neighbor_id=right_neighbor_id,
        injection_scale=injection_scale,
    )

    dataset = AVDataset(
        parquet_path=Path(data_dir) / "av_sft_train.parquet",
        tokenizer=tokenizer,
        injection_char=injection_char,
        max_response_length=av_cfg.get("max_response_length", 150),
    )
    loader = DataLoader(
        dataset,
        batch_size=av_cfg["per_device_train_batch_size"],
        shuffle=True,
        num_workers=av_cfg["dataloader_num_workers"],
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(av_model.parameters(), lr=av_cfg["learning_rate"])
    total_steps = len(loader) * av_cfg["num_train_epochs"]
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=av_cfg["warmup_steps"],
        num_training_steps=total_steps,
    )

    output_dir = Path(av_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    accum_steps = av_cfg["gradient_accumulation_steps"]
    global_step = 0
    best_loss = float("inf")

    for epoch in range(av_cfg["num_train_epochs"]):
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"AV SFT epoch {epoch+1}", dynamic_ncols=True)

        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            activation_vectors = batch["activation_vector"].to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = av_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    activation_vectors=activation_vectors,
                )
                loss = outputs.loss / accum_steps

            loss.backward()
            epoch_loss += loss.item() * accum_steps

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(av_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                avg_loss = epoch_loss / (step + 1)
                pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

                if global_step % av_cfg["logging_steps"] == 0:
                    logger.info(
                        "Step %d | loss=%.4f | lr=%.2e",
                        global_step, avg_loss, scheduler.get_last_lr()[0],
                    )

                if global_step % av_cfg["save_steps"] == 0:
                    ckpt = output_dir / f"checkpoint-{global_step}"
                    lora_model.save_pretrained(ckpt)
                    logger.info("Saved checkpoint: %s", ckpt)

        avg_epoch_loss = epoch_loss / len(loader)
        logger.info("Epoch %d complete | avg_loss=%.4f", epoch + 1, avg_epoch_loss)

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            lora_model.save_pretrained(output_dir / "best")
            logger.info("New best checkpoint (loss=%.4f) → %s/best", best_loss, output_dir)

    lora_model.save_pretrained(output_dir / "final")
    av_model.remove_hooks()
    logger.info("AV SFT complete. Final model at %s/final", output_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AV SFT Training")
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
    train_av_sft(
        config_path=args.config,
        data_dir=args.data_dir,
        nla_meta_path=args.nla_meta,
    )


if __name__ == "__main__":
    main()

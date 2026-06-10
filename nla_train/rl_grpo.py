"""
RL GRPO: Joint AV + AR training via Group Relative Policy Optimization.

The AV model generates descriptions of activations; the AR model reconstructs
the activation from those descriptions. The reward is the negative normalized MSE
between the AR reconstruction and the gold activation:

    reward = -MSE(L2_norm(AR(description)), L2_norm(gold_activation))

Critically, the AR model is NOT frozen — it is updated every `ar_update_every_n_steps`
GRPO steps. A frozen AR would produce stale rewards that the AV model learns to game.
Joint training keeps the reward signal honest throughout.

Implementation uses TRL's GRPOTrainer with a custom reward function. The AR model
runs on the same GPU as the AV model (both LoRA-adapted, total ~25GB on H100).

Usage:
    python -m nla_train.rl_grpo \
        --config configs/qwen7b_layer20.yaml \
        --data-dir data/train \
        --nla-meta data/labeled/nla_meta_av.yaml \
        --av-checkpoint checkpoints/av_sft/final \
        --ar-checkpoint checkpoints/ar_sft/final
"""

from __future__ import annotations

import argparse
import gc
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import yaml
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from nla_train.ar_sft import TruncatedARModel, mse_loss_normalized
from nla_train.injection import AV_PROMPT_TEMPLATE, AR_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class RLDataset(Dataset):
    """
    Dataset for GRPO RL training.

    Each item: tokenized AV prompt (with injection placeholder) + gold activation.
    No explanation labels — the AV model generates those during rollouts.
    """

    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer: AutoTokenizer,
        injection_char: str,
        max_prompt_length: int = 300,
    ) -> None:
        table = pq.read_table(parquet_path)
        n = len(table)
        d = len(table.column("activation_vector")[0].as_py())
        act_col = table.column("activation_vector").combine_chunks()
        self._activations: np.ndarray = act_col.values.to_numpy(zero_copy_only=False).reshape(n, d)
        self._tokenizer = tokenizer
        self._injection_char = injection_char
        self._max_prompt_length = max_prompt_length

    def __len__(self) -> int:
        return len(self._activations)

    def __getitem__(self, idx: int) -> dict:
        activation = self._activations[idx].tolist()
        prompt_content = AV_PROMPT_TEMPLATE.format(injection_char=self._injection_char)
        messages = [{"role": "user", "content": prompt_content}]
        prompt: str = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return {
            "prompt": prompt,
            "activation_vector": activation,
        }


class ARRewardModel:
    """
    Wraps the AR model to compute reconstruction rewards for GRPO.

    Runs the AR model forward pass on (description → activation) pairs and
    returns -MSE(normalized_reconstruction, normalized_gold) as reward.
    """

    def __init__(
        self,
        ar_model: TruncatedARModel,
        tokenizer: AutoTokenizer,
        device: torch.device,
        max_length: int = 256,
    ) -> None:
        self._ar_model = ar_model
        self._tokenizer = tokenizer
        self._device = device
        self._max_length = max_length

    @torch.no_grad()
    def compute_rewards(
        self,
        completions: list[str],
        gold_activations: list[list[float]],
    ) -> list[float]:
        """
        Compute GRPO rewards for a batch of AV-generated completions.

        Batches all completions into a single AR forward pass for throughput.

        Args:
            completions: List of AV-generated description strings.
            gold_activations: List of gold activation vectors (d_model,) each.

        Returns:
            List of scalar rewards, one per completion.
        """
        prompts = [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": AR_PROMPT_TEMPLATE.format(explanation=c)}],
                tokenize=False,
                add_generation_prompt=False,
            )
            for c in completions
        ]
        enc = self._tokenizer(
            prompts,
            max_length=self._max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self._device)

        golds = torch.tensor(gold_activations, dtype=torch.float32, device=self._device)
        preds = self._ar_model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )
        pred_norm = torch.nn.functional.normalize(preds, dim=-1)
        gold_norm = torch.nn.functional.normalize(golds, dim=-1)
        per_sample_mse = ((pred_norm - gold_norm) ** 2).mean(dim=-1)
        return (-per_sample_mse).tolist()

    def update(
        self,
        completions: list[str],
        gold_activations: list[list[float]],
        n_steps: int = 1,
        lr: float = 1e-5,
    ) -> float:
        """
        Fine-tune the AR model on the current batch of (description, activation) pairs.

        Called every `ar_update_every_n_steps` GRPO steps to keep the reward model
        from going stale.

        Returns:
            Mean loss for this update.
        """
        self._ar_model.train()
        optimizer = torch.optim.AdamW(self._ar_model.parameters(), lr=lr)
        total_loss = 0.0

        prompts = [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": AR_PROMPT_TEMPLATE.format(explanation=c)}],
                tokenize=False,
                add_generation_prompt=False,
            )
            for c in completions
        ]
        enc = self._tokenizer(
            prompts,
            max_length=self._max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        ).to(self._device)
        golds = torch.tensor(gold_activations, dtype=torch.float32, device=self._device)

        for _ in range(n_steps):
            preds = self._ar_model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
            )
            loss = mse_loss_normalized(preds, golds)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()

        self._ar_model.eval()
        return total_loss / n_steps


def build_reward_fn(
    ar_reward_model: ARRewardModel,
    ar_update_every_n_steps: int,
    step_counter: dict,
    recent_completions: dict,
) -> callable:
    """
    Build the reward function for GRPOTrainer.

    TRL calls this as: reward_fn(completions, **kwargs) where kwargs includes
    any extra columns from the dataset. We store activation_vector in the dataset
    and pass it through via TRL's extra_columns mechanism.

    Args:
        ar_reward_model: ARRewardModel instance.
        ar_update_every_n_steps: Update AR every N GRPO steps.
        step_counter: Mutable dict with key "n" tracking global step count.
        recent_completions: Mutable dict storing recent (completions, activations)
            for AR update.

    Returns:
        Callable reward function compatible with TRL GRPOTrainer.
    """
    def reward_fn(completions: list[str], activation_vector: list, **kwargs) -> list[float]:
        gold_activations = [
            v if isinstance(v, list) else v.tolist()
            for v in activation_vector
        ]
        rewards = ar_reward_model.compute_rewards(completions, gold_activations)

        recent_completions["completions"] = completions
        recent_completions["activations"] = gold_activations
        step_counter["n"] += 1

        if step_counter["n"] % ar_update_every_n_steps == 0:
            ar_loss = ar_reward_model.update(
                completions=recent_completions["completions"],
                gold_activations=recent_completions["activations"],
            )
            logger.info(
                "AR update at step %d | ar_loss=%.4f",
                step_counter["n"], ar_loss,
            )

        return rewards

    return reward_fn


def train_rl_grpo(
    config_path: str,
    data_dir: str | Path,
    nla_meta_path: str | Path,
    av_checkpoint: str | Path,
    ar_checkpoint: str | Path,
) -> None:
    """
    Full RL GRPO training loop.

    Args:
        config_path: Path to qwen7b_layer20.yaml.
        data_dir: Directory containing rl_train.parquet.
        nla_meta_path: Path to nla_meta_av.yaml.
        av_checkpoint: Path to trained AV SFT checkpoint (LoRA).
        ar_checkpoint: Path to trained AR SFT checkpoint (LoRA).
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    with open(nla_meta_path) as f:
        nla_meta = yaml.safe_load(f)

    grpo_cfg = cfg["grpo"]
    seed = cfg["seed"]

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    injection_char = nla_meta["tokens"]["injection_char"]

    logger.info("Loading tokenizer: %s", cfg["verbalizer_model"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["verbalizer_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load AV model (LoRA from SFT checkpoint) ----
    logger.info("Loading AV model from: %s", av_checkpoint)
    av_base = AutoModelForCausalLM.from_pretrained(
        cfg["verbalizer_model"],
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    av_model = PeftModel.from_pretrained(av_base, av_checkpoint, is_trainable=True)

    # ---- Load AR model (LoRA from SFT checkpoint) ----
    logger.info("Loading AR model from: %s", ar_checkpoint)
    ar_base = AutoModelForCausalLM.from_pretrained(
        cfg["target_model"],
        dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    ar_truncated = TruncatedARModel(
        base_model=ar_base,
        target_layer=cfg["target_layer"],
        d_model=cfg["d_model"],
    )
    del ar_base
    gc.collect()

    ar_truncated = PeftModel.from_pretrained(ar_truncated, ar_checkpoint, is_trainable=True)
    ar_truncated.to(device)
    ar_truncated.eval()

    ar_reward_model = ARRewardModel(
        ar_model=ar_truncated,
        tokenizer=tokenizer,
        device=device,
    )

    step_counter: dict = {"n": 0}
    recent_completions: dict = {"completions": [], "activations": []}

    reward_fn = build_reward_fn(
        ar_reward_model=ar_reward_model,
        ar_update_every_n_steps=grpo_cfg["ar_update_every_n_steps"],
        step_counter=step_counter,
        recent_completions=recent_completions,
    )

    dataset = RLDataset(
        parquet_path=Path(data_dir) / "rl_train.parquet",
        tokenizer=tokenizer,
        injection_char=injection_char,
    )

    output_dir = Path(grpo_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    import inspect as _inspect
    _grpo_params = set(_inspect.signature(GRPOConfig.__init__).parameters)

    _grpo_kwargs: dict = dict(
        output_dir=str(output_dir),
        num_train_epochs=grpo_cfg["num_train_epochs"],
        per_device_train_batch_size=grpo_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=grpo_cfg["gradient_accumulation_steps"],
        learning_rate=grpo_cfg["learning_rate"],
        bf16=grpo_cfg["bf16"],
        gradient_checkpointing=grpo_cfg.get("gradient_checkpointing", True),
        max_completion_length=grpo_cfg["max_completion_length"],
        save_steps=grpo_cfg["save_steps"],
        logging_steps=grpo_cfg["logging_steps"],
        seed=seed,
        report_to="none",
    )
    # num_generations vs num_sample_generations across TRL versions
    if "num_generations" in _grpo_params:
        _grpo_kwargs["num_generations"] = grpo_cfg["num_generations"]
    elif "num_sample_generations" in _grpo_params:
        _grpo_kwargs["num_sample_generations"] = grpo_cfg["num_generations"]
    # kl_coef renamed to beta in TRL >= 0.9
    if "beta" in _grpo_params:
        _grpo_kwargs["beta"] = grpo_cfg["kl_coef"]
    elif "kl_coef" in _grpo_params:
        _grpo_kwargs["kl_coef"] = grpo_cfg["kl_coef"]

    grpo_training_args = GRPOConfig(**_grpo_kwargs)

    trainer = GRPOTrainer(
        model=av_model,
        args=grpo_training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        reward_funcs=[reward_fn],
    )

    logger.info("Starting GRPO training")
    trainer.train()

    av_model.save_pretrained(output_dir / "final_av")
    ar_truncated.save_pretrained(output_dir / "final_ar")
    logger.info("GRPO complete. Final models at %s", output_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RL GRPO Training")
    p.add_argument("--config", default="configs/qwen7b_layer20.yaml")
    p.add_argument("--data-dir", default="data/train")
    p.add_argument("--nla-meta", default="data/labeled/nla_meta_av.yaml")
    p.add_argument("--av-checkpoint", default="checkpoints/av_sft/final")
    p.add_argument("--ar-checkpoint", default="checkpoints/ar_sft/final")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    train_rl_grpo(
        config_path=args.config,
        data_dir=args.data_dir,
        nla_meta_path=args.nla_meta,
        av_checkpoint=args.av_checkpoint,
        ar_checkpoint=args.ar_checkpoint,
    )


if __name__ == "__main__":
    main()

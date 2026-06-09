"""
Stage 3: Pack final training datasets.

Joins the raw activation splits (ar_sft, rl) with the labeled split (av_sft)
to produce three final training parquets:

  av_sft_train.parquet  — activation + gold explanation → AV SFT training
  ar_sft_train.parquet  — explanation + activation → AR SFT training
  rl_train.parquet      — activation only → GRPO RL training (no labels needed)

The AR SFT parquet is built by cross-referencing the same explanations produced
in Stage 2. Specifically, for each (activation, explanation) pair in av_sft_labeled,
we also add it to the AR training set (with the explanation as input and the
activation as the reconstruction target). The ar_sft split from Stage 1 provides
additional activation-only rows that are labeled on-the-fly in this stage using
the labeled av_sft as a reference distribution (or omitted if ar_sft explanations
were not generated).

Usage:
    python -m nla_train.datagen.stage3_pack \
        --config configs/qwen7b_layer20.yaml \
        --split-dir data/split \
        --labeled-dir data/labeled \
        --output-dir data/train
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

logger = logging.getLogger(__name__)

AV_TRAIN_SCHEMA = pa.schema([
    pa.field("doc_id", pa.string()),
    pa.field("text_snippet", pa.string()),
    pa.field("activation_vector", pa.list_(pa.float32())),
    pa.field("explanation", pa.string()),
    pa.field("layer", pa.int32()),
])

AR_TRAIN_SCHEMA = pa.schema([
    pa.field("doc_id", pa.string()),
    pa.field("text_snippet", pa.string()),
    pa.field("activation_vector", pa.list_(pa.float32())),
    pa.field("explanation", pa.string()),
    pa.field("layer", pa.int32()),
])

RL_TRAIN_SCHEMA = pa.schema([
    pa.field("doc_id", pa.string()),
    pa.field("text_snippet", pa.string()),
    pa.field("activation_vector", pa.list_(pa.float32())),
    pa.field("layer", pa.int32()),
])


def pack_datasets(
    split_dir: str | Path,
    labeled_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, int]:
    """
    Pack the three final training datasets from split + labeled parquets.

    Args:
        split_dir: Directory containing av_sft.parquet, ar_sft.parquet, rl.parquet
            from Stage 1.
        labeled_dir: Directory containing av_sft_labeled.parquet from Stage 2.
        output_dir: Directory to write final training parquets.

    Returns:
        Dict mapping dataset name → row count.
    """
    split_dir = Path(split_dir)
    labeled_dir = Path(labeled_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labeled = pq.read_table(labeled_dir / "av_sft_labeled.parquet")
    logger.info("Loaded av_sft_labeled: %d rows", len(labeled))

    # ---- AV SFT training data ----
    # Direct passthrough: activation + explanation pairs from Stage 2 labels.
    av_table = pa.table(
        {
            "doc_id": labeled.column("doc_id"),
            "text_snippet": labeled.column("text_snippet"),
            "activation_vector": labeled.column("activation_vector"),
            "explanation": labeled.column("explanation"),
            "layer": labeled.column("layer"),
        },
        schema=AV_TRAIN_SCHEMA,
    )
    av_path = output_dir / "av_sft_train.parquet"
    pq.write_table(av_table, av_path, compression="zstd")
    logger.info("av_sft_train: %d rows → %s", len(av_table), av_path)

    # ---- AR SFT training data ----
    # Same labeled pairs but used in the reverse direction:
    # input = explanation → output = activation reconstruction.
    # We also include the raw ar_sft split rows if they exist and have explanations.
    ar_table = pa.table(
        {
            "doc_id": labeled.column("doc_id"),
            "text_snippet": labeled.column("text_snippet"),
            "activation_vector": labeled.column("activation_vector"),
            "explanation": labeled.column("explanation"),
            "layer": labeled.column("layer"),
        },
        schema=AR_TRAIN_SCHEMA,
    )

    ar_labeled_path = labeled_dir / "ar_sft_labeled.parquet"
    if ar_labeled_path.exists():
        ar_extra = pq.read_table(ar_labeled_path)
        logger.info("Found ar_sft_labeled: %d extra rows", len(ar_extra))
        ar_extra_table = pa.table(
            {
                "doc_id": ar_extra.column("doc_id"),
                "text_snippet": ar_extra.column("text_snippet"),
                "activation_vector": ar_extra.column("activation_vector"),
                "explanation": ar_extra.column("explanation"),
                "layer": ar_extra.column("layer"),
            },
            schema=AR_TRAIN_SCHEMA,
        )
        ar_table = pa.concat_tables([ar_table, ar_extra_table])

    ar_path = output_dir / "ar_sft_train.parquet"
    pq.write_table(ar_table, ar_path, compression="zstd")
    logger.info("ar_sft_train: %d rows → %s", len(ar_table), ar_path)

    # ---- RL training data ----
    # No explanations needed — just activations. The AV model generates
    # descriptions on-the-fly during GRPO rollouts.
    rl_raw = pq.read_table(split_dir / "rl.parquet")
    rl_table = pa.table(
        {
            "doc_id": rl_raw.column("doc_id"),
            "text_snippet": rl_raw.column("text_snippet"),
            "activation_vector": rl_raw.column("activation_vector"),
            "layer": rl_raw.column("layer"),
        },
        schema=RL_TRAIN_SCHEMA,
    )
    rl_path = output_dir / "rl_train.parquet"
    pq.write_table(rl_table, rl_path, compression="zstd")
    logger.info("rl_train: %d rows → %s", len(rl_table), rl_path)

    return {
        "av_sft_train": len(av_table),
        "ar_sft_train": len(ar_table),
        "rl_train": len(rl_table),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 3: Pack final training datasets")
    p.add_argument("--config", default="configs/qwen7b_layer20.yaml")
    p.add_argument("--split-dir", default=None)
    p.add_argument("--labeled-dir", default=None)
    p.add_argument("--output-dir", default=None)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dg = cfg["datagen"]
    base = Path(dg["output_dir"]).parent

    split_dir = args.split_dir or str(base / "split")
    labeled_dir = args.labeled_dir or cfg["labeling"]["output_dir"]
    output_dir = args.output_dir or str(base / "train")

    counts = pack_datasets(
        split_dir=split_dir,
        labeled_dir=labeled_dir,
        output_dir=output_dir,
    )
    logger.info("Stage 3 complete: %s", counts)


if __name__ == "__main__":
    main()

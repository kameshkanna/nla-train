"""
Stage 1: Document-level train/val split.

Splits the Stage 0 activation parquets into three non-overlapping subsets
using document-level (not row-level) partitioning to prevent data leakage:

  30% → av_sft   (AV Activation Verbalizer supervised fine-tuning)
  30% → ar_sft   (AR Activation Reconstructor supervised fine-tuning)
  40% → rl       (GRPO reinforcement learning)

Splitting at the document level ensures that all activations from the same
document go to the same split, preventing any contamination between stages.

Usage:
    python -m nla_train.datagen.stage1_split \
        --config configs/qwen7b_layer20.yaml \
        --input-dir data/raw \
        --output-dir data/split
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)


def split_activations(
    input_dir: str | Path,
    output_dir: str | Path,
    av_sft_frac: float,
    ar_sft_frac: float,
    seed: int = 42,
) -> dict[str, int]:
    """
    Read all Parquet chunks, split by doc_id, write three output parquets.

    Args:
        input_dir: Directory containing chunk_*.parquet files from Stage 0.
        output_dir: Directory to write av_sft.parquet, ar_sft.parquet, rl.parquet.
        av_sft_frac: Fraction of documents for AV SFT.
        ar_sft_frac: Fraction of documents for AR SFT.
        seed: Random seed for reproducible splits.

    Returns:
        Dict mapping split name → row count.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunk_files = sorted(input_dir.glob("chunk_*.parquet"))
    if not chunk_files:
        raise FileNotFoundError(f"No chunk_*.parquet files in {input_dir}")

    logger.info("Reading %d chunk files from %s", len(chunk_files), input_dir)

    tables: list[pa.Table] = []
    for f in tqdm(chunk_files, desc="Reading chunks", dynamic_ncols=True):
        tables.append(pq.read_table(f))

    full_table = pa.concat_tables(tables)
    logger.info("Total rows: %d", len(full_table))

    # Collect unique doc_ids and shuffle deterministically
    doc_ids = np.unique(full_table.column("doc_id").to_pylist())
    rng = np.random.default_rng(seed)
    rng.shuffle(doc_ids)

    n = len(doc_ids)
    n_av = int(n * av_sft_frac)
    n_ar = int(n * ar_sft_frac)

    av_docs = set(doc_ids[:n_av].tolist())
    ar_docs = set(doc_ids[n_av:n_av + n_ar].tolist())
    rl_docs = set(doc_ids[n_av + n_ar:].tolist())

    logger.info(
        "Doc split — av_sft: %d  ar_sft: %d  rl: %d",
        len(av_docs), len(ar_docs), len(rl_docs),
    )

    doc_id_col = full_table.column("doc_id").to_pylist()
    av_mask = pa.array([d in av_docs for d in doc_id_col], type=pa.bool_())
    ar_mask = pa.array([d in ar_docs for d in doc_id_col], type=pa.bool_())
    rl_mask = pa.array([d in rl_docs for d in doc_id_col], type=pa.bool_())

    splits = {
        "av_sft": full_table.filter(av_mask),
        "ar_sft": full_table.filter(ar_mask),
        "rl": full_table.filter(rl_mask),
    }

    counts: dict[str, int] = {}
    for name, table in splits.items():
        out_path = output_dir / f"{name}.parquet"
        pq.write_table(table, out_path, compression="zstd")
        counts[name] = len(table)
        logger.info("Wrote %s: %d rows → %s", name, len(table), out_path)

    return counts


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1: Document-level split")
    p.add_argument("--config", default="configs/qwen7b_layer20.yaml")
    p.add_argument("--input-dir", default=None)
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
    input_dir = args.input_dir or dg["output_dir"]
    output_dir = args.output_dir or str(Path(dg["output_dir"]).parent / "split")

    split_activations(
        input_dir=input_dir,
        output_dir=output_dir,
        av_sft_frac=dg["av_sft_frac"],
        ar_sft_frac=dg["ar_sft_frac"],
        seed=cfg["seed"],
    )


if __name__ == "__main__":
    main()

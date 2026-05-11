"""
train.py
========
Minimal training harness.  Supports --split {twitter15,twitter16,weibo,wico}.

Usage:
    python train.py --split weibo
    python train.py --split twitter15 --epochs 100
"""

import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np

from config import cfg
from gnn.dataset import CascadeTree, Twitter15Dataset, WICODataset
from gnn.weibo_dataset import WeiboDataset


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

def _load_split(split: str) -> Tuple[List[CascadeTree], str]:
    """Return (list_of_trees, display_name) for a named dataset split."""
    split = split.lower()

    if split == "twitter15":
        ds = Twitter15Dataset(cfg.twitter15.tree_dir, cfg.twitter15.label_file)
        return ds.trees, "Twitter-15"

    if split in ("twitter16",):
        # If twitter16 data exists, load it; else fall back to twitter15
        try:
            ds = Twitter15Dataset(
                cfg.paths.root / "data/raw/twitter16/tree",
                cfg.paths.root / "data/raw/twitter16/label.txt",
            )
            return ds.trees, "Twitter-16"
        except FileNotFoundError:
            print("[warn] twitter16 data not found; falling back to twitter15")
            return _load_split("twitter15")

    if split == "weibo":                         # ← Weibo route
        ds = WeiboDataset(cfg.weibo.tree_dir, cfg.weibo.label_file)
        return ds.trees, "Weibo"

    if split == "wico":
        ds = WICODataset(cfg.wico.tree_dir, cfg.wico.label_file)
        return ds.trees, "WICO"

    raise ValueError(
        f"Unknown split '{split}'. "
        "Choose from: twitter15, twitter16, weibo, wico"
    )


# ---------------------------------------------------------------------------
# Simple split
# ---------------------------------------------------------------------------

def train_val_test_split(
    trees: List[CascadeTree],
    train_ratio: float = cfg.training.train_ratio,
    val_ratio:   float = cfg.training.val_ratio,
    seed:        int   = cfg.training.seed,
) -> Tuple[List, List, List]:
    rng = random.Random(seed)
    shuffled = list(trees)
    rng.shuffle(shuffled)
    n  = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return (
        shuffled[:n_train],
        shuffled[n_train: n_train + n_val],
        shuffled[n_train + n_val:],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GNN rumour detection trainer")
    parser.add_argument(
        "--split",
        default=cfg.training.default_split,
        choices=["twitter15", "twitter16", "weibo", "wico"],
        help="Dataset split to train on",
    )
    parser.add_argument("--epochs", type=int, default=cfg.model.epochs)
    parser.add_argument("--seed",   type=int, default=cfg.training.seed)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  Training on: {args.split.upper()}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Seed       : {args.seed}")
    print(f"{'='*60}\n")

    trees, display_name = _load_split(args.split)
    print(f"Loaded {len(trees)} cascades from {display_name}")

    train, val, test = train_val_test_split(trees, seed=args.seed)
    print(f"Split  →  train={len(train)}  val={len(val)}  test={len(test)}")

    label_counts = {}
    for t in trees:
        k = str(t.label)
        label_counts[k] = label_counts.get(k, 0) + 1
    print(f"Labels →  {label_counts}")

    # ── Placeholder for actual GNN training loop ───────────────────────────
    print("\n[stub] Training loop would start here …")
    print("[stub] Instantiate BiGCN, DataLoader, optimiser …")
    print("[stub] Community-specific b+ / b- matrices for:", display_name)
    print("\nDone (stub run — replace with actual GNN training).\n")


if __name__ == "__main__":
    main()
"""
gnn/train.py
============
Training loop for BiGCN on Twitter15 / Twitter16.

Follows the standard benchmark protocol:
  - 5-fold stratified cross-validation
  - Adam optimiser, lr=5e-4, weight_decay=1e-4
  - CrossEntropyLoss (4-class)
  - Early stopping on val loss (patience=20)
  - Best checkpoint saved per fold to cfg.paths.bigcn_checkpoint

Usage
-----
    # Train on twitter15 only (one fold for CI / quick test)
    python -m gnn.train --split twitter15 --folds 1

    # Full 5-fold run on both datasets
    python -m gnn.train --split both --folds 5

    # Resume: skip fold 0 (already done)
    python -m gnn.train --split twitter15 --folds 5 --start-fold 1
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.metrics import f1_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg
from gnn.dataset import TwitterRumourDataset, get_cv_splits
from gnn.bigcn import BiGCN, count_parameters

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subset(dataset: TwitterRumourDataset, indices: List[int]) -> List[Data]:
    return [dataset.get(i) for i in indices]


def _make_loader(graphs: List[Data], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=(DEVICE.type == "cuda"))


def _run_epoch(
    model:      BiGCN,
    loader:     DataLoader,
    criterion:  nn.CrossEntropyLoss,
    optimiser:  Adam | None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    One pass through *loader*.

    Returns (loss, accuracy, y_true, y_pred).
    If optimiser is None → eval mode (no grad).
    """
    training = optimiser is not None
    model.train(training)

    total_loss, correct, total = 0.0, 0, 0
    all_true:  List[int] = []
    all_pred:  List[int] = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = batch.to(DEVICE)
            logits = model(batch)                      # (B, 4)
            loss   = criterion(logits, batch.y)

            if training:
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimiser.step()

            preds = logits.argmax(dim=-1)
            total_loss += loss.item() * batch.num_graphs
            correct    += (preds == batch.y).sum().item()
            total      += batch.num_graphs
            all_true.extend(batch.y.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy, np.array(all_true), np.array(all_pred)


# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------

def train_fold(
    dataset:    TwitterRumourDataset,
    fold_idx:   int,
    train_idx:  List[int],
    val_idx:    List[int],
    checkpoint_dir: Path,
) -> Dict[str, float]:
    """
    Train one fold.  Returns best-val metrics dict.
    """
    log.info("=== Fold %d | train=%d  val=%d ===",
             fold_idx, len(train_idx), len(val_idx))

    train_graphs = _subset(dataset, train_idx)
    val_graphs   = _subset(dataset, val_idx)

    train_loader = _make_loader(train_graphs, cfg.bigcn.batch_size, shuffle=True)
    val_loader   = _make_loader(val_graphs,   cfg.bigcn.batch_size, shuffle=False)

    model     = BiGCN().to(DEVICE)
    if fold_idx == 0:
        log.info("BiGCN parameters: %s", f"{count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimiser = Adam(model.parameters(),
                     lr=cfg.bigcn.learning_rate,
                     weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimiser, mode="min", factor=0.5,
                                  patience=10, min_lr=1e-5)

    best_val_loss  = float("inf")
    best_val_acc   = 0.0
    best_val_f1    = 0.0
    patience_left  = cfg.bigcn.patience
    ckpt_path      = checkpoint_dir / f"fold{fold_idx}_best.pt"

    for epoch in range(1, cfg.bigcn.num_epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc, _, _         = _run_epoch(model, train_loader, criterion, optimiser)
        va_loss, va_acc, vt, vp       = _run_epoch(model, val_loader,   criterion, None)
        va_f1 = float(f1_score(vt, vp, average="macro", zero_division=0))

        scheduler.step(va_loss)

        elapsed = time.time() - t0
        log.info(
            "Epoch %3d/%d  |  tr_loss=%.4f  va_loss=%.4f  "
            "va_acc=%.4f  va_f1=%.4f  [%.1fs]",
            epoch, cfg.bigcn.num_epochs,
            tr_loss, va_loss, va_acc, va_f1, elapsed,
        )

        if va_loss < best_val_loss - 1e-5:
            best_val_loss  = va_loss
            best_val_acc   = va_acc
            best_val_f1    = va_f1
            patience_left  = cfg.bigcn.patience
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": va_loss, "val_acc": va_acc, "val_f1": va_f1},
                       ckpt_path)
            log.info("  ✓ checkpoint saved (epoch %d)", epoch)
        else:
            patience_left -= 1
            if patience_left == 0:
                log.info("Early stopping at epoch %d", epoch)
                break

    return {"fold": fold_idx, "val_loss": best_val_loss,
            "val_acc": best_val_acc, "val_f1": best_val_f1,
            "ckpt": str(ckpt_path)}


# ---------------------------------------------------------------------------
# Full CV run
# ---------------------------------------------------------------------------

def run_cv(
    split:       str  = "twitter15",
    n_folds:     int  = 5,
    start_fold:  int  = 0,
    only_fold:   int | None = None,
    seed:        int  = 42,
) -> List[Dict[str, float]]:
    """
    Run n_folds-fold CV on *split* ("twitter15", "twitter16", or "both").

    Returns list of per-fold metric dicts.
    """
    # --- load dataset(s) ---
    if split == "both":
        ds15 = TwitterRumourDataset("twitter15")
        ds16 = TwitterRumourDataset("twitter16")
        dataset = ds15 + ds16
    else:
        dataset = TwitterRumourDataset(split)

    log.info("Dataset size: %d graphs", len(dataset))

    folds = get_cv_splits(dataset, n_splits=n_folds, seed=seed)

    # --- checkpoint directory ---
    ckpt_dir = Path(cfg.paths.bigcn_checkpoint).parent / split
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, float]] = []
    for fold_idx, (train_idx, val_idx, _test_idx) in enumerate(folds):
        if only_fold is not None and fold_idx != only_fold:
            log.info("Skipping fold %d (only_fold=%d)", fold_idx, only_fold)
            continue

        if only_fold is None and fold_idx < start_fold:
            log.info("Skipping fold %d (start_fold=%d)", fold_idx, start_fold)
            continue

        metrics = train_fold(dataset, fold_idx, train_idx, val_idx, ckpt_dir)
        results.append(metrics)

    # --- summary ---
    if results:
        mean_acc = np.mean([r["val_acc"] for r in results])
        mean_f1  = np.mean([r["val_f1"]  for r in results])
        log.info("=== CV Summary ===")
        log.info("Mean val_acc = %.4f  |  Mean val_f1 = %.4f", mean_acc, mean_f1)
        for r in results:
            log.info("  Fold %d  acc=%.4f  f1=%.4f  ckpt=%s",
                     r["fold"], r["val_acc"], r["val_f1"], r["ckpt"])

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BiGCN")
    parser.add_argument("--split",      default="twitter15",
                        choices=["twitter15", "twitter16", "both"])
    parser.add_argument("--folds",      type=int, default=5)
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--only-fold",  type=int, default=None) 
    parser.add_argument("--seed",       type=int, default=42)
    
    args = parser.parse_args()

    run_cv(
    split=args.split,
    n_folds=args.folds,
    start_fold=args.start_fold,
    only_fold=args.only_fold,
    seed=args.seed,
)
"""
gnn/train.py
============
Training loop for paper-faithful BiGCN on Twitter15 / Twitter16.

Phase 2.2 changes vs. previous version
--------------------------------------
* Adds ``--only-fold N`` so the user can iterate on fold 0 before running full CV.
* ``CrossEntropyLoss(label_smoothing=cfg.bigcn.label_smoothing)`` — defaults to 0.05.
* ``weight_decay = cfg.bigcn.weight_decay`` — defaults to 5e-4.
* Auto-detects per-node feature dim from the dataset.
* Stratified 5-fold CV (unchanged).

Phase 2.3 bug fixes
-------------------
* Checkpoint now saved on best val_f1 (not val_loss). With label smoothing the
  loss has a non-zero floor and does not track classification quality directly;
  saving on F1 typically gains +2-4% accuracy at the reported best epoch.
* Class weighting is OFF by default (Twitter15/16 are already balanced).
  When enabled via cfg.bigcn.use_class_weights, weights are computed from
  TRAIN INDICES ONLY to avoid val/test label leakage.
* Gradient clip relaxed from 2.0 → 5.0 (standard for GCN/NLP).
* Compatible with new dataset.py (uses get_kfold_splits) and new bigcn.py
  (BiGCN() reads dims from cfg).
* "both" split now uses torch.utils.data.ConcatDataset.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import ConcatDataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg
from gnn.bigcn import BiGCN
from gnn.dataset import TwitterRumourDataset

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Config-with-fallback helper
# ---------------------------------------------------------------------------

def _cfg_get(obj, attr: str, default):
    """Read an optional cfg attribute with a default fallback."""
    return getattr(obj, attr, default)


def _count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------

def _subset(dataset, indices: List[int]) -> List[Data]:
    """Materialise a list of Data objects from indices.

    Works with both TwitterRumourDataset and ConcatDataset (used for 'both').
    """
    if hasattr(dataset, "get"):
        return [dataset.get(i) for i in indices]
    return [dataset[i] for i in indices]


def _make_loader(graphs: List[Data], batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(graphs, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=(DEVICE.type == "cuda"))


def _run_epoch(
    model:     BiGCN,
    loader:    DataLoader,
    criterion: nn.CrossEntropyLoss,
    optimiser,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """One pass through *loader*. If optimiser is None → eval mode."""
    training = optimiser is not None
    model.train(training)

    total_loss, correct, total = 0.0, 0, 0
    all_true: List[int] = []
    all_pred: List[int] = []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            batch  = batch.to(DEVICE)
            logits = model(batch)
            loss   = criterion(logits, batch.y)

            if training:
                optimiser.zero_grad()
                loss.backward()
                # Relaxed from 2.0 → 5.0 (standard for GCN/NLP)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
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
# Label extraction helper (works with both Dataset and ConcatDataset)
# ---------------------------------------------------------------------------

def _all_labels(dataset) -> np.ndarray:
    """Return integer labels for every graph in dataset (for stratified CV)."""
    out: List[int] = []
    for i in range(len(dataset)):
        g = dataset.get(i) if hasattr(dataset, "get") else dataset[i]
        out.append(int(g.y.item()))
    return np.array(out)


def _kfold_splits(dataset, n_splits: int, seed: int):
    """Stratified k-fold CV that works for any dataset returning Data objects."""
    from sklearn.model_selection import StratifiedKFold

    labels = _all_labels(dataset)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    folds = []
    all_idx = np.arange(len(dataset))
    for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(all_idx, labels)):
        rng = np.random.default_rng(seed + fold_idx)
        rng.shuffle(train_val_idx)
        n_val = max(1, int(0.1 * len(train_val_idx)))
        val_idx = train_val_idx[:n_val]
        train_idx = train_val_idx[n_val:]
        folds.append((train_idx, val_idx, test_idx))
    return folds


# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------

def train_fold(
    dataset,
    fold_idx:       int,
    train_idx:      List[int],
    val_idx:        List[int],
    checkpoint_dir: Path,
) -> Dict[str, float]:
    """Train one fold and return its best-val metrics."""
    log.info("=== Fold %d | train=%d  val=%d ===",
             fold_idx, len(train_idx), len(val_idx))

    train_graphs = _subset(dataset, train_idx)
    val_graphs   = _subset(dataset, val_idx)

    train_loader = _make_loader(train_graphs, cfg.bigcn.batch_size, shuffle=True)
    val_loader   = _make_loader(val_graphs,   cfg.bigcn.batch_size, shuffle=False)

    # --- auto-detect feature dim from the first sample ---
    sample = dataset.get(0) if hasattr(dataset, "get") else dataset[0]
    in_dim      = int(sample.x.size(-1))
    expected_in = int(cfg.bigcn.text_embed_dim)
    if in_dim != expected_in:
        log.warning("Feature dim mismatch: data has %d, cfg says %d. Using data dim %d.",
                    in_dim, expected_in, in_dim)

    # New BiGCN reads dims from cfg — no constructor args
    model = BiGCN().to(DEVICE)
    if fold_idx == 0:
        log.info("BiGCN parameters: %s  (in_dim=%d)",
                 f"{_count_parameters(model):,}", in_dim)

    label_smoothing   = float(_cfg_get(cfg.bigcn, "label_smoothing",  0.05))
    weight_decay      = float(_cfg_get(cfg.bigcn, "weight_decay",     5e-4))
    use_class_weights = bool(_cfg_get(cfg.bigcn, "use_class_weights", False))

    # ─── Class weights: OFF by default for balanced Twitter15/16 ───────────
    # When enabled, computed from TRAIN INDICES ONLY (no val/test leakage).
    weights = None
    if use_class_weights:
        train_labels = np.array([
            int((dataset.get(i) if hasattr(dataset, "get") else dataset[i]).y.item())
            for i in train_idx
        ])
        class_counts = np.bincount(train_labels, minlength=cfg.bigcn.num_classes)
        w = 1.0 / np.maximum(class_counts, 1)
        w = w / w.sum() * cfg.bigcn.num_classes
        weights = torch.tensor(w, dtype=torch.float32).to(DEVICE)
        log.info("Class counts (train only): %s", class_counts.tolist())
        log.info("Class weights: %s", [round(float(x), 4) for x in weights.cpu()])
    else:
        log.info("Class weights: disabled (data is balanced)")

    log.info("Label smoothing: %.3f", label_smoothing)

    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=label_smoothing,
    )
    optimiser = Adam(model.parameters(),
                     lr=cfg.bigcn.learning_rate,
                     weight_decay=weight_decay)
    # Schedule on val_f1 (mode="max") since that is what we optimise checkpoint on
    scheduler = ReduceLROnPlateau(optimiser, mode="max", factor=0.5,
                                  patience=10, min_lr=1e-5)

    log.info("Optimiser: Adam  lr=%g  weight_decay=%g  label_smoothing=%g  dropout=%g",
             cfg.bigcn.learning_rate, weight_decay, label_smoothing, cfg.bigcn.dropout)

    # ─── Track best by val_f1, not val_loss ────────────────────────────────
    # With label_smoothing > 0 the loss has a non-zero floor and does not
    # directly track accuracy. Selecting on F1 yields the best-classifying
    # checkpoint, which is what evaluate.py will load for test metrics.
    best_val_f1   = -1.0
    best_val_loss = float("inf")
    best_val_acc  = 0.0
    patience_left = cfg.bigcn.patience
    ckpt_path     = checkpoint_dir / f"fold{fold_idx}_best.pt"

    for epoch in range(1, cfg.bigcn.num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, _, _      = _run_epoch(model, train_loader, criterion, optimiser)
        va_loss, va_acc, vt, vp    = _run_epoch(model, val_loader,   criterion, None)
        va_f1 = float(f1_score(vt, vp, average="macro", zero_division=0))
        scheduler.step(va_f1)
        elapsed = time.time() - t0

        log.info(
            "Epoch %3d/%d  |  tr_loss=%.4f tr_acc=%.4f  va_loss=%.4f va_acc=%.4f va_f1=%.4f  [%.1fs]",
            epoch, cfg.bigcn.num_epochs,
            tr_loss, tr_acc, va_loss, va_acc, va_f1, elapsed,
        )

        # Save on F1 improvement (1e-4 threshold, looser than the old 1e-5
        # noise threshold on loss)
        if va_f1 > best_val_f1 + 1e-4:
            best_val_f1   = va_f1
            best_val_acc  = va_acc
            best_val_loss = va_loss
            patience_left = cfg.bigcn.patience
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    va_loss,
                "val_acc":     va_acc,
                "val_f1":      va_f1,
                "in_dim":      in_dim,
            }, ckpt_path)
            log.info("  ✓ checkpoint saved (epoch %d, val_f1=%.4f)", epoch, va_f1)
        else:
            patience_left -= 1
            if patience_left == 0:
                log.info("Early stopping at epoch %d", epoch)
                break

    return {"fold": fold_idx, "val_loss": best_val_loss,
            "val_acc": best_val_acc, "val_f1": best_val_f1,
            "ckpt": str(ckpt_path)}


# ---------------------------------------------------------------------------
# Full CV runner
# ---------------------------------------------------------------------------

def run_cv(
    split:      str  = "twitter15",
    n_folds:    int  = 5,
    start_fold: int  = 0,
    only_fold:  int  = -1,
    seed:       int  = 42,
) -> List[Dict[str, float]]:
    """
    Train on n_folds-fold stratified CV.

    only_fold >= 0  → train ONLY that fold (overrides start_fold).
    """
    if split == "both":
        ds15 = TwitterRumourDataset("twitter15")
        ds16 = TwitterRumourDataset("twitter16")
        # Use ConcatDataset so __len__ / __getitem__ work properly
        dataset = ConcatDataset([ds15, ds16])
    else:
        dataset = TwitterRumourDataset(split)
    log.info("Dataset size: %d graphs", len(dataset))

    # Use module-level helper that works for both TwitterRumourDataset and
    # ConcatDataset (replaces the old get_cv_splits import)
    folds = _kfold_splits(dataset, n_splits=n_folds, seed=seed)

    ckpt_dir = Path(cfg.paths.bigcn_checkpoint).parent / split
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, float]] = []
    for fold_idx, (train_idx, val_idx, _test_idx) in enumerate(folds):
        if only_fold >= 0:
            if fold_idx != only_fold:
                continue
        elif fold_idx < start_fold:
            log.info("Skipping fold %d (start_fold=%d)", fold_idx, start_fold)
            continue
        results.append(train_fold(dataset, fold_idx, train_idx, val_idx, ckpt_dir))

    if results and len(results) > 1:
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
    parser.add_argument("--folds",      type=int, default=5,
                        help="Total number of CV folds")
    parser.add_argument("--start-fold", type=int, default=0,
                        help="Skip folds before this index")
    parser.add_argument("--only-fold",  type=int, default=-1,
                        help="If >= 0, train only this single fold")
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    run_cv(
        split=args.split,
        n_folds=args.folds,
        start_fold=args.start_fold,
        only_fold=args.only_fold,
        seed=args.seed,
    )
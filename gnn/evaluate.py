"""
gnn/evaluate.py
===============
Evaluate a trained BiGCN on the held-out test fold.

Metrics reported
----------------
4-class (Twitter15 protocol):
  • Accuracy
  • Macro F1
  • Per-class F1 (true / false / unverified / non-rumor)
  • Confusion matrix

Binary (true vs false, for SBM pipeline):
  • Accuracy, F1 — ignoring "unverified" and "non-rumor" samples

Usage
-----
    # Evaluate fold 0 on twitter15
    python -m gnn.evaluate --split twitter15 --fold 0

    # Evaluate all folds and average
    python -m gnn.evaluate --split twitter15 --all-folds
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg
from gnn.dataset import TwitterRumourDataset, get_cv_splits
from gnn.bigcn import BiGCN

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                    datefmt="%H:%M:%S")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names for Twitter15/16 (4-class)
CLASS_NAMES = ["true", "false", "unverified", "non-rumor"]

# Binary mapping: which integer labels count as "false" (1) for SBM pipeline
# cfg.twitter15.binary_label_map: {0:"true", 1:"false", 2:"uncertain", 3:"true"}
_BINARY_MAP = {0: 0, 1: 1, 2: -1, 3: 0}   # -1 = ignore


def _load_checkpoint(ckpt_path: Path) -> Dict:
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    return ckpt


def _predict(model: BiGCN, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (y_true, y_pred, y_prob) where y_prob is shape (N, num_classes).
    """
    model.eval()
    all_true: List[int]          = []
    all_pred: List[int]          = []
    all_prob: List[np.ndarray]   = []

    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(DEVICE)
            logits = model(batch)                           # (B, 4)
            probs  = torch.softmax(logits, dim=-1)
            preds  = logits.argmax(dim=-1)
            all_true.extend(batch.y.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
            all_prob.append(probs.cpu().numpy())

    return (np.array(all_true),
            np.array(all_pred),
            np.vstack(all_prob))


def evaluate_fold(
    dataset:   TwitterRumourDataset,
    fold_idx:  int,
    test_idx:  List[int],
    ckpt_dir:  Path,
    verbose:   bool = True,
) -> Dict[str, float]:
    """
    Evaluate one fold.  Returns metric dict.
    """
    ckpt_path = ckpt_dir / f"fold{fold_idx}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run gnn/train.py first."
        )

    ckpt  = _load_checkpoint(ckpt_path)
    model = BiGCN().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    log.info("Loaded fold %d checkpoint (epoch %d, val_acc=%.4f)",
             fold_idx, ckpt.get("epoch", -1), ckpt.get("val_acc", float("nan")))

    test_graphs = [dataset.get(i) for i in test_idx]
    loader = DataLoader(test_graphs, batch_size=cfg.bigcn.batch_size,
                        shuffle=False, num_workers=0)

    y_true, y_pred, y_prob = _predict(model, loader)

    # --- 4-class metrics ---
    acc_4    = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_per   = f1_score(y_true, y_pred, average=None,
                        labels=list(range(4)), zero_division=0).tolist()
    cm       = confusion_matrix(y_true, y_pred, labels=list(range(4)))

    if verbose:
        log.info("── Fold %d  Test Results ──────────────────────", fold_idx)
        log.info("4-class accuracy : %.4f", acc_4)
        log.info("4-class macro F1 : %.4f", f1_macro)
        for i, name in enumerate(CLASS_NAMES):
            log.info("  F1 [%-12s] : %.4f", name, f1_per[i])
        log.info("Confusion matrix (rows=true, cols=pred):")
        log.info("  %s", "\t".join(CLASS_NAMES))
        for i, row in enumerate(cm):
            log.info("  %-12s %s", CLASS_NAMES[i], "\t".join(map(str, row)))

    # --- binary metrics (true vs false, ignore uncertain / non-rumor) ---
    bin_mask = np.array([_BINARY_MAP[y] for y in y_true])
    keep     = bin_mask >= 0
    if keep.sum() > 0:
        bt = (bin_mask[keep] == 1).astype(int)         # 1 = false
        bp = np.array([_BINARY_MAP[p] for p in y_pred[keep]])
        bp = np.clip(bp, 0, 1)
        acc_bin  = float(accuracy_score(bt, bp))
        f1_bin   = float(f1_score(bt, bp, average="binary", zero_division=0))
    else:
        acc_bin, f1_bin = float("nan"), float("nan")

    if verbose:
        log.info("Binary (true vs false) accuracy : %.4f", acc_bin)
        log.info("Binary (true vs false) F1       : %.4f", f1_bin)

    return {
        "fold":         fold_idx,
        "acc_4class":   acc_4,
        "f1_macro":     f1_macro,
        "f1_true":      f1_per[0],
        "f1_false":     f1_per[1],
        "f1_unverified":f1_per[2],
        "f1_nonrumor":  f1_per[3],
        "acc_binary":   acc_bin,
        "f1_binary":    f1_bin,
        "n_test":       len(test_idx),
    }


def run_evaluation(
    split:      str = "twitter15",
    fold_idx:   int = -1,
    all_folds:  bool = False,
    n_splits:   int = 5,
    seed:       int = 42,
) -> List[Dict[str, float]]:
    """
    Evaluate trained BiGCN.

    Parameters
    ----------
    fold_idx  : evaluate a single fold (-1 → use all_folds flag)
    all_folds : if True, evaluate every fold and average
    """
    if split == "both":
        ds15 = TwitterRumourDataset("twitter15")
        ds16 = TwitterRumourDataset("twitter16")
        dataset = ds15 + ds16
    else:
        dataset = TwitterRumourDataset(split)

    folds   = get_cv_splits(dataset, n_splits=n_splits, seed=seed)
    ckpt_dir = Path(cfg.paths.bigcn_checkpoint).parent / split

    if all_folds:
        fold_indices = list(range(n_splits))
    else:
        fold_indices = [fold_idx if fold_idx >= 0 else 0]

    results: List[Dict[str, float]] = []
    for fi in fold_indices:
        _train, _val, test_idx = folds[fi]
        r = evaluate_fold(dataset, fi, test_idx, ckpt_dir)
        results.append(r)

    if len(results) > 1:
        log.info("=== Averaged across %d folds ===", len(results))
        for key in ["acc_4class", "f1_macro", "f1_false", "acc_binary", "f1_binary"]:
            vals = [r[key] for r in results if not np.isnan(r.get(key, float("nan")))]
            if vals:
                log.info("  %-20s : %.4f ± %.4f", key, np.mean(vals), np.std(vals))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BiGCN")
    parser.add_argument("--split",      default="twitter15",
                        choices=["twitter15", "twitter16", "both"])
    parser.add_argument("--fold",       type=int, default=0)
    parser.add_argument("--all-folds",  action="store_true")
    parser.add_argument("--n-splits",   type=int, default=5)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    run_evaluation(
        split=args.split,
        fold_idx=args.fold,
        all_folds=args.all_folds,
        n_splits=args.n_splits,
        seed=args.seed,
    )
"""
gnn/evaluate.py
===============
Evaluate a trained BiGCN on the held-out test fold(s).

Changes from previous version
------------------------------
* Removed get_cv_splits import — folds now come from dataset.get_kfold_splits().
* Removed graph_features / _denormalise_graph_features — that field no longer
  exists on Data objects in the current dataset.py.
* BiGCN is instantiated with no constructor args (parameters come from cfg).
* split="both" no longer relies on the removed __add__ operator.
* Per-prediction CSV retains: fold, tweet_id, true/pred labels, confidence,
  and num_nodes (derived from batch.ptr, always available).

Usage
-----
    # Evaluate fold 0
    python -m gnn.evaluate --split twitter15 --fold 0

    # Evaluate all folds and average
    python -m gnn.evaluate --split twitter15 --all-folds
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from torch_geometric.loader import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg
from gnn.bigcn import BiGCN
from gnn.dataset import TwitterRumourDataset

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["true", "false", "unverified", "non-rumor"]

# Binary mapping (per spec):
#   true(0) / non-rumor(3) → 0,  false(1) → 1,  unverified(2) → skip
_BINARY_MAP = {0: 0, 1: 1, 2: -1, 3: 0}

# Where prediction CSVs are written
_PRED_DIR = Path(__file__).parent.parent / "evaluation"
_PRED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def _load_dataset(split: str) -> TwitterRumourDataset:
    """
    Load dataset for the given split.

    For split="both" the two datasets are merged into a single
    TwitterRumourDataset-like object by directly combining their data lists.
    This avoids the removed __add__ operator.
    """
    if split == "both":
        ds15 = TwitterRumourDataset("twitter15")
        ds16 = TwitterRumourDataset("twitter16")
        # Build a lightweight merged wrapper
        combined = TwitterRumourDataset.__new__(TwitterRumourDataset)
        combined.split = "both"
        combined._data = ds15._data + ds16._data
        return combined
    return TwitterRumourDataset(split)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _load_checkpoint(ckpt_path: Path) -> Dict:
    return torch.load(ckpt_path, map_location=DEVICE, weights_only=False)


def _predict(
    model:  BiGCN,
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[int]]:
    """
    Run inference over *loader* and collect per-sample outputs.

    Returns
    -------
    y_true      : int array (N,)
    y_pred      : int array (N,)
    y_prob      : float array (N, num_classes)
    tweet_ids   : list of str, length N
    num_nodes   : list of int, length N — nodes per graph, from batch.ptr
    """
    model.eval()
    all_true:     List[int]        = []
    all_pred:     List[int]        = []
    all_prob:     List[np.ndarray] = []
    all_tids:     List[str]        = []
    all_num_nodes: List[int]       = []

    with torch.no_grad():
        for batch in loader:
            batch  = batch.to(DEVICE)
            logits = model(batch)
            probs  = torch.softmax(logits, dim=-1)
            preds  = logits.argmax(dim=-1)

            all_true.extend(batch.y.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())
            all_prob.append(probs.cpu().numpy())

            # tweet_ids — PyG batches string attributes as a plain list
            tids = batch.tweet_id
            if isinstance(tids, str):
                tids = [tids]
            all_tids.extend(list(tids))

            # Per-graph node counts from the batch pointer tensor
            # batch.ptr shape: (B+1,); diff gives nodes per graph
            if hasattr(batch, "ptr") and batch.ptr is not None:
                sizes = (batch.ptr[1:] - batch.ptr[:-1]).cpu().tolist()
            else:
                # Fallback: single graph
                sizes = [int(batch.num_nodes)]
            all_num_nodes.extend([int(s) for s in sizes])

    return (
        np.array(all_true),
        np.array(all_pred),
        np.vstack(all_prob),
        all_tids,
        all_num_nodes,
    )


# ---------------------------------------------------------------------------
# Per-prediction CSV
# ---------------------------------------------------------------------------

def _write_predictions_csv(
    fold_idx:   int,
    split:      str,
    tweet_ids:  List[str],
    y_true:     np.ndarray,
    y_pred:     np.ndarray,
    y_prob:     np.ndarray,
    num_nodes:  List[int],
) -> Path:
    out = _PRED_DIR / f"predictions_{split}_fold{fold_idx}.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "fold", "tweet_id",
            "true_label", "true_label_name",
            "pred_label", "pred_label_name",
            "confidence",
            "num_nodes",
        ])
        for i, tid in enumerate(tweet_ids):
            yt, yp = int(y_true[i]), int(y_pred[i])
            conf   = float(y_prob[i, yp])
            w.writerow([
                fold_idx, tid,
                yt, CLASS_NAMES[yt],
                yp, CLASS_NAMES[yp],
                f"{conf:.4f}",
                num_nodes[i],
            ])
    log.info("Wrote per-prediction CSV: %s", out)
    return out


# ---------------------------------------------------------------------------
# Single-fold evaluation
# ---------------------------------------------------------------------------

def evaluate_fold(
    dataset:  TwitterRumourDataset,
    fold_idx: int,
    test_idx: np.ndarray,
    ckpt_dir: Path,
    split:    str,
    verbose:  bool = True,
) -> Dict[str, float]:
    ckpt_path = ckpt_dir / f"fold{fold_idx}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run gnn/train.py first."
        )

    ckpt  = _load_checkpoint(ckpt_path)

    # BiGCN draws all hyperparameters from cfg — no constructor args needed.
    # The checkpoint's in_dim/graph_dim keys are no longer written by train.py,
    # but if an older checkpoint has them we just ignore them.
    model = BiGCN().to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    log.info(
        "Loaded fold %d checkpoint (epoch %d, val_acc=%.4f)",
        fold_idx,
        ckpt.get("epoch", -1),
        ckpt.get("val_acc", float("nan")),
    )

    test_graphs = [dataset.get(i) for i in test_idx]
    loader = DataLoader(
        test_graphs,
        batch_size=cfg.bigcn.batch_size,
        shuffle=False,
        num_workers=0,
    )

    y_true, y_pred, y_prob, tweet_ids, num_nodes = _predict(model, loader)

    # ── 4-class metrics ──────────────────────────────────────────────────
    acc_4    = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_per   = f1_score(
        y_true, y_pred, average=None,
        labels=list(range(4)), zero_division=0,
    ).tolist()
    cm = confusion_matrix(y_true, y_pred, labels=list(range(4)))

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

    # ── Binary (true vs false) metrics ───────────────────────────────────
    bin_mask = np.array([_BINARY_MAP[int(y)] for y in y_true])
    keep = bin_mask >= 0
    if keep.sum() > 0:
        bt = (bin_mask[keep] == 1).astype(int)           # 1 = false
        bp = np.array([_BINARY_MAP[int(p)] for p in y_pred[keep]])
        bp = np.clip(bp, 0, 1)
        acc_bin = float(accuracy_score(bt, bp))
        f1_bin  = float(f1_score(bt, bp, average="binary", zero_division=0))
    else:
        acc_bin, f1_bin = float("nan"), float("nan")

    if verbose:
        log.info("Binary (true vs false) accuracy : %.4f", acc_bin)
        log.info("Binary (true vs false) F1       : %.4f", f1_bin)

    # ── Per-prediction CSV ────────────────────────────────────────────────
    _write_predictions_csv(
        fold_idx, split, tweet_ids,
        y_true, y_pred, y_prob, num_nodes,
    )

    return {
        "fold":           fold_idx,
        "acc_4class":     acc_4,
        "f1_macro":       f1_macro,
        "f1_true":        f1_per[0],
        "f1_false":       f1_per[1],
        "f1_unverified":  f1_per[2],
        "f1_nonrumor":    f1_per[3],
        "acc_binary":     acc_bin,
        "f1_binary":      f1_bin,
        "n_test":         int(len(test_idx)),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_evaluation(
    split:     str  = "twitter15",
    fold_idx:  int  = 0,
    all_folds: bool = False,
    n_splits:  int  = 5,
    seed:      int  = 42,
) -> List[Dict[str, float]]:
    dataset = _load_dataset(split)

    # get_kfold_splits is a method on TwitterRumourDataset.
    # Previously this called the standalone get_cv_splits() function.
    folds = dataset.get_kfold_splits(n_splits=n_splits, seed=seed)

    ckpt_dir = Path(cfg.paths.bigcn_checkpoint).parent / split

    fold_indices = list(range(n_splits)) if all_folds else [fold_idx]

    results: List[Dict[str, float]] = []
    for fi in fold_indices:
        _train, _val, test_idx = folds[fi]
        r = evaluate_fold(dataset, fi, test_idx, ckpt_dir, split=split)
        results.append(r)

    if len(results) > 1:
        log.info("=== Averaged across %d folds ===", len(results))
        for key in ["acc_4class", "f1_macro", "f1_false", "acc_binary", "f1_binary"]:
            vals = [r[key] for r in results
                    if not np.isnan(r.get(key, float("nan")))]
            if vals:
                log.info(
                    "  %-20s : %.4f ± %.4f",
                    key, np.mean(vals), np.std(vals),
                )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BiGCN")
    parser.add_argument("--split",     default="twitter15",
                        choices=["twitter15", "twitter16", "both"])
    parser.add_argument("--fold",      type=int, default=0)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--n-splits",  type=int, default=5)
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    run_evaluation(
        split=args.split,
        fold_idx=args.fold,
        all_folds=args.all_folds,
        n_splits=args.n_splits,
        seed=args.seed,
    )
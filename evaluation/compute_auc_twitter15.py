"""
Compute multi-class AUC (One-vs-Rest) for Twitter15 fold predictions.

Since only per-prediction confidence scores are available (not full class
probability vectors), OvR scores are approximated as:
  - class k predicted → score for k = confidence
  - other class predicted → score for k = 1 - confidence
"""

import glob
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
CLASSES = {0: "true", 1: "false", 2: "unverified", 3: "non-rumor"}
COLORS = ["#2196F3", "#F44336", "#FF9800", "#4CAF50"]


def load_folds(pattern: str) -> pd.DataFrame:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def build_ovr_scores(df: pd.DataFrame, n_classes: int = 4) -> np.ndarray:
    """
    Build an (n_samples, n_classes) score matrix from predicted label + confidence.
    For class k: score = confidence if pred==k, else (1-confidence).
    """
    scores = np.full((len(df), n_classes), fill_value=np.nan)
    for k in range(n_classes):
        mask = df["pred_label"].values == k
        scores[mask, k] = df.loc[mask, "confidence"].values
        scores[~mask, k] = 1.0 - df.loc[~mask, "confidence"].values
    return scores


def compute_fold_auc(df: pd.DataFrame):
    """Return per-class and macro AUC for a single fold dataframe."""
    y_true = df["true_label"].values
    scores = build_ovr_scores(df)
    y_bin = label_binarize(y_true, classes=list(CLASSES.keys()))

    per_class = {}
    for k in CLASSES:
        if y_bin[:, k].sum() == 0:
            per_class[k] = float("nan")
        else:
            per_class[k] = roc_auc_score(y_bin[:, k], scores[:, k])

    macro = np.nanmean(list(per_class.values()))
    return per_class, macro


def plot_roc_curves(df_all: pd.DataFrame, out_path: str):
    y_true = df_all["true_label"].values
    scores = build_ovr_scores(df_all)
    y_bin = label_binarize(y_true, classes=list(CLASSES.keys()))

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.flatten()

    overall_aucs = []
    for idx, (k, name) in enumerate(CLASSES.items()):
        ax = axes[idx]
        fpr, tpr, _ = roc_curve(y_bin[:, k], scores[:, k])
        roc_auc = auc(fpr, tpr)
        overall_aucs.append(roc_auc)

        ax.plot(fpr, tpr, color=COLORS[idx], lw=2,
                label=f"AUC = {roc_auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f'ROC — class "{name}"')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)

    fig.suptitle(
        f"Twitter15 One-vs-Rest ROC Curves (all folds)\n"
        f"Macro-AUC = {np.mean(overall_aucs):.4f}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC plot -> {out_path}")


def plot_fold_macro_auc(fold_macros: dict, out_path: str):
    folds = sorted(fold_macros.keys())
    macros = [fold_macros[f] for f in folds]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar([f"Fold {f}" for f in folds], macros,
                  color=COLORS[:len(folds)], edgecolor="black", alpha=0.85)
    ax.axhline(np.mean(macros), color="red", linestyle="--",
               label=f"Mean = {np.mean(macros):.4f}")
    for bar, val in zip(bars, macros):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro-AUC")
    ax.set_title("Twitter15 — Macro-AUC per Fold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved fold AUC plot -> {out_path}")


def main():
    pattern = os.path.join(EVAL_DIR, "predictions_twitter15_fold*.csv")
    df_all = load_folds(pattern)
    print(f"Loaded {len(df_all)} samples across folds: {sorted(df_all['fold'].unique())}\n")

    # ── Per-fold AUC ──────────────────────────────────────────────────────────
    fold_macros = {}
    rows = []
    print(f"{'Fold':<6} {'true':>8} {'false':>8} {'unverif':>9} {'non-rumor':>10} {'macro':>8}")
    print("-" * 55)
    for fold_id, group in df_all.groupby("fold"):
        per_class, macro = compute_fold_auc(group)
        fold_macros[fold_id] = macro
        rows.append({
            "fold": fold_id,
            **{f"auc_{CLASSES[k]}": per_class[k] for k in CLASSES},
            "macro_auc": macro,
        })
        vals = [f"{per_class[k]:.4f}" for k in CLASSES]
        print(f"{fold_id:<6} {vals[0]:>8} {vals[1]:>8} {vals[2]:>9} {vals[3]:>10} {macro:>8.4f}")

    # ── Overall (pooled) AUC ──────────────────────────────────────────────────
    overall_per_class, overall_macro = compute_fold_auc(df_all)
    print("-" * 55)
    vals = [f"{overall_per_class[k]:.4f}" for k in CLASSES]
    print(f"{'ALL':<6} {vals[0]:>8} {vals[1]:>8} {vals[2]:>9} {vals[3]:>10} {overall_macro:>8.4f}")

    rows.append({
        "fold": "all",
        **{f"auc_{CLASSES[k]}": overall_per_class[k] for k in CLASSES},
        "macro_auc": overall_macro,
    })

    # ── Save CSV summary ──────────────────────────────────────────────────────
    summary_path = os.path.join(EVAL_DIR, "auc_summary_twitter15.csv")
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"\nSaved AUC summary -> {summary_path}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_roc_curves(df_all, os.path.join(EVAL_DIR, "roc_curves_twitter15.png"))
    plot_fold_macro_auc(
        {k: v for k, v in fold_macros.items()},
        os.path.join(EVAL_DIR, "macro_auc_per_fold_twitter15.png"),
    )


if __name__ == "__main__":
    main()

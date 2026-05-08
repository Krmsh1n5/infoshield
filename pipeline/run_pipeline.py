"""
pipeline/run_pipeline.py
========================
End-to-end InfoGuard pipeline reproducing Table II from the paper.

Paper Section V.B (exact quote):
    "We simulate the content propagation under these dropout-based alterations
     by sampling a random content from the dataset and then following its
     propagation while randomly dropping content transfers based on dropout
     probabilities d* given by Algorithm 2."

KEY DESIGN DECISIONS (paper-aligned):
──────────────────────────────────────
1. SIMULATION MODEL — cascade-following, NOT synthetic SIR.
   The paper follows the REAL observed WICO cascade graph with probabilistic
   edge removal. It does NOT generate synthetic cascades from b values.
   - Control (no dropout): BFS covers the full cascade → E[R∞] ≈ mean cascade size ≈ 50
   - With dropout: edges are removed with probability (1 - d*[u,v])

2. S_t AND I_t IN THE LP — use GLOBAL class sizes, not cascade-local.
   The LP was derived for the full union network (153,779 nodes).
   At each BFS level:
     S_counts[v] = global_class_sizes[v] - nodes_in_Cv_already_reached
     I_counts[u] = nodes_in_Cu_currently_in_BFS_frontier

3. DISCRIMINATION comes from CASCADE GRAPH STRUCTURE, not from b values.
   - False (conspiracy) cascades have more cross-class edges
     (fringe content spreading to general audience)
   - True (non-conspiracy) cascades have more within-class edges
     (verified content staying in trusted communities)
   LP sets d*[u,v≠u] low (suppress cross-class) and d*[u,u] high (preserve within-class).
   Applied to a false cascade (mostly cross-class): cascade shrinks.
   Applied to a true cascade (mostly within-class): cascade preserved.

4. b VALUES are used only by the LP, NOT by the simulation itself.
   b_uv = P(one infected Cu-user infects one specific susceptible Cv-user per step)
   Must divide by global_class_sizes[v] to get per-pair probability.
   (See network_model.py _estimate_b fix)

Usage
-----
    python -m pipeline.run_pipeline --n-samples 500
    python -m pipeline.run_pipeline --n-samples 500 --label-source bigcn --fold 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg

sys.path.insert(0, str(Path(__file__).parent.parent / "graph_engine"))
from network_model import SBM, make_synthetic_sbm
from optimizer import DropoutOptimizer

from .sbm_fitter import fit_sbm, load_wico_all_cascades, find_root_user
from .segmented_sbm_fitter import load_segment, DEFAULT_SEGMENTS_DIR

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# Paper Table II (α, λ) settings — control is (None, None)
ALPHA_LAMBDA_PAIRS = [
    (None,  None),
    (1.5,   1.0),
    (2.0,   1.5),
    (3.0,   2.0),
]


# ── Core simulation: paper Section V.B ───────────────────────────────────────

def simulate_cascade_following(
    G:                   nx.DiGraph,
    partition:           dict,
    root:                str,
    sbm:                 SBM,
    alpha:               Optional[float],
    lam:                 Optional[float],
    global_class_sizes:  np.ndarray,
    seed:                int = 42,
) -> tuple[int, list[str]]:
    rng      = np.random.default_rng(seed)
    k        = sbm.k
    lp_types: list[str] = []          # FIX BUG 1: initialise before the loop

    if alpha is not None:
        opt = DropoutOptimizer(
            sbm.b_minus, sbm.b_plus,
            alpha=alpha, lambda_weight=lam,
        )

    # Seed: root + depth-1 retweeters are already "observed" — skip to step 2.
    # reachable and I_set reflect real cascade up to depth 1.
    depth1    = {n for n in G.successors(root) if n in sbm.partition}
    reachable = {root} | depth1
    I_set     = depth1 if depth1 else {root}

    # LP I_counts: always normalised to 1 effective seed in the root class.
    # This keeps the LP constraint at α × 1 regardless of how many depth-1
    # nodes exist, preventing over-preservation of both content types.
    lp_I_counts = np.zeros(k)                  # FIX BUG 2: build the override
    root_class  = sbm.partition.get(root)       # that will actually be passed
    if root_class is not None and 0 <= root_class < k:
        lp_I_counts[root_class] = 1.0

    while I_set:
        # S_counts: global scale (153,779-node network), minus already-reached.
        S_counts = global_class_sizes.astype(float).copy()
        for node in reachable:
            u = partition.get(node)
            if u is not None and 0 <= u < k:
                S_counts[u] = max(0.0, S_counts[u] - 1.0)

        # At step 0 use the normalised single-seed LP I_counts.
        # From step 1 onward use the real BFS frontier class counts.
        if not lp_types:                        # step 0 (first LP call)
            i_for_lp = lp_I_counts
        else:                                   # steps 1+
            i_for_lp = np.zeros(k)
            for node in I_set:
                u = partition.get(node)
                if u is not None and 0 <= u < k:
                    i_for_lp[u] += 1.0

        if alpha is not None:
            lp_result = opt.solve(S_counts, i_for_lp)  # FIX BUG 2: use i_for_lp
            d_star    = lp_result.dropout_matrix
            lp_types.append(lp_result.lp_type)
        else:
            d_star = np.ones((k, k))

        new_I = set()
        for node in I_set:
            u = partition.get(node, 0)
            for neighbour in G.successors(node):
                if neighbour in reachable:
                    continue
                v = partition.get(neighbour, 0)
                if rng.random() < d_star[u, v]:
                    new_I.add(neighbour)
                    reachable.add(neighbour)

        I_set = new_I

    return len(reachable), lp_types   # FIX BUG 3: lp_types always defined now


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    n_samples:       int  = 500,
    label_source:    str  = "wico_folders",
    fold:            int  = 0,
    split:           str  = "twitter15",
    skip_sbm_fit:    bool = False,
    force_sbm_refit: bool = False,
    output_dir:      Optional[Path] = None,
    seed:            int  = 42,
    segment_name:    Optional[str]  = None,
) -> pd.DataFrame:
    """
    Run the full WICO evaluation pipeline.

    For each of n_samples simulations:
      1. Sample a random true-content WICO cascade AND a random false-content cascade.
      2. For each of the 4 (α, λ) settings:
         a. Run cascade-following BFS on the true cascade with LP dropout.
         b. Run cascade-following BFS on the false cascade with LP dropout.
         c. Record R∞ for each.

    Parameters
    ----------
    segment_name : if provided, load the SBM from the named segment directory
        (data/processed/sbm_segments/{segment_name}/) instead of the global SBM.
        One of: "conspiracy_5g", "other_conspiracy", "all_conspiracy".
        When set, results are saved with the segment name as a suffix, e.g.
        pipeline_table2_conspiracy_5g.csv.

    Returns a DataFrame with columns:
        sample_id, alpha, lambda, pair_label, label, cascade_size
    """
    if output_dir is None:
        output_dir = Path(cfg.paths.evaluation)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # ── Step 1: SBM ──────────────────────────────────────────────────────────
    if segment_name is not None:
        # Load segment-specific SBM from sbm_segments/{segment_name}/
        seg_dir = DEFAULT_SEGMENTS_DIR / segment_name
        log.info("Loading segment SBM: %s from %s", segment_name, seg_dir)
        try:
            sbm = load_segment(segment_name)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Segment {segment_name!r} not found at {seg_dir}. "
                "Run `python -m pipeline.segmented_sbm_fitter --segment "
                f"{segment_name}` first."
            )
    else:
        sbm_dir = Path(cfg.paths.sbm_matrices)
        if skip_sbm_fit and (sbm_dir / "b_plus.npy").exists():
            log.info("Loading SBM from %s", sbm_dir)
            sbm = SBM.load(sbm_dir)
        else:
            log.info("Fitting SBM (label_source=%s) ...", label_source)
            sbm = fit_sbm(
                label_source   = label_source,
                fold           = fold,
                split          = split,
                force_refit    = force_sbm_refit,
            )

    k = sbm.k
    log.info("SBM ready: %s", sbm)

    # b+ and b- sanity check — they should be in 1e-8 to ~1e-3
    b_max = max(sbm.b_plus.max(), sbm.b_minus.max())
    if b_max > 0.1:
        log.warning(
            "b values reach %.4f — likely un-normalised. "
            "Re-fit with --force-sbm-refit after applying the "
            "class_sizes[v] fix to network_model._estimate_b.",
            b_max,
        )

    # ── Step 2: Load WICO cascades ────────────────────────────────────────────
    log.info("Loading WICO Graph cascades ...")
    raw_cascades = load_wico_all_cascades()

    def _reachable_from_root(G, root):
        if root is None or root not in G:
            return 0
        return len(nx.descendants(G, root)) + 1

    MIN_REACHABLE = 20   # minimum nodes actually reachable from root

    true_cascades = [
        (G, cid) for G, cid, lbl, _ in raw_cascades
        if lbl == "true"
        and _reachable_from_root(G, find_root_user(G)) >= MIN_REACHABLE
    ]
    false_cascades = [
        (G, cid) for G, cid, lbl, _ in raw_cascades
        if lbl == "false"
        and _reachable_from_root(G, find_root_user(G)) >= MIN_REACHABLE
    ]

    if not true_cascades or not false_cascades:
        raise RuntimeError(
            f"Missing cascade pools: true={len(true_cascades)}, false={len(false_cascades)}. "
            "Check cfg.paths.wico_graph."
        )
    log.info("True cascades: %d | False cascades: %d",
             len(true_cascades), len(false_cascades))

    # Global class sizes for LP S_t computation
    global_class_sizes = sbm.class_sizes.astype(float)

    # ── Step 3: Simulate ─────────────────────────────────────────────────────
    n_settings = len(ALPHA_LAMBDA_PAIRS)
    log.info(
        "Running %d samples × %d settings × 2 labels = %d simulations",
        n_samples, n_settings, n_samples * n_settings * 2,
    )

    rows = []

    for sample_id in range(n_samples):
        if sample_id % 50 == 0:
            log.info("  Sample %d / %d", sample_id, n_samples)

        sim_seed = int(rng.integers(0, 2**31))

        # Sample one true and one false cascade for this sample
        t_idx = int(rng.integers(0, len(true_cascades)))
        f_idx = int(rng.integers(0, len(false_cascades)))
        G_true,  _ = true_cascades[t_idx]
        G_false, _ = false_cascades[f_idx]

        # Find root users
        root_true  = find_root_user(G_true)  or next(iter(G_true.nodes()), None)
        root_false = find_root_user(G_false) or next(iter(G_false.nodes()), None)

        if root_true is None or root_false is None:
            continue

        for alpha, lam in ALPHA_LAMBDA_PAIRS:
            pair_label = "control" if alpha is None else f"α={alpha},λ={lam}"

            for label, G, root in [("true",  G_true,  root_true),
                                    ("false", G_false, root_false)]:
                size, _ = simulate_cascade_following(
                    G                  = G,
                    partition          = sbm.partition,
                    root               = root,
                    sbm                = sbm,
                    alpha              = alpha,
                    lam                = lam,
                    global_class_sizes = global_class_sizes,
                    seed               = sim_seed,
                )
                rows.append({
                    "sample_id":    sample_id,
                    "alpha":        alpha if alpha is not None else "control",
                    "lambda":       lam   if lam   is not None else "control",
                    "pair_label":   pair_label,
                    "label":        label,
                    "cascade_size": size,
                })

    df = pd.DataFrame(rows)

    # ── Step 4: Save and display ──────────────────────────────────────────────
    suffix    = f"_{segment_name}" if segment_name else ""
    raw_path  = output_dir / f"pipeline_results_raw{suffix}.csv"
    df.to_csv(raw_path, index=False)
    log.info("Raw results: %s (%d rows)", raw_path, len(df))

    table2      = _build_table2(df)
    table2_path = output_dir / f"pipeline_table2{suffix}.csv"
    table2.to_csv(table2_path, index=False)
    log.info("Table II: %s", table2_path)

    _print_table2(table2, n_samples)
    return df


# ── Table II builder ──────────────────────────────────────────────────────────

def _build_table2(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (pair_label, alpha, lam), grp in df.groupby(
            ["pair_label", "alpha", "lambda"]):
        true_s  = grp[grp["label"] == "true" ]["cascade_size"]
        false_s = grp[grp["label"] == "false"]["cascade_size"]
        rows.append({
            "alpha":             alpha,
            "lambda":            lam,
            "pair_label":        pair_label,
            "E_R_inf_true":      round(float(true_s.mean()),  1),
            "E_R_inf_false":     round(float(false_s.mean()), 1),
            "P_R_inf_lt5_true":  round(float((true_s  < 5).mean()), 2),
            "P_R_inf_lt5_false": round(float((false_s < 5).mean()), 2),
            "n_samples_true":    len(true_s),
            "n_samples_false":   len(false_s),
        })
    table2 = pd.DataFrame(rows)

    def _sort_key(row):
        a = row["alpha"]
        return (0, 0) if a == "control" else (1, float(a))

    table2["_sort"] = table2.apply(_sort_key, axis=1)
    return table2.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)


def _print_table2(table2: pd.DataFrame, n_samples: int) -> None:
    paper = {
        "control":       (50.1, 48.7, 0.01, 0.00),
        "α=1.5,λ=1.0":  (32.8, 26.1, 0.05, 0.13),
        "α=2.0,λ=1.5":  (39.4, 28.2, 0.04, 0.09),
        "α=3.0,λ=2.0":  (41.1, 36.0, 0.00, 0.02),
    }
    sep = "─" * 88
    print(f"\n{'='*88}")
    print(f"  InfoGuard Pipeline — Table II Reproduction  (n_samples={n_samples})")
    print(f"{'='*88}")
    print(f"  {'Setting':<20} {'E[R∞] true':>12} {'E[R∞] false':>12} "
          f"{'P<5 true':>10} {'P<5 false':>10}")
    print(sep)
    for _, row in table2.iterrows():
        pl = row["pair_label"]
        print(f"  {pl:<20} {row['E_R_inf_true']:>12.1f} {row['E_R_inf_false']:>12.1f} "
              f"{row['P_R_inf_lt5_true']:>10.2f} {row['P_R_inf_lt5_false']:>10.2f}")
    print(sep)
    print("  Paper (Table II):")
    for setting, (et, ef, pt, pf) in paper.items():
        print(f"  {setting:<20} {et:>12.1f} {ef:>12.1f} {pt:>10.2f} {pf:>10.2f}")
    print(f"{'='*88}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run InfoGuard pipeline (paper-aligned)")
    parser.add_argument("--n-samples",       type=int, default=500)
    parser.add_argument("--label-source",    default="wico_folders",
                        choices=["wico_folders", "bigcn"])
    parser.add_argument("--fold",            type=int, default=0)
    parser.add_argument("--split",           default="twitter15")
    parser.add_argument("--skip-sbm-fit",    action="store_true")
    parser.add_argument("--force-sbm-refit", action="store_true")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--output-dir",      default=None)
    parser.add_argument(
        "--segment",
        default=None,
        choices=["all_conspiracy", "conspiracy_5g", "other_conspiracy"],
        help=(
            "Use a per-segment SBM instead of the global SBM. "
            "The segment must have been fitted first via "
            "pipeline.segmented_sbm_fitter."
        ),
    )
    args = parser.parse_args()

    run_pipeline(
        n_samples       = args.n_samples,
        label_source    = args.label_source,
        fold            = args.fold,
        split           = args.split,
        skip_sbm_fit    = args.skip_sbm_fit,
        force_sbm_refit = args.force_sbm_refit,
        seed            = args.seed,
        output_dir      = Path(args.output_dir) if args.output_dir else None,
        segment_name    = args.segment,
    )
"""
pipeline/run_pipeline.py
========================
End-to-end InfoGuard pipeline: load WICO cascades → fit SBM (if needed)
→ run Algorithm 2 at each cascade step → measure R∞ reduction.

Reproduces Table II from the paper:
    Bayiz & Topcu (2022). arXiv:2211.04617v1

Table II layout:
    α    λ    E[R∞]_true  E[R∞]_false  P[R∞<5]_true  P[R∞<5]_false
    —    —    50.1        48.7         0.01           0.00    ← control
    1.5  1    32.8        26.1         0.05           0.13
    2    1.5  39.4        28.2         0.04           0.09
    3    2    41.1        36.0         0.00           0.02

Usage
-----
    # Quick run (50 samples, folder labels)
    python -m pipeline.run_pipeline --n-samples 50 --label-source wico_folders

    # Full reproduction (500 samples, folder labels)
    python -m pipeline.run_pipeline --n-samples 500 --label-source wico_folders

    # With BiGCN labels
    python -m pipeline.run_pipeline --n-samples 500 --label-source bigcn --fold 0

    # Skip SBM fitting (use saved matrices)
    python -m pipeline.run_pipeline --skip-sbm-fit

Output
------
Saves to cfg.paths.evaluation/:
    pipeline_results_raw.csv   — one row per (sample, alpha, lambda)
    pipeline_table2.csv        — Table II reproduction
    pipeline_cascade_sizes.csv — per-sample cascade sizes for plots
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg

sys.path.insert(0, str(Path(__file__).parent.parent / "graph_engine"))
from network_model import SBM, make_synthetic_sbm
from sir_simulation import SIRState, NodeSIRSimulator, SBMSIRSimulator

from sbm_fitter import fit_sbm, load_wico_all_cascades, find_root_user

sys.path.insert(0, str(Path(__file__).parent.parent / "graph_engine"))
from optimizer import DropoutOptimizer

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# ── Paper's (α, λ) settings from Table II ────────────────────────────────────
ALPHA_LAMBDA_PAIRS = [
    (None,  None),   # control — no dropout
    (1.5,   1.0),
    (2.0,   1.5),
    (3.0,   2.0),
]


# ── Single cascade simulation ─────────────────────────────────────────────────

def _simulate_cascade(
    G:        nx.DiGraph,
    sbm:      SBM,
    label:    str,
    alpha:    Optional[float],
    lam:      Optional[float],
    seed:     int = 42,
    max_steps: int = 50,
) -> int:
    """
    Run one SIR cascade on graph G using the SBM model.

    If alpha is None → control run (no dropout, d* = 1 everywhere).
    Returns R∞ (total number of users infected).
    """
    root = find_root_user(G)
    if root is None or root not in G:
        root = next(iter(G.nodes()))
    seed_nodes = {root}

    if alpha is None:
        # Control: no dropout
        sim     = NodeSIRSimulator(G, sbm, content=label, rng_seed=seed)
        return sim.final_cascade_size(seed_nodes=seed_nodes, max_steps=max_steps)

    # Algorithm 2: LP dropout at each step
    opt = DropoutOptimizer(
        sbm.b_minus, sbm.b_plus,
        alpha=alpha, lambda_weight=lam,
    )
    sim = NodeSIRSimulator(G, sbm, content=label, rng_seed=seed)

    # Build dropout sequence by running LP at each step
    # We pre-run the simulation to get S/I counts per class, then compute d*
    k = sbm.k
    all_nodes = set(G.nodes())
    I_set = seed_nodes & all_nodes
    S_set = all_nodes - I_set
    R_set: set = set()

    state = SIRState(
        S_set=S_set, I_set=I_set, R_set=R_set,
        S_counts=sim._counts_from_set(S_set, k),
        I_counts=sim._counts_from_set(I_set, k),
        R_counts=np.zeros(k),
        t=0,
    )

    dropout_sequence = []
    for _ in range(max_steps):
        if state.is_terminated():
            break
        lp_result = opt.solve(state.S_counts, state.I_counts)
        dropout_sequence.append(lp_result.dropout_matrix.copy())
        state = sim.step(state, dropout=lp_result.dropout_matrix)

    return state.cascade_size


def _sbm_simulate_cascade(
    sbm:      SBM,
    label:    str,
    alpha:    Optional[float],
    lam:      Optional[float],
    seed:     int = 42,
    max_steps: int = 50,
) -> int:
    """
    SBM-level (class-count) cascade simulation. Fast fallback when
    individual cascade graphs are not available.
    """
    # Seed: proportional to class sizes, 1% infected at t=0
    total = sbm.class_sizes.astype(float)
    I0    = np.maximum(np.round(total * 0.01), 1.0)
    I0    = I0.astype(float)
    S0    = total - I0

    if alpha is None:
        sim = SBMSIRSimulator(sbm, content=label, rng_seed=seed)
        return sim.final_cascade_size(I0, total, max_steps=max_steps)

    opt = DropoutOptimizer(
        sbm.b_minus, sbm.b_plus,
        alpha=alpha, lambda_weight=lam,
    )
    sim = SBMSIRSimulator(sbm, content=label, rng_seed=seed)

    state = SIRState(
        S_counts=S0.copy(), I_counts=I0.copy(),
        R_counts=np.zeros_like(S0)
    )
    dropout_seq = []
    for _ in range(max_steps):
        if state.is_terminated():
            break
        lp_result = opt.solve(state.S_counts, state.I_counts)
        dropout_seq.append(lp_result.dropout_matrix.copy())
        state = sim.step(state, dropout=lp_result.dropout_matrix)

    return state.cascade_size


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(
    n_samples:      int  = 500,
    label_source:   str  = "wico_folders",
    fold:           int  = 0,
    split:          str  = "twitter15",
    skip_sbm_fit:   bool = False,
    force_sbm_refit:bool = False,
    output_dir:     Optional[Path] = None,
    seed:           int  = 42,
    use_node_level: bool = True,
) -> pd.DataFrame:
    """
    Run the full evaluation pipeline and return results DataFrame.

    Parameters
    ----------
    n_samples      : number of cascade simulations per (alpha, lambda) setting
    label_source   : "wico_folders" | "bigcn"
    fold           : BiGCN fold (only used when label_source="bigcn")
    split          : BiGCN split
    skip_sbm_fit   : if True, load existing SBM without re-fitting
    force_sbm_refit: force re-fitting even if matrices exist
    output_dir     : where to save results (default: cfg.paths.evaluation)
    seed           : random seed for reproducibility
    use_node_level : if True, simulate on real WICO graphs (accurate);
                     if False, simulate at class-count level (fast)

    Returns
    -------
    DataFrame with columns:
        sample_id, alpha, lambda, label, cascade_size
    """
    if output_dir is None:
        output_dir = Path(cfg.paths.evaluation)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # ── Step 1: Fit or load SBM ───────────────────────────────────────────────
    if skip_sbm_fit and (Path(cfg.paths.sbm_matrices) / "b_plus.npy").exists():
        log.info("Loading existing SBM matrices from %s", cfg.paths.sbm_matrices)
        sbm = SBM.load(Path(cfg.paths.sbm_matrices))
    else:
        log.info("Fitting SBM (label_source=%s)...", label_source)
        sbm = fit_sbm(
            label_source  = label_source,
            fold          = fold,
            split         = split,
            force_refit   = force_sbm_refit,
        )

    log.info("SBM ready: %s", sbm)

    # ── Step 2: Load WICO cascades ────────────────────────────────────────────
    if use_node_level:
        log.info("Loading WICO Graph cascades for node-level simulation...")
        raw_cascades = load_wico_all_cascades()

        # Split by label
        true_graphs  = [(G, cid) for G, cid, lbl, _ in raw_cascades if lbl == "true"]
        false_graphs = [(G, cid) for G, cid, lbl, _ in raw_cascades if lbl == "false"]

        if not true_graphs or not false_graphs:
            log.warning(
                "Missing true or false graphs (true=%d, false=%d). "
                "Falling back to SBM-level simulation.",
                len(true_graphs), len(false_graphs),
            )
            use_node_level = False
        else:
            log.info("True cascades: %d | False cascades: %d",
                     len(true_graphs), len(false_graphs))

    # ── Step 3: Run simulations ───────────────────────────────────────────────
    rows = []
    n_alpha_lambda = len(ALPHA_LAMBDA_PAIRS)

    log.info(
        "Running %d samples × %d (α,λ) pairs × 2 labels = %d simulations",
        n_samples, n_alpha_lambda, n_samples * n_alpha_lambda * 2,
    )

    for sample_id in range(n_samples):
        if sample_id % 50 == 0:
            log.info("  Sample %d / %d", sample_id, n_samples)

        sim_seed = int(rng.integers(0, 2**31))

        for alpha, lam in ALPHA_LAMBDA_PAIRS:
            pair_label = "control" if alpha is None else f"α={alpha},λ={lam}"

            for label in ("true", "false"):
                if use_node_level:
                    pool = true_graphs if label == "true" else false_graphs
                    idx  = int(rng.integers(0, len(pool)))
                    G, _ = pool[idx]
                    size = _simulate_cascade(
                        G=G, sbm=sbm, label=label,
                        alpha=alpha, lam=lam,
                        seed=sim_seed, max_steps=50,
                    )
                else:
                    size = _sbm_simulate_cascade(
                        sbm=sbm, label=label,
                        alpha=alpha, lam=lam,
                        seed=sim_seed, max_steps=50,
                    )

                rows.append({
                    "sample_id":  sample_id,
                    "alpha":      alpha if alpha is not None else "control",
                    "lambda":     lam   if lam   is not None else "control",
                    "pair_label": pair_label,
                    "label":      label,
                    "cascade_size": size,
                })

    df = pd.DataFrame(rows)

    # ── Step 4: Save raw results ───────────────────────────────────────────────
    raw_path = output_dir / "pipeline_results_raw.csv"
    df.to_csv(raw_path, index=False)
    log.info("Raw results saved: %s (%d rows)", raw_path, len(df))

    # ── Step 5: Build Table II ────────────────────────────────────────────────
    table2 = _build_table2(df)
    table2_path = output_dir / "pipeline_table2.csv"
    table2.to_csv(table2_path, index=False)
    log.info("Table II saved: %s", table2_path)

    # ── Step 6: Print Table II ────────────────────────────────────────────────
    _print_table2(table2, n_samples)

    return df


def _build_table2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build Table II from raw results DataFrame.

    Matches the paper's Table II format:
        α  λ  E[R∞]_true  E[R∞]_false  P[R∞<5]_true  P[R∞<5]_false
    """
    rows = []
    for (pair_label, alpha, lam), grp in df.groupby(["pair_label", "alpha", "lambda"]):
        true_sizes  = grp[grp["label"] == "true" ]["cascade_size"]
        false_sizes = grp[grp["label"] == "false"]["cascade_size"]
        rows.append({
            "alpha":              alpha,
            "lambda":             lam,
            "pair_label":         pair_label,
            "E_R_inf_true":       round(float(true_sizes.mean()),  1) if len(true_sizes)  > 0 else float("nan"),
            "E_R_inf_false":      round(float(false_sizes.mean()), 1) if len(false_sizes) > 0 else float("nan"),
            "P_R_inf_lt5_true":   round(float((true_sizes  < 5).mean()), 2) if len(true_sizes)  > 0 else float("nan"),
            "P_R_inf_lt5_false":  round(float((false_sizes < 5).mean()), 2) if len(false_sizes) > 0 else float("nan"),
            "n_samples_true":     len(true_sizes),
            "n_samples_false":    len(false_sizes),
        })

    table2 = pd.DataFrame(rows)

    # Order rows: control first, then by alpha ascending
    def _sort_key(row):
        a = row["alpha"]
        return (0, 0) if a == "control" else (1, float(a))
    table2["_sort"] = table2.apply(_sort_key, axis=1)
    table2 = table2.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    return table2


def _print_table2(table2: pd.DataFrame, n_samples: int) -> None:
    """Print Table II in the paper's format for visual inspection."""
    paper = {
        "control": (50.1, 48.7, 0.01, 0.00),
        "α=1.5,λ=1.0": (32.8, 26.1, 0.05, 0.13),
        "α=2.0,λ=1.5": (39.4, 28.2, 0.04, 0.09),
        "α=3.0,λ=2.0": (41.1, 36.0, 0.00, 0.02),
    }

    sep = "─" * 88
    print(f"\n{'='*88}")
    print(f"  InfoGuard Pipeline — Table II Reproduction  (n_samples={n_samples})")
    print(f"{'='*88}")
    print(f"  {'Setting':<20} {'E[R∞] true':>12} {'E[R∞] false':>12} {'P<5 true':>10} {'P<5 false':>10}")
    print(sep)
    for _, row in table2.iterrows():
        pl = row["pair_label"]
        print(
            f"  {pl:<20} "
            f"{row['E_R_inf_true']:>12.1f} "
            f"{row['E_R_inf_false']:>12.1f} "
            f"{row['P_R_inf_lt5_true']:>10.2f} "
            f"{row['P_R_inf_lt5_false']:>10.2f}"
        )
    print(sep)
    print("  Paper (Table II):")
    for setting, (et, ef, pt, pf) in paper.items():
        print(f"  {setting:<20} {et:>12.1f} {ef:>12.1f} {pt:>10.2f} {pf:>10.2f}")
    print(f"{'='*88}\n")


# ── Synthetic validation (no WICO data needed) ───────────────────────────────

def run_synthetic_validation(
    k:        int   = 2,
    n_users:  int   = 1000,
    n_samples:int   = 200,
    seed:     int   = 42,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Run Table I validation using synthetic SBMs.
    Reproduces Table I from Section V.A of the paper.
    No dataset required.

    Parameters
    ----------
    k        : number of partitions (2 or 3)
    n_users  : total synthetic users
    n_samples: Monte Carlo samples per (x, y, alpha, lambda) combination
    """
    if output_dir is None:
        output_dir = Path(cfg.paths.evaluation)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Sweep x and y (50 values each, matching paper)
    x_vals = np.linspace(0.0, 0.01, 10)   # reduced for speed; paper uses 50
    y_vals = np.linspace(0.0, 0.001, 10)

    rows = []
    total = len(x_vals) * len(y_vals) * len(ALPHA_LAMBDA_PAIRS[1:])  # skip control
    done  = 0

    for x in x_vals:
        for y in y_vals:
            sbm = make_synthetic_sbm(k=k, x=x, y=y, n_users=n_users, balanced=True)

            for alpha, lam in [(1.5, 1.0), (2.0, 1.5)]:
                done += 1
                if done % 20 == 0:
                    log.info("  Synthetic sweep: %d / %d", done, total)

                sizes_false = []
                sizes_true  = []

                for s in range(n_samples):
                    sim_seed = int(rng.integers(0, 2**31))
                    I0 = np.ones(k) * 2.0
                    total_u = sbm.class_sizes.astype(float)

                    sf = _sbm_simulate_cascade(sbm, "false", alpha, lam, sim_seed)
                    st = _sbm_simulate_cascade(sbm, "true",  alpha, lam, sim_seed + 1)
                    sizes_false.append(sf)
                    sizes_true.append(st)

                n = n_users
                rows.append({
                    "k": k, "x": round(x, 4), "y": round(y, 4),
                    "alpha": alpha, "lambda": lam,
                    "E_R_inf_N_false": round(np.mean(sizes_false) / n, 3),
                    "E_R_inf_N_true":  round(np.mean(sizes_true)  / n, 3),
                    "P_lt_N10_false":  round(np.mean(np.array(sizes_false) < n / 10), 3),
                    "P_lt_N10_true":   round(np.mean(np.array(sizes_true)  < n / 10), 3),
                })

    df = pd.DataFrame(rows)
    path = output_dir / f"synthetic_table1_k{k}.csv"
    df.to_csv(path, index=False)
    log.info("Synthetic Table I saved: %s", path)
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run InfoGuard pipeline")
    parser.add_argument("--n-samples",      type=int, default=500)
    parser.add_argument("--label-source",   default="wico_folders",
                        choices=["wico_folders", "bigcn"])
    parser.add_argument("--fold",           type=int, default=0)
    parser.add_argument("--split",          default="twitter15")
    parser.add_argument("--skip-sbm-fit",   action="store_true")
    parser.add_argument("--force-sbm-refit",action="store_true")
    parser.add_argument("--no-node-level",  action="store_true",
                        help="Use SBM-level (fast) simulation instead of node-level")
    parser.add_argument("--seed",           type=int, default=42)
    parser.add_argument("--output-dir",     default=None)
    parser.add_argument("--synthetic",      action="store_true",
                        help="Run synthetic Table I validation (no WICO needed)")
    parser.add_argument("--synthetic-k",    type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.synthetic:
        run_synthetic_validation(
            k=args.synthetic_k,
            output_dir=output_dir,
        )
    else:
        run_pipeline(
            n_samples       = args.n_samples,
            label_source    = args.label_source,
            fold            = args.fold,
            split           = args.split,
            skip_sbm_fit    = args.skip_sbm_fit,
            force_sbm_refit = args.force_sbm_refit,
            output_dir      = output_dir,
            seed            = args.seed,
            use_node_level  = not args.no_node_level,
        )
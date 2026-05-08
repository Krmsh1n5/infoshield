"""
pipeline/segmented_sbm_fitter.py
=================================
Extends the SBM fitter to support multiple community segments, each with its
own b⁺ and b⁻ matrices.  This directly addresses the criticism that a single
matrix cannot represent Facebook's (or any platform's) segmented communities
— e.g. diaspora vs regional vs political vs health sub-networks.

Theory
------
The original SBMFitter (sbm_fitter.py) builds ONE union graph from ALL WICO
cascades and fits a single (b⁺, b⁻) pair.  That means the dropout matrix
d* returned by the LP is a population-average that blurs differences between
community types.

SegmentedSBMFitter trains one SBM per named segment, reusing the *same*
Louvain partition (fitted on the full union graph) so that b matrices are
directly comparable across segments — the class indices mean the same users
in every segment.

Segments defined here (WICO proof-of-concept)
----------------------------------------------
"conspiracy_5g"    : false=5G_Conspiracy_Graphs, true=Non_Conspiracy_Graphs
"other_conspiracy" : false=Other_Graphs,         true=Non_Conspiracy_Graphs
"all_conspiracy"   : false=5G + Other,           true=Non_Conspiracy  (≡ global SBM)

Key insight to verify
---------------------
5G conspiracy content should cross class boundaries MORE aggressively than
"other conspiracy" content, because 5G misinformation was specifically spread
via influencers who bridged political and health communities.  The comparison
DataFrame exposes this via the b_minus off-diagonal mean.

Output layout
-------------
data/processed/sbm_segments/
    conspiracy_5g/
        b_plus.npy, b_minus.npy, class_sizes.npy,
        partition_keys.npy, partition_values.npy, k.npy
        metadata.json
    other_conspiracy/   … same
    all_conspiracy/     … same

Usage
-----
    # Fit all segments (uses folder labels, reuses global partition if present)
    python -m pipeline.segmented_sbm_fitter

    # Force re-fit even if saved data exist
    python -m pipeline.segmented_sbm_fitter --force-refit

    # Fit a single named segment
    python -m pipeline.segmented_sbm_fitter --segment conspiracy_5g

    # Print comparison table
    python -m pipeline.segmented_sbm_fitter --compare
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg

sys.path.insert(0, str(Path(__file__).parent.parent / "graph_engine"))
from network_model import SBM, SBMFitter

from .sbm_fitter import load_wico_all_cascades

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


# ── Segment definitions ───────────────────────────────────────────────────────

#: Each segment specifies which WICO class directories supply its false/true
#: cascades.  ``k_override`` can pin k to a specific value; None means the
#: fitter's default (from cfg or the global partition's k).
SEGMENTS: dict[str, dict] = {
    "conspiracy_5g": {
        "false_dirs": ["5G_Conspiracy_Graphs"],
        "true_dirs":  ["Non_Conspiracy_Graphs"],
        "k_override": None,
        "description": (
            "5G conspiracy cascades vs verified non-conspiracy content. "
            "Hypothesis: 5G content crosses class boundaries more aggressively "
            "than other misinformation types due to health-political bridging."
        ),
    },
    "other_conspiracy": {
        "false_dirs": ["Other_Graphs"],
        "true_dirs":  ["Non_Conspiracy_Graphs"],
        "k_override": None,
        "description": (
            "General conspiracy cascades vs verified non-conspiracy content. "
            "Baseline segment for comparison with conspiracy_5g."
        ),
    },
    "all_conspiracy": {
        "false_dirs": ["5G_Conspiracy_Graphs", "Other_Graphs"],
        "true_dirs":  ["Non_Conspiracy_Graphs"],
        "k_override": None,
        "description": (
            "All conspiracy cascades combined (equivalent to the global SBM). "
            "Used as the population-average baseline."
        ),
    },
}

#: Default output directory for segment SBMs
DEFAULT_SEGMENTS_DIR = Path(cfg.paths.data_processed) / "sbm_segments"


# ── SegmentedSBMFitter ────────────────────────────────────────────────────────

class SegmentedSBMFitter:
    """
    Fits one SBM per named community segment from WICO Graph cascades.

    The fitting strategy is two-phase:

    Phase 1 — Global partition
        Build the union graph from ALL cascades (all three WICO class dirs)
        and run Louvain clustering once.  This gives a single ``partition``
        dict and ``k`` that is shared across all segments, ensuring that
        b-matrix entries are directly comparable (class-u in segment A is
        the same set of users as class-u in segment B).

    Phase 2 — Per-segment b estimation
        For each segment, filter the pre-loaded cascades by the segment's
        ``false_dirs`` and ``true_dirs``, then call ``SBMFitter._estimate_b``
        (via a lightweight wrapper) using the shared partition.

    Parameters
    ----------
    wico_graph_dir : path to the WICO Graph root
        (defaults to cfg.paths.wico_graph)
    output_dir : where to write per-segment subdirectories
        (defaults to DEFAULT_SEGMENTS_DIR)
    num_partitions : target k for Louvain
        (defaults to cfg.sbm.num_partitions)
    clustering_resolution : Louvain resolution
        (defaults to cfg.sbm.clustering_resolution)
    min_partition_fraction : tiny-class merge threshold
        (defaults to cfg.sbm.min_partition_fraction)
    seed : random seed
        (defaults to cfg.seed)
    """

    def __init__(
        self,
        wico_graph_dir:          Optional[Path]  = None,
        output_dir:              Optional[Path]  = None,
        num_partitions:          Optional[int]   = None,
        clustering_resolution:   Optional[float] = None,
        min_partition_fraction:  Optional[float] = None,
        seed:                    Optional[int]   = None,
    ) -> None:
        self._wico_dir  = Path(wico_graph_dir or cfg.paths.wico_graph)
        self._out_dir   = Path(output_dir or DEFAULT_SEGMENTS_DIR)

        # Delegate partition/fitting config to SBMFitter (which reads cfg)
        self._fitter_kwargs = dict(
            num_partitions         = num_partitions,
            clustering_resolution  = clustering_resolution,
            min_partition_fraction = min_partition_fraction,
            seed                   = seed,
        )

        # Populated after _load_cascades() / _build_global_partition()
        self._cascades:  list[tuple[nx.DiGraph, str, str, str]] = []
        self._partition: Optional[dict]       = None
        self._k:         Optional[int]        = None
        self._class_sizes: Optional[np.ndarray] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_all_segments(
        self,
        segments: Optional[dict] = None,
        force_refit: bool = False,
    ) -> dict[str, SBM]:
        """
        Fit one SBM per segment and save each to ``output_dir/{segment_name}/``.

        Parameters
        ----------
        segments : mapping of segment-name → segment-config dict
            Defaults to the module-level ``SEGMENTS`` constant.
        force_refit : if True, ignore existing saved matrices and refit.

        Returns
        -------
        dict mapping segment_name → fitted SBM
        """
        if segments is None:
            segments = SEGMENTS

        self._ensure_cascades_loaded()
        self._ensure_partition_built()

        results: dict[str, SBM] = {}
        for name, seg_cfg in segments.items():
            seg_dir = self._out_dir / name
            sbm = self._fit_one_segment(
                name        = name,
                seg_cfg     = seg_cfg,
                output_dir  = seg_dir,
                force_refit = force_refit,
            )
            results[name] = sbm

        log.info("All segments fitted: %s", list(results.keys()))
        return results

    def fit_segment(
        self,
        name:        str,
        force_refit: bool = False,
    ) -> SBM:
        """
        Fit and save a single named segment.

        The global partition is built from ALL cascades before fitting,
        guaranteeing cross-segment comparability.
        """
        if name not in SEGMENTS:
            raise ValueError(
                f"Unknown segment {name!r}. "
                f"Available: {list(SEGMENTS.keys())}"
            )
        self._ensure_cascades_loaded()
        self._ensure_partition_built()

        seg_dir = self._out_dir / name
        return self._fit_one_segment(
            name        = name,
            seg_cfg     = SEGMENTS[name],
            output_dir  = seg_dir,
            force_refit = force_refit,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_cascades_loaded(self) -> None:
        if not self._cascades:
            log.info("Loading all WICO cascades from %s …", self._wico_dir)
            self._cascades = load_wico_all_cascades(self._wico_dir)
            if not self._cascades:
                raise RuntimeError(
                    f"No WICO cascades found in {self._wico_dir}. "
                    "Check cfg.paths.wico_graph."
                )
            log.info("Loaded %d cascades total.", len(self._cascades))

    def _ensure_partition_built(self) -> None:
        """Build (or load from disk) the global Louvain partition."""
        # Cache: reuse if already built this session
        if self._partition is not None:
            return

        # Try loading the global SBM partition first (saves ~minutes)
        global_sbm_dir = Path(cfg.paths.sbm_matrices)
        if (global_sbm_dir / "partition_keys.npy").exists():
            log.info(
                "Reusing global partition from %s (k will be inferred).",
                global_sbm_dir,
            )
            sbm = SBM.load(global_sbm_dir)
            self._partition   = sbm.partition
            self._k           = sbm.k
            self._class_sizes = sbm.class_sizes
            log.info(
                "Global partition loaded: k=%d, total_nodes=%d",
                self._k, len(self._partition),
            )
            return

        # Build partition from scratch using ALL cascades
        log.info("Building global Louvain partition from all cascades …")
        all_graphs = [G for G, *_ in self._cascades]

        # Use a fresh SBMFitter just for partition building
        tmp_fitter = SBMFitter(**{k: v for k, v in self._fitter_kwargs.items()
                                  if v is not None})
        for G in all_graphs:
            # We only need the union graph, so label doesn't matter here.
            # Add every graph as "true" (dummy) — fit() will cluster on the union.
            tmp_fitter._true_graphs.append(G)

        # Trigger partition building without saving (we'll save per-segment)
        _ = tmp_fitter.fit()
        self._partition   = tmp_fitter._partition
        self._k           = tmp_fitter._k
        self._class_sizes = tmp_fitter._class_sizes
        log.info(
            "Global partition ready: k=%d, total_nodes=%d",
            self._k, len(self._partition),
        )

    def _fit_one_segment(
        self,
        name:        str,
        seg_cfg:     dict,
        output_dir:  Path,
        force_refit: bool,
    ) -> SBM:
        """Fit (or load) one segment's SBM."""
        b_plus_path = output_dir / "b_plus.npy"
        if b_plus_path.exists() and not force_refit:
            log.info("[%s] Loading existing SBM from %s", name, output_dir)
            return SBM.load(output_dir)

        log.info("[%s] Fitting …", name)

        false_dirs = set(seg_cfg["false_dirs"])
        true_dirs  = set(seg_cfg["true_dirs"])

        false_graphs: list[nx.DiGraph] = []
        true_graphs:  list[nx.DiGraph] = []
        n_false = n_true = 0

        for G, cid, label, class_dir in self._cascades:
            dir_name = class_dir   # e.g. "5G_Conspiracy_Graphs"
            if dir_name in false_dirs:
                false_graphs.append(G)
                n_false += 1
            elif dir_name in true_dirs:
                true_graphs.append(G)
                n_true += 1
            # else: skip (e.g. Other_Graphs when we only want 5G)

        log.info(
            "[%s] %d false cascades, %d true cascades",
            name, n_false, n_true,
        )
        if not false_graphs or not true_graphs:
            raise RuntimeError(
                f"Segment {name!r}: missing cascades "
                f"(false={n_false}, true={n_true}). "
                "Check segment false_dirs/true_dirs against WICO folder names."
            )

        k         = self._k
        partition = self._partition

        # Use SBMFitter's _estimate_b via a temporary instance
        estimator = _BEstimator(
            k                  = k,
            partition          = partition,
            class_sizes        = self._class_sizes,
            fitter_kwargs      = self._fitter_kwargs,
        )
        b_plus  = estimator.estimate(true_graphs,  label="true")
        b_minus = estimator.estimate(false_graphs, label="false")

        sbm = SBM(
            b_plus      = b_plus,
            b_minus     = b_minus,
            k           = k,
            partition   = partition,
            class_sizes = self._class_sizes,
        )

        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        sbm.save(output_dir)
        _write_metadata(output_dir, name=name, seg_cfg=seg_cfg,
                        n_true=n_true, n_false=n_false, k=k)
        log.info("[%s] Saved to %s", name, output_dir)
        return sbm


# ── Thin wrapper around SBMFitter._estimate_b ─────────────────────────────────

class _BEstimator:
    """
    Wraps SBMFitter._estimate_b so we can re-use its MLE logic without
    duplicating code, while supplying an externally-computed partition.
    """

    def __init__(
        self,
        k:            int,
        partition:    dict,
        class_sizes:  np.ndarray,
        fitter_kwargs: dict,
    ) -> None:
        self._fitter            = SBMFitter(
            **{k_: v for k_, v in fitter_kwargs.items() if v is not None}
        )
        # Inject the shared global partition directly
        self._fitter._partition   = partition
        self._fitter._k           = k
        self._fitter._class_sizes = class_sizes

    def estimate(self, graphs: list[nx.DiGraph], label: str) -> np.ndarray:
        return self._fitter._estimate_b(
            graphs    = graphs,
            partition = self._fitter._partition,
            k         = self._fitter._k,
            label     = label,
        )


# ── Persistence helpers ────────────────────────────────────────────────────────

def _write_metadata(
    directory: Path,
    name:      str,
    seg_cfg:   dict,
    n_true:    int,
    n_false:   int,
    k:         int,
) -> None:
    meta = {
        "segment":     name,
        "false_dirs":  seg_cfg["false_dirs"],
        "true_dirs":   seg_cfg["true_dirs"],
        "description": seg_cfg.get("description", ""),
        "n_true_cascades":  n_true,
        "n_false_cascades": n_false,
        "k":           k,
    }
    with open(directory / "metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)


def load_segment(
    segment_name: str,
    segments_dir: Optional[Path] = None,
) -> SBM:
    """
    Load a previously fitted segment SBM from disk.

    Parameters
    ----------
    segment_name : one of the keys in SEGMENTS ("conspiracy_5g", …)
    segments_dir : root of the sbm_segments directory tree
        (defaults to DEFAULT_SEGMENTS_DIR)

    Returns
    -------
    Fitted SBM object for the requested segment.

    Raises
    ------
    FileNotFoundError if the segment has not been fitted yet.
    """
    if segments_dir is None:
        segments_dir = DEFAULT_SEGMENTS_DIR

    seg_dir    = Path(segments_dir) / segment_name
    b_plus_npy = seg_dir / "b_plus.npy"

    if not b_plus_npy.exists():
        raise FileNotFoundError(
            f"Segment {segment_name!r} not found at {seg_dir}. "
            "Run fit_all_segments() first."
        )
    return SBM.load(seg_dir)


# ── Comparison ────────────────────────────────────────────────────────────────

def compare_segments(
    segment_names: Optional[list[str]] = None,
    segments_dir:  Optional[Path]      = None,
) -> pd.DataFrame:
    """
    Load all fitted segments and return a comparison DataFrame.

    Columns
    -------
    segment              : segment name
    k                    : number of polarization classes
    n_classes            : same as k (alias)
    b_plus_diag_mean     : mean of b⁺ diagonal (within-class true-content transfer)
    b_plus_offdiag_mean  : mean of b⁺ off-diagonal (cross-class true-content transfer)
    b_minus_diag_mean    : mean of b⁻ diagonal
    b_minus_offdiag_mean : mean of b⁻ off-diagonal (key: cross-class false spread)
    plus_diag_offdiag_ratio  : b_plus_diag / b_plus_offdiag  (within vs cross, true)
    minus_diag_offdiag_ratio : b_minus_diag / b_minus_offdiag (within vs cross, false)
    cross_class_asymmetry    : b_minus_offdiag / b_plus_offdiag
        > 1 means false content crosses class boundaries more than true content.
        Higher values = more discriminable = better LP performance.
    b_plus_max           : max entry in b⁺
    b_minus_max          : max entry in b⁻

    Key insight
    -----------
    A segment where ``cross_class_asymmetry`` >> 1 is MORE discriminable by the
    LP optimizer: the dropout d* can aggressively cut cross-class edges
    (suppressing false content) while leaving within-class edges intact
    (preserving true content).  The hypothesis is that ``conspiracy_5g``
    shows a higher ratio than ``other_conspiracy``.
    """
    if segment_names is None:
        segment_names = list(SEGMENTS.keys())
    if segments_dir is None:
        segments_dir = DEFAULT_SEGMENTS_DIR

    rows = []
    for name in segment_names:
        try:
            sbm  = load_segment(name, segments_dir=segments_dir)
        except FileNotFoundError as exc:
            log.warning("Skipping %s: %s", name, exc)
            continue

        bp = sbm.b_plus
        bm = sbm.b_minus
        k  = sbm.k

        diag_mask    = np.eye(k, dtype=bool)
        offdiag_mask = ~diag_mask

        bp_diag    = bp[diag_mask].mean()
        bp_offdiag = bp[offdiag_mask].mean() if k > 1 else np.nan
        bm_diag    = bm[diag_mask].mean()
        bm_offdiag = bm[offdiag_mask].mean() if k > 1 else np.nan

        def _ratio(num, den):
            return float(num / den) if den > 0 else float("inf")

        rows.append({
            "segment":                  name,
            "k":                        k,
            "n_classes":                k,
            "b_plus_diag_mean":         float(bp_diag),
            "b_plus_offdiag_mean":      float(bp_offdiag) if not np.isnan(bp_offdiag) else None,
            "b_minus_diag_mean":        float(bm_diag),
            "b_minus_offdiag_mean":     float(bm_offdiag) if not np.isnan(bm_offdiag) else None,
            "plus_diag_offdiag_ratio":  _ratio(bp_diag, bp_offdiag) if k > 1 else None,
            "minus_diag_offdiag_ratio": _ratio(bm_diag, bm_offdiag) if k > 1 else None,
            "cross_class_asymmetry":    _ratio(bm_offdiag, bp_offdiag) if k > 1 else None,
            "b_plus_max":               float(bp.max()),
            "b_minus_max":              float(bm.max()),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("cross_class_asymmetry", ascending=False).reset_index(drop=True)
    return df


def print_comparison(df: pd.DataFrame) -> None:
    """Pretty-print the segment comparison table."""
    if df.empty:
        print("No segments fitted yet. Run fit_all_segments() first.")
        return

    sep = "─" * 100
    print(f"\n{'='*100}")
    print("  InfoGuard — Segment Comparison: Cross-class Asymmetry  (b⁻_offdiag / b⁺_offdiag)")
    print(f"{'='*100}")
    print(f"  {'Segment':<22} {'k':>3}  {'b⁺ diag':>10} {'b⁻ offdiag':>12} "
          f"{'asym':>8}  Interpretation")
    print(sep)
    for _, row in df.iterrows():
        asym = row.get("cross_class_asymmetry")
        asym_str = f"{asym:.2f}x" if asym is not None else "  N/A"
        interp = (
            "HIGH — 5G bridges political & health communities" if row["segment"] == "conspiracy_5g"
            else "BASELINE — general conspiracy cross-class spread" if row["segment"] == "other_conspiracy"
            else "AVERAGE — population-level SBM"
        )
        print(
            f"  {row['segment']:<22} {row['k']:>3}  "
            f"{row['b_plus_diag_mean']:>10.2e} "
            f"{row['b_minus_offdiag_mean']:>12.2e} "
            f"{asym_str:>8}  {interp}"
        )
    print(sep)
    print()
    if len(df) >= 2:
        top    = df.iloc[0]
        bottom = df.iloc[-1]
        ratio  = (top["cross_class_asymmetry"] or 1) / max(1e-12, bottom["cross_class_asymmetry"] or 1)
        print(
            f"  Hypothesis test: {top['segment']} asymmetry "
            f"{'>' if ratio >= 1 else '<'} {bottom['segment']} asymmetry  "
            f"(ratio = {ratio:.2f}x)"
        )
        if top["segment"] == "conspiracy_5g" and ratio > 1.0:
            print("  ✓ Confirmed: 5G conspiracy crosses class boundaries more aggressively.")
        elif top["segment"] == "conspiracy_5g":
            print("  ✗ Not confirmed: 5G asymmetry not higher than other_conspiracy.")
        print()
    print(f"{'='*100}\n")


# ── Top-level convenience functions ──────────────────────────────────────────

def fit_all_segments(
    wico_graph_dir: Optional[Path] = None,
    output_dir:     Optional[Path] = None,
    force_refit:    bool = False,
) -> dict[str, SBM]:
    """
    Module-level convenience wrapper for SegmentedSBMFitter.fit_all_segments().

    Equivalent to::

        fitter = SegmentedSBMFitter(wico_graph_dir, output_dir)
        return fitter.fit_all_segments(force_refit=force_refit)
    """
    fitter = SegmentedSBMFitter(
        wico_graph_dir = wico_graph_dir,
        output_dir     = output_dir,
    )
    return fitter.fit_all_segments(force_refit=force_refit)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit per-segment SBMs from WICO Graph cascades."
    )
    parser.add_argument(
        "--segment",
        default=None,
        choices=list(SEGMENTS.keys()),
        help="Fit only this segment (default: fit all).",
    )
    parser.add_argument(
        "--force-refit",
        action="store_true",
        help="Re-fit even if saved matrices exist.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print cross-segment comparison table and exit.",
    )
    parser.add_argument(
        "--wico-graph-dir",
        default=None,
        help="Override cfg.paths.wico_graph.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override default sbm_segments output directory.",
    )
    args = parser.parse_args()

    wico_dir   = Path(args.wico_graph_dir) if args.wico_graph_dir else None
    output_dir = Path(args.output_dir)     if args.output_dir     else None

    if args.compare:
        df = compare_segments(segments_dir=output_dir)
        print_comparison(df)
        sys.exit(0)

    fitter = SegmentedSBMFitter(
        wico_graph_dir = wico_dir,
        output_dir     = output_dir,
    )

    if args.segment:
        sbm = fitter.fit_segment(args.segment, force_refit=args.force_refit)
        print(f"\n=== Fitted SBM ({args.segment}) ===")
        print(sbm)
    else:
        results = fitter.fit_all_segments(force_refit=args.force_refit)
        print("\n=== All Segments Fitted ===")
        for name, sbm in results.items():
            print(f"  {name:<22}: {sbm}")

        df = compare_segments(segments_dir=output_dir)
        print_comparison(df)
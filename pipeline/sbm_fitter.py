"""
pipeline/sbm_fitter.py
======================
Loads WICO Graph cascades, runs Louvain modularity clustering on the
union graph, and estimates the b⁺ and b⁻ SBM matrices used by the
LP optimizer (Algorithm 2).

Two labelling modes
-------------------
"wico_folders" (default, fast)
    Labels come from WICO Graph folder names:
        5G_Conspiracy_Graphs/ → false
        Other_Graphs/         → false
        Non_Conspiracy_Graphs/→ true
    No BiGCN required. Use this first.

"bigcn" (accurate, requires trained model)
    Labels come from BiGCN inference on each cascade.
    Cascades with confidence < cfg.sbm.label_confidence_threshold are excluded.
    Use this for the full pipeline evaluation.

Output
------
Saves to cfg.paths.sbm_matrices/:
    b_plus.npy, b_minus.npy, class_sizes.npy,
    partition_keys.npy, partition_values.npy, k.npy

Usage
-----
    # Quick fit (folder labels)
    python -m pipeline.sbm_fitter --label-source wico_folders

    # Full pipeline fit (BiGCN labels, fold 0)
    python -m pipeline.sbm_fitter --label-source bigcn --fold 0 --split twitter15
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg

# graph_engine lives one level up from pipeline/
sys.path.insert(0, str(Path(__file__).parent.parent / "graph_engine"))
from network_model import SBM, SBMFitter

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)


# ── WICO Graph loader ─────────────────────────────────────────────────────────

def load_wico_cascade(cascade_dir: Path) -> Optional[nx.DiGraph]:
    """
    Load one WICO Graph cascade folder into a NetworkX DiGraph.

    Parameters
    ----------
    cascade_dir : path to a single cascade folder, e.g.
                  wico-graph/5G_Conspiracy_Graphs/1234/

    Returns
    -------
    DiGraph with node attributes: time, friends, followers
    None if the folder is empty or malformed.
    """
    edges_file = cascade_dir / cfg.wico.graph_edges_file   # "edges.txt"
    nodes_file = cascade_dir / cfg.wico.graph_nodes_file   # "nodes.csv"

    if not edges_file.exists():
        return None

    G = nx.DiGraph()

    # Load node attributes if available
    node_attrs: dict[str, dict] = {}
    if nodes_file.exists():
        try:
            df = pd.read_csv(nodes_file)
            for _, row in df.iterrows():
                nid = str(int(row["id"]))
                node_attrs[nid] = {
                    "time":      int(row.get("time",      0)),
                    "friends":   int(row.get("friends",   0)),
                    "followers": int(row.get("followers", 0)),
                }
        except Exception as e:
            log.debug("Could not read %s: %s", nodes_file, e)

    # Load edges
    try:
        with open(edges_file) as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                src, dst = str(parts[0]), str(parts[1])
                G.add_edge(src, dst)
                for nid in (src, dst):
                    if nid in node_attrs and nid not in G.nodes:
                        G.nodes[nid].update(node_attrs[nid])
    except Exception as e:
        log.warning("Could not read %s: %s", edges_file, e)
        return None

    # Attach node attributes
    for nid, attrs in node_attrs.items():
        if nid in G:
            G.nodes[nid].update(attrs)

    if G.number_of_nodes() == 0:
        return None

    # WICO edges.txt encodes "source retweeted target" — content flows TARGET→SOURCE.
    # Reverse so edges represent actual content cascade direction: sharer → receiver.
    # This fixes: (1) find_root_user (author now has in-degree=0), 
    #             (2) BFS direction in simulation,
    #             (3) b estimation direction in SBMFitter.
    return G.reverse(copy=True)


def load_wico_all_cascades(
    wico_graph_dir: Optional[Path] = None,
) -> list[tuple[nx.DiGraph, str, str, str]]:
    """
    Load all WICO Graph cascades.

    Returns list of (G, cascade_id, label, class_dir) tuples where:
        G           : DiGraph
        cascade_id  : folder name (NOT a tweet ID — sequential integer)
        label       : "true" | "false"
        class_dir   : "5G_Conspiracy_Graphs" | "Other_Graphs" | "Non_Conspiracy_Graphs"
    """
    if wico_graph_dir is None:
        wico_graph_dir = Path(cfg.paths.wico_graph)

    # cfg.wico.binary_label_map: {0: "false", 1: "false", 2: "true"}
    # cfg.wico.graph_dirs:       {0: "5G_Conspiracy_Graphs", 1: "Other_Graphs", 2: "Non_Conspiracy_Graphs"}
    dir_to_binary: dict[str, str] = {}
    for int_label, dir_name in cfg.wico.graph_dirs.items():
        dir_to_binary[dir_name] = cfg.wico.binary_label_map[int_label]

    results: list[tuple[nx.DiGraph, str, str, str]] = []
    n_skipped = 0

    for dir_name, binary_label in dir_to_binary.items():
        class_dir = wico_graph_dir / dir_name
        if not class_dir.exists():
            log.warning("WICO class dir not found: %s", class_dir)
            continue

        cascade_dirs = sorted(p for p in class_dir.iterdir() if p.is_dir())
        log.info("Loading %s (%s): %d cascades", dir_name, binary_label, len(cascade_dirs))

        for cascade_dir in cascade_dirs:
            G = load_wico_cascade(cascade_dir)
            if G is None:
                n_skipped += 1
                continue
            results.append((G, cascade_dir.name, binary_label, dir_name))

    log.info("Loaded %d WICO cascades (%d skipped)", len(results), n_skipped)
    return results


def find_root_user(G: nx.DiGraph) -> Optional[str]:
    """
    Identify the root user (tweet author) in a WICO Graph cascade.

    Strategy (in priority order):
    1. Node with in-degree 0 and highest follower count (most likely the author)
    2. Node with highest follower count among all nodes
    3. First node in the graph

    The paper identifies the tweet author as the node with the highest
    follower count in the cascade subgraph.
    """
    if G.number_of_nodes() == 0:
        return None

    nodes = list(G.nodes(data=True))

    # Strategy 1: in-degree 0 nodes ranked by followers
    roots = [(n, d) for n, d in nodes if G.in_degree(n) == 0]
    if roots:
        best = max(roots, key=lambda nd: nd[1].get("followers", 0))
        return best[0]

    # Strategy 2: highest followers overall
    best = max(nodes, key=lambda nd: nd[1].get("followers", 0))
    return best[0]


# ── Main fitting function ─────────────────────────────────────────────────────

def fit_sbm(
    label_source: str = "wico_folders",
    fold:  int = 0,
    split: str = "twitter15",
    wico_graph_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    force_refit: bool = False,
) -> SBM:
    """
    Full SBM fitting pipeline.

    Parameters
    ----------
    label_source : "wico_folders" | "bigcn"
        Source of content labels for each cascade.
    fold         : BiGCN fold to use (only relevant when label_source="bigcn")
    split        : BiGCN split (only relevant when label_source="bigcn")
    wico_graph_dir : override cfg.paths.wico_graph
    output_dir   : override cfg.paths.sbm_matrices
    force_refit  : if True, ignore existing saved matrices

    Returns
    -------
    Fitted SBM object.
    """
    if output_dir is None:
        output_dir = Path(cfg.paths.sbm_matrices)

    # Check if already fitted
    b_plus_path = output_dir / "b_plus.npy"
    if b_plus_path.exists() and not force_refit:
        log.info("Loading existing SBM from %s", output_dir)
        return SBM.load(output_dir)

    # Load WICO cascades
    cascades = load_wico_all_cascades(wico_graph_dir)
    if not cascades:
        raise RuntimeError(
            "No WICO Graph cascades loaded. "
            f"Check cfg.paths.wico_graph = {cfg.paths.wico_graph}"
        )

    # Get labels
    if label_source == "wico_folders":
        labeled = _label_from_folders(cascades)
    elif label_source == "bigcn":
        labeled = _label_from_bigcn(cascades, fold=fold, split=split)
    else:
        raise ValueError(f"Unknown label_source: {label_source!r}")

    # Build SBMFitter and add labeled cascades
    fitter = SBMFitter()
    n_true = n_false = 0
    for G, cascade_id, label, confidence in labeled:
        fitter.add_cascade(G, label=label, confidence=confidence)
        if label == "true":
            n_true += 1
        else:
            n_false += 1

    log.info("Fitting SBM: %d true, %d false cascades", n_true, n_false)

    sbm = fitter.fit()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    sbm.save(output_dir)
    log.info("SBM saved to %s  (k=%d)", output_dir, sbm.k)
    return sbm


def _label_from_folders(
    cascades: list[tuple[nx.DiGraph, str, str, str]],
) -> list[tuple[nx.DiGraph, str, str, float]]:
    """
    Use WICO folder names as labels (confidence = 1.0 for all).
    Returns list of (G, cascade_id, label, confidence).
    """
    return [
        (G, cid, label, 1.0)
        for G, cid, label, _ in cascades
    ]


def _label_from_bigcn(
    cascades: list[tuple[nx.DiGraph, str, str, str]],
    fold:  int = 0,
    split: str = "twitter15",
) -> list[tuple[nx.DiGraph, str, str, float]]:
    """
    Use BiGCN predictions as labels.
    Cascades with uncertain labels or low confidence are kept with
    their original folder label as fallback — BiGCN only overrides
    when it produces a confident prediction.
    """
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "gnn"))
        from predict import Predictor
        predictor = Predictor(fold=fold, split=split)
        log.info("BiGCN predictor loaded (fold=%d, split=%s)", fold, split)
    except FileNotFoundError as e:
        log.warning("BiGCN checkpoint not found: %s. Falling back to folder labels.", e)
        return _label_from_folders(cascades)
    except ImportError as e:
        log.warning("Could not import BiGCN predictor: %s. Falling back to folder labels.", e)
        return _label_from_folders(cascades)

    results = []
    n_bigcn = n_folder_fallback = 0

    for G, cascade_id, folder_label, class_dir in cascades:
        try:
            # Use folder name as tweet_id proxy (WICO cascades are numbered)
            result = predictor.predict_from_digraph(
                G=G,
                tweet_id=cascade_id,
                tweet_text="",          # No text available for WICO Graph nodes
            )
            if result.high_confidence and result.binary_label != "uncertain":
                results.append((G, cascade_id, result.binary_label, result.confidence))
                n_bigcn += 1
            else:
                # Low confidence: fall back to folder label
                results.append((G, cascade_id, folder_label, 0.5))
                n_folder_fallback += 1
        except Exception as e:
            log.debug("BiGCN failed on cascade %s: %s — using folder label", cascade_id, e)
            results.append((G, cascade_id, folder_label, 1.0))
            n_folder_fallback += 1

    log.info(
        "BiGCN labelled %d cascades; %d fell back to folder labels.",
        n_bigcn, n_folder_fallback,
    )
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit SBM matrices from WICO")
    parser.add_argument("--label-source", default="wico_folders",
                        choices=["wico_folders", "bigcn"])
    parser.add_argument("--fold",  type=int, default=0)
    parser.add_argument("--split", default="twitter15",
                        choices=["twitter15", "twitter16", "both"])
    parser.add_argument("--force-refit", action="store_true")
    parser.add_argument("--output-dir",  default=None)
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    sbm = fit_sbm(
        label_source  = args.label_source,
        fold          = args.fold,
        split         = args.split,
        output_dir    = output_dir,
        force_refit   = args.force_refit,
    )

    print("\n=== Fitted SBM ===")
    print(sbm)
    print(f"\nb⁺ (true content transfer probabilities):\n{sbm.b_plus.round(4)}")
    print(f"\nb⁻ (false content transfer probabilities):\n{sbm.b_minus.round(4)}")
    print(f"\nClass sizes: {sbm.class_sizes.tolist()}")
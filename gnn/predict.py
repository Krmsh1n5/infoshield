"""
gnn/predict.py
==============
Inference module: run trained BiGCN on a new propagation tree and return
a structured prediction that feeds the SBM pipeline in Phase 3/4.

Two entry points
----------------
1. predict_from_path(graph_path, tweet_text) — raw tree file path
2. predict_from_digraph(G, tweet_text)        — pre-built NetworkX DiGraph

Both return a PredictionResult dataclass.

Output fields
-------------
label_4class  : int          — 0=true, 1=false, 2=unverified, 3=non-rumor
label_name    : str          — human-readable
binary_label  : str          — "true" | "false" | "uncertain"
confidence    : float        — max softmax probability (0–1)
probs         : list[float]  — softmax vector (4 values)
high_confidence : bool       — confidence ≥ cfg.sbm.label_confidence_threshold

Usage
-----
    from gnn.predict import Predictor

    predictor = Predictor(fold=0, split="twitter15")

    result = predictor.predict_from_path(
        graph_path="data/raw/twitter15/tree/12345.txt",
        tweet_text="5G towers are causing COVID symptoms",
    )
    print(result.binary_label, result.confidence)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import networkx as nx
import torch
from torch_geometric.data import Data

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg
from gnn.bigcn import BiGCN
from gnn.dataset import _parse_tree_file, _encoder

log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 4-class label names (Twitter15 convention)
_LABEL_NAMES = {0: "true", 1: "false", 2: "unverified", 3: "non-rumor"}

# Binary mapping for SBM pipeline
# true(0) → "true", false(1) → "false", unverified(2) → "uncertain", non-rumor(3) → "true"
_BINARY_MAP = {0: "true", 1: "false", 2: "uncertain", 3: "true"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    tweet_id:        str
    label_4class:    int
    label_name:      str
    binary_label:    str
    confidence:      float
    probs:           List[float]
    high_confidence: bool
    num_nodes:       int

    def to_dict(self) -> dict:
        return {
            "tweet_id":        self.tweet_id,
            "label_4class":    self.label_4class,
            "label_name":      self.label_name,
            "binary_label":    self.binary_label,
            "confidence":      round(self.confidence, 4),
            "probs":           [round(p, 4) for p in self.probs],
            "high_confidence": self.high_confidence,
            "num_nodes":       self.num_nodes,
        }

    def __repr__(self) -> str:
        return (f"PredictionResult(tweet_id={self.tweet_id!r}, "
                f"label={self.label_name!r}, binary={self.binary_label!r}, "
                f"conf={self.confidence:.3f}, high_conf={self.high_confidence})")


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class Predictor:
    """
    Wraps a trained BiGCN checkpoint and exposes predict_* methods.

    Parameters
    ----------
    fold  : int    — which fold's checkpoint to load (0-indexed)
    split : str    — "twitter15" | "twitter16" | "both"
    """

    def __init__(self, fold: int = 0, split: str = "twitter15"):
        self.fold  = fold
        self.split = split
        self._model: Optional[BiGCN] = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> BiGCN:
        if self._model is not None:
            return self._model

        ckpt_dir  = Path(cfg.paths.bigcn_checkpoint).parent / self.split
        ckpt_path = ckpt_dir / f"fold{self.fold}_best.pt"

        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}\n"
                "Train the model with gnn/train.py first."
            )

        ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model = BiGCN().to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        self._model = model
        log.info("Loaded BiGCN checkpoint: %s (epoch=%d, val_acc=%.4f)",
                 ckpt_path.name, ckpt.get("epoch", -1), ckpt.get("val_acc", float("nan")))
        return model

    # ------------------------------------------------------------------
    # Graph → PyG Data
    # ------------------------------------------------------------------

    @staticmethod
    def _edges_to_pyg(
        tweet_id:  str,
        edges:     List[Tuple[str, str]],
        tweet_text: str,
    ) -> Data:
        """Build a PyG Data object from parsed edges + root tweet text."""
        all_nodes = [tweet_id]
        for p, c in edges:
            if p not in all_nodes:
                all_nodes.append(p)
            if c not in all_nodes:
                all_nodes.append(c)

        node2idx = {n: i for i, n in enumerate(all_nodes)}
        n = len(all_nodes)

        # Root embedding broadcast
        root_emb = _encoder.encode(tweet_text if tweet_text else "")
        x = root_emb.unsqueeze(0).expand(n, -1).clone()

        if edges:
            src = torch.tensor([node2idx[p] for p, _ in edges], dtype=torch.long)
            dst = torch.tensor([node2idx[c] for _, c in edges], dtype=torch.long)
            edge_index    = torch.stack([src, dst], dim=0)
            edge_index_bu = torch.stack([dst, src], dim=0)
        else:
            edge_index    = torch.zeros((2, 0), dtype=torch.long)
            edge_index_bu = torch.zeros((2, 0), dtype=torch.long)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_index_bu=edge_index_bu,
            num_nodes=n,
        )

    @staticmethod
    def _digraph_to_pyg(G: nx.DiGraph, tweet_id: str, tweet_text: str) -> Data:
        """Convert a NetworkX DiGraph into a PyG Data object."""
        nodes     = list(G.nodes())
        node2idx  = {n: i for i, n in enumerate(nodes)}
        n         = len(nodes)

        root_emb  = _encoder.encode(tweet_text if tweet_text else "")
        x         = root_emb.unsqueeze(0).expand(n, -1).clone()

        edge_list = list(G.edges())
        if edge_list:
            src = torch.tensor([node2idx[u] for u, _ in edge_list], dtype=torch.long)
            dst = torch.tensor([node2idx[v] for _, v in edge_list], dtype=torch.long)
            edge_index    = torch.stack([src, dst], dim=0)
            edge_index_bu = torch.stack([dst, src], dim=0)
        else:
            edge_index    = torch.zeros((2, 0), dtype=torch.long)
            edge_index_bu = torch.zeros((2, 0), dtype=torch.long)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_index_bu=edge_index_bu,
            num_nodes=n,
        )

    # ------------------------------------------------------------------
    # Core inference
    # ------------------------------------------------------------------

    def _infer(self, data: Data, tweet_id: str) -> PredictionResult:
        model = self._load_model()
        data  = data.to(DEVICE)

        with torch.no_grad():
            logits = model(data)                        # (1, 4)
            probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

        label_4class = int(torch.tensor(probs).argmax().item())
        confidence   = float(probs[label_4class])
        label_name   = _LABEL_NAMES[label_4class]
        binary_label = _BINARY_MAP[label_4class]
        threshold    = float(cfg.sbm.label_confidence_threshold)

        return PredictionResult(
            tweet_id=tweet_id,
            label_4class=label_4class,
            label_name=label_name,
            binary_label=binary_label,
            confidence=confidence,
            probs=probs,
            high_confidence=confidence >= threshold,
            num_nodes=data.num_nodes,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_from_path(
        self,
        graph_path: str,
        tweet_text: str = "",
    ) -> PredictionResult:
        """
        Classify a propagation tree from its raw .txt file path.

        Parameters
        ----------
        graph_path : path to twitter15/16 tree file
        tweet_text : root tweet text (optional but recommended)
        """
        path     = Path(graph_path)
        tweet_id = path.stem
        edges, _ = _parse_tree_file(path)
        data     = self._edges_to_pyg(tweet_id, edges, tweet_text)
        return self._infer(data, tweet_id)

    def predict_from_digraph(
        self,
        G:          nx.DiGraph,
        tweet_id:   str,
        tweet_text: str = "",
    ) -> PredictionResult:
        """
        Classify a propagation tree already in NetworkX DiGraph form.

        Parameters
        ----------
        G          : directed graph (nodes = tweet/user IDs, edges = propagation)
        tweet_id   : ID of the root/source tweet
        tweet_text : root tweet text (optional but recommended)
        """
        data = self._digraph_to_pyg(G, tweet_id, tweet_text)
        return self._infer(data, tweet_id)

    def predict_batch(
        self,
        items: List[Tuple[str, str]],
    ) -> List[PredictionResult]:
        """
        Classify multiple (graph_path, tweet_text) pairs efficiently.

        Useful for bulk SBM fitting in Phase 4.
        """
        results: List[PredictionResult] = []
        for graph_path, tweet_text in items:
            try:
                r = self.predict_from_path(graph_path, tweet_text)
                results.append(r)
            except Exception as e:
                log.warning("Skipping %s: %s", graph_path, e)
        return results


# ---------------------------------------------------------------------------
# CLI quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BiGCN inference")
    parser.add_argument("--tree",  required=True, help="Path to tree .txt file")
    parser.add_argument("--text",  default="",    help="Root tweet text")
    parser.add_argument("--fold",  type=int, default=0)
    parser.add_argument("--split", default="twitter15")
    args = parser.parse_args()

    predictor = Predictor(fold=args.fold, split=args.split)
    result    = predictor.predict_from_path(args.tree, args.text)
    print(result)
    print(result.to_dict())
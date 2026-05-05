"""
gnn/predict.py
==============
Inference module: run trained BiGCN on a new propagation tree and return a
structured prediction that feeds the SBM pipeline in Phase 3/4.

This module re-uses the dataset's _build_pyg_graph helper so that inference
features always match training features exactly (RoBERTa text + 4 structural
node features + 5 graph-level features).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import torch
from torch_geometric.data import Data

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg
from gnn.bigcn import BiGCN
from gnn.dataset import _build_pyg_graph, _parse_tree_file

log = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_LABEL_NAMES = {0: "true", 1: "false", 2: "unverified", 3: "non-rumor"}
_BINARY_MAP  = {0: "true", 1: "false", 2: "uncertain", 3: "true"}


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
class Predictor:
    """Wrap a trained BiGCN checkpoint and expose predict_* methods."""

    def __init__(self, fold: int = 0, split: str = "twitter15"):
        self.fold  = fold
        self.split = split
        self._model: Optional[BiGCN] = None
        self._embed_cache: Dict[str, torch.Tensor] = {}

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

        ckpt      = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        in_dim    = int(ckpt.get("in_dim",    cfg.bigcn.text_embed_dim))
        graph_dim = int(ckpt.get("graph_dim", 5))
        model     = BiGCN(in_dim=in_dim, graph_dim=graph_dim).to(DEVICE)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        self._model = model
        log.info("Loaded BiGCN checkpoint: %s (epoch=%d, val_acc=%.4f)",
                 ckpt_path.name, ckpt.get("epoch", -1),
                 ckpt.get("val_acc", float("nan")))
        return model

    # ------------------------------------------------------------------
    def _build(
        self,
        tweet_id:   str,
        edges:      List[Tuple[str, str]],
        tweet_text: str,
    ) -> Data:
        """Use the dataset's exact feature construction so train/test stay aligned."""
        return _build_pyg_graph(
            tweet_id=tweet_id,
            edges=edges,
            label=0,                                       # placeholder, ignored
            root_text=tweet_text or "",
            embed_cache=self._embed_cache,
            per_node_text=None,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _digraph_to_edges(G: nx.DiGraph, tweet_id: str) -> List[Tuple[str, str]]:
        """Return edges as (parent, child) pairs. Drops self-loops."""
        # Make sure the root is in the node set so _build_pyg_graph works
        edges = [(str(u), str(v)) for u, v in G.edges() if u != v]
        return edges

    # ------------------------------------------------------------------
    def _infer(self, data: Data, tweet_id: str) -> PredictionResult:
        model = self._load_model()
        data  = data.to(DEVICE)

        with torch.no_grad():
            logits = model(data)                            # (1, 4)
            probs  = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

        label_4class = int(torch.tensor(probs).argmax().item())
        confidence   = float(probs[label_4class])
        threshold    = float(cfg.sbm.label_confidence_threshold)

        return PredictionResult(
            tweet_id=tweet_id,
            label_4class=label_4class,
            label_name=_LABEL_NAMES[label_4class],
            binary_label=_BINARY_MAP[label_4class],
            confidence=confidence,
            probs=probs,
            high_confidence=confidence >= threshold,
            num_nodes=data.num_nodes,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def predict_from_path(self, graph_path: str, tweet_text: str = "") -> PredictionResult:
        path     = Path(graph_path)
        tweet_id = path.stem
        edges, _ = _parse_tree_file(path)
        data     = self._build(tweet_id, edges, tweet_text)
        return self._infer(data, tweet_id)

    def predict_from_digraph(
        self, G: nx.DiGraph, tweet_id: str, tweet_text: str = "",
    ) -> PredictionResult:
        edges = self._digraph_to_edges(G, tweet_id)
        data  = self._build(tweet_id, edges, tweet_text)
        return self._infer(data, tweet_id)

    def predict_batch(self, items: List[Tuple[str, str]]) -> List[PredictionResult]:
        results: List[PredictionResult] = []
        for graph_path, tweet_text in items:
            try:
                results.append(self.predict_from_path(graph_path, tweet_text))
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
"""
api/routes/classify.py
======================
POST /api/v1/classify

Accepts a text claim and optional graph edges, returns BiGCN classification.
Falls back to a text-only mock classifier if no checkpoint is available.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid

import networkx as nx
from flask import Blueprint, current_app, jsonify, request

log = logging.getLogger(__name__)

classify_bp = Blueprint("classify", __name__)


def _edges_to_digraph(edges: list) -> nx.DiGraph:
    """Convert [[src, dst], ...] to a nx.DiGraph with minimal node attrs."""
    G = nx.DiGraph()
    for e in edges:
        if len(e) >= 2:
            src, dst = str(e[0]), str(e[1])
            G.add_edge(src, dst)
    for node in G.nodes():
        G.nodes[node].setdefault("followers", 0)
        G.nodes[node].setdefault("friends", 0)
        G.nodes[node].setdefault("time", 0.0)
    return G


def _infer_cascade_pattern(G: nx.DiGraph) -> str:
    """Heuristic: classify cascade shape from graph structure."""
    if G.number_of_nodes() == 0:
        return "unknown"
    n, e = G.number_of_nodes(), G.number_of_edges()
    avg_deg = e / max(n, 1)
    try:
        depth = max(
            (len(path) for src in G.nodes()
             for path in [nx.single_source_shortest_path_length(G, src)]
             for path in [path.values()]
             if path),
            default=1,
        )
    except Exception:
        depth = 1

    if avg_deg > 3:
        return "wide_burst"
    if depth > 8:
        return "deep_chain"
    return "slow_diffusion"


def _mock_classify(text: str) -> dict:
    """Deterministic mock classification based on text hash."""
    h = int(hashlib.md5(text.encode()).hexdigest(), 16)
    labels = ["true", "false", "unverified", "non-rumor"]
    label = labels[h % 4]
    conf = 0.55 + (h % 40) / 100.0
    binary = "false" if label in ("false",) else "true" if label in ("true", "non-rumor") else "uncertain"
    return {
        "label": label,
        "binary_label": binary,
        "confidence": round(conf, 4),
        "mock": True,
    }


@classify_bp.route("/api/v1/classify", methods=["POST"])
def classify():
    body = request.get_json(force=True, silent=True) or {}
    text = body.get("text", "")
    edges = body.get("graph_edges")  # optional list of [src, dst]

    if not text and not edges:
        return jsonify({"error": "Bad request", "code": 400,
                        "detail": "Provide 'text' or 'graph_edges'."}), 400

    predictor = current_app.extensions.get("predictor")
    cascade_id = str(uuid.uuid4())

    try:
        if edges:
            G = _edges_to_digraph(edges)
            pattern = _infer_cascade_pattern(G)
            if predictor is not None:
                result = predictor.predict_from_digraph(G, tweet_id=cascade_id, tweet_text=text)
                return jsonify({
                    "label": result.label_name,
                    "binary_label": result.binary_label,
                    "confidence": result.confidence,
                    "pattern": pattern,
                    "cascade_id": cascade_id,
                    "num_nodes": result.num_nodes,
                    "mock": False,
                })
            # No predictor — use mock
            r = _mock_classify(text or str(edges[:5]))
            r.update({"pattern": pattern, "cascade_id": cascade_id, "num_nodes": G.number_of_nodes()})
            return jsonify(r)

        # Text-only path
        pattern = "unknown"
        if predictor is not None:
            # Build a minimal single-node graph for text-only inference
            G = nx.DiGraph()
            G.add_node(cascade_id, followers=0, friends=0, time=0.0)
            result = predictor.predict_from_digraph(G, tweet_id=cascade_id, tweet_text=text)
            return jsonify({
                "label": result.label_name,
                "binary_label": result.binary_label,
                "confidence": result.confidence,
                "pattern": pattern,
                "cascade_id": cascade_id,
                "num_nodes": 1,
                "mock": False,
            })

        r = _mock_classify(text)
        r.update({"pattern": pattern, "cascade_id": cascade_id, "num_nodes": 1})
        return jsonify(r)

    except Exception as exc:
        log.exception("classify error")
        return jsonify({"error": "Internal server error", "code": 500,
                        "detail": str(exc)}), 500

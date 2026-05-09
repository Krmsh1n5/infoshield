"""
api/routes/simulate.py
======================
POST /api/v1/simulate

Runs simulate_cascade_following and returns step-by-step data for dashboard
animation including d_star matrices at each BFS level.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import networkx as nx
import numpy as np
from flask import Blueprint, current_app, jsonify, request

log = logging.getLogger(__name__)

simulate_bp = Blueprint("simulate", __name__)


# ---------------------------------------------------------------------------
# Step-tracking wrapper around simulate_cascade_following
# ---------------------------------------------------------------------------

def simulate_with_steps(
    G: nx.DiGraph,
    partition: dict,
    root: str,
    sbm,
    alpha,
    lam,
    global_class_sizes: np.ndarray,
    seed: int = 42,
) -> dict:
    """
    Re-implement the BFS loop from run_pipeline.simulate_cascade_following
    but record every step's d_star and infection counts for the API response.
    """
    from graph_engine.optimizer import DropoutOptimizer

    rng = np.random.default_rng(seed)
    k = sbm.k
    lp_types: list[str] = []
    steps: list[dict] = []

    opt = None
    if alpha is not None:
        opt = DropoutOptimizer(sbm.b_minus, sbm.b_plus,
                               alpha=alpha, lambda_weight=lam)

    depth1 = {n for n in G.successors(root) if n in sbm.partition}
    reachable = {root} | depth1
    I_set = depth1 if depth1 else {root}

    lp_I_counts = np.zeros(k)
    root_class = sbm.partition.get(root)
    if root_class is not None and 0 <= root_class < k:
        lp_I_counts[root_class] = 1.0

    step_idx = 0
    while I_set:
        S_counts = global_class_sizes.astype(float).copy()
        for node in reachable:
            u = partition.get(node)
            if u is not None and 0 <= u < k:
                S_counts[u] = max(0.0, S_counts[u] - 1.0)

        if not lp_types:
            i_for_lp = lp_I_counts
        else:
            i_for_lp = np.zeros(k)
            for node in I_set:
                u = partition.get(node)
                if u is not None and 0 <= u < k:
                    i_for_lp[u] += 1.0

        if opt is not None:
            lp_result = opt.solve(S_counts, i_for_lp)
            d_star = lp_result.dropout_matrix
            lp_types.append(lp_result.lp_type)
            lp_type = lp_result.lp_type
        else:
            d_star = np.ones((k, k))
            lp_type = "control"

        new_I: set[str] = set()
        for node in I_set:
            u = partition.get(node, 0)
            for neighbour in G.successors(node):
                if neighbour in reachable:
                    continue
                v = partition.get(neighbour, 0)
                if rng.random() < d_star[u, v]:
                    new_I.add(neighbour)
                    reachable.add(neighbour)

        steps.append({
            "step": step_idx,
            "i_set_size": len(I_set),
            "new_infected": len(new_I),
            "total_reached": len(reachable),
            "lp_type": lp_type,
            "d_star": d_star.tolist(),
        })

        I_set = new_I
        step_idx += 1

    return {
        "steps": steps,
        "r_inf": len(reachable),
        "lp_types": lp_types,
        "d_star_final": steps[-1]["d_star"] if steps else np.ones((k, k)).tolist(),
    }


# ---------------------------------------------------------------------------
# Mock simulation (for demo without real SBM data)
# ---------------------------------------------------------------------------

def _mock_simulate(content: str, alpha, seed: int) -> dict:
    rng = random.Random(seed)
    k = 4
    steps = []
    total = 1
    i_size = rng.randint(3, 8)
    for s in range(rng.randint(4, 10)):
        new_inf = max(0, int(i_size * (0.6 if content == "false" and alpha is None else 0.3)))
        total += new_inf
        d = [[round(rng.uniform(0.3, 0.9), 3) for _ in range(k)] for _ in range(k)]
        steps.append({
            "step": s,
            "i_set_size": i_size,
            "new_infected": new_inf,
            "total_reached": total,
            "lp_type": "hard" if alpha else "control",
            "d_star": d,
        })
        i_size = new_inf
        if i_size == 0:
            break
    return {
        "steps": steps,
        "r_inf": total,
        "lp_types": [st["lp_type"] for st in steps],
        "d_star_final": steps[-1]["d_star"] if steps else [[1.0]*k for _ in range(k)],
        "mock": True,
    }


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@simulate_bp.route("/api/v1/simulate", methods=["POST"])
def simulate():
    body = request.get_json(force=True, silent=True) or {}
    cascade_id = body.get("cascade_id", "")
    alpha = body.get("alpha")          # float or null
    lam = body.get("lambda", 1.0)
    content = body.get("content", "false")  # "true" | "false"
    seed = int(body.get("seed", 42))

    sbm = current_app.extensions.get("sbm")
    cascade_store = current_app.extensions.get("cascade_store", {})

    # Mock mode if no SBM loaded
    if sbm is None:
        result = _mock_simulate(content, alpha, seed)
        result["cascade_id"] = cascade_id
        return jsonify(result)

    # Try to retrieve stored cascade graph
    G = cascade_store.get(cascade_id)
    if G is None:
        # Fall back to a random WICO cascade of appropriate label
        try:
            from pipeline.sbm_fitter import load_wico_all_cascades, find_root_user
            cascades = load_wico_all_cascades()
            pool = [(g, cid) for g, cid, lbl, _ in cascades if lbl == content]
            if not pool:
                pool = [(g, cid) for g, cid, lbl, _ in cascades]
            rng_pick = np.random.default_rng(seed)
            G, _ = pool[int(rng_pick.integers(0, len(pool)))]
        except Exception as exc:
            log.warning("Could not load WICO cascade: %s — using mock", exc)
            result = _mock_simulate(content, alpha, seed)
            result["cascade_id"] = cascade_id
            return jsonify(result)

    try:
        from pipeline.sbm_fitter import find_root_user
        root = find_root_user(G) or next(iter(G.nodes()), None)
        if root is None:
            return jsonify({"error": "Empty cascade graph", "code": 422,
                            "detail": "No nodes in cascade."}), 422

        result = simulate_with_steps(
            G=G,
            partition=sbm.partition,
            root=root,
            sbm=sbm,
            alpha=alpha,
            lam=lam,
            global_class_sizes=sbm.class_sizes.astype(float),
            seed=seed,
        )
        result["cascade_id"] = cascade_id
        result["mock"] = False
        return jsonify(result)

    except Exception as exc:
        log.exception("simulate error")
        return jsonify({"error": "Simulation failed", "code": 500,
                        "detail": str(exc)}), 500

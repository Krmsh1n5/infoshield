"""
api/routes/monitoring.py
========================
GET /api/v1/live_cascades
GET /api/v1/sbm/matrices

live_cascades: returns active cascades from Telegram monitor (or mock data).
sbm/matrices:  returns fitted SBM b+ / b- matrices.
"""

from __future__ import annotations

import datetime
import logging
import random

from flask import Blueprint, current_app, jsonify

log = logging.getLogger(__name__)

monitoring_bp = Blueprint("monitoring", __name__)


# ---------------------------------------------------------------------------
# Mock cascades for demo mode
# ---------------------------------------------------------------------------

_MOCK_LABELS = ["false", "false", "true", "unverified"]
_MOCK_PATTERNS = ["wide_burst", "deep_chain", "slow_diffusion"]

def _generate_mock_cascades(n: int = 5) -> list[dict]:
    rng = random.Random(42)
    results = []
    base_time = datetime.datetime.utcnow()
    for i in range(n):
        label = _MOCK_LABELS[i % len(_MOCK_LABELS)]
        started = (base_time - datetime.timedelta(minutes=rng.randint(1, 120))).isoformat() + "Z"
        results.append({
            "id": f"mock-cascade-{i+1:04d}",
            "nodes": rng.randint(5, 150),
            "edges": rng.randint(4, 200),
            "label": label,
            "binary_label": "false" if label in ("false", "unverified") else "true",
            "confidence": round(rng.uniform(0.55, 0.97), 3),
            "pattern": _MOCK_PATTERNS[i % len(_MOCK_PATTERNS)],
            "started": started,
            "platform": "telegram",
            "mock": True,
        })
    return results


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@monitoring_bp.route("/api/v1/live_cascades", methods=["GET"])
def live_cascades():
    monitor = current_app.extensions.get("telegram_monitor")

    if monitor is None:
        log.debug("Telegram monitor not available — returning mock cascades")
        return jsonify(_generate_mock_cascades())

    try:
        cascades = monitor.get_active_cascades()
        return jsonify(cascades)
    except Exception as exc:
        log.warning("Monitor error: %s — returning mock", exc)
        return jsonify(_generate_mock_cascades())


@monitoring_bp.route("/api/v1/sbm/matrices", methods=["GET"])
def sbm_matrices():
    sbm = current_app.extensions.get("sbm")

    if sbm is None:
        # Return dummy matrices so the dashboard can still render
        k = 4
        import random as _r
        rng = _r.Random(0)
        b_plus = [[round(rng.uniform(1e-5, 5e-4), 8) for _ in range(k)] for _ in range(k)]
        b_minus = [[round(rng.uniform(1e-5, 5e-4), 8) for _ in range(k)] for _ in range(k)]
        return jsonify({
            "b_plus": b_plus,
            "b_minus": b_minus,
            "k": k,
            "class_sizes": [38055, 55057, 8551, 12572],
            "mock": True,
        })

    try:
        return jsonify({
            "b_plus": sbm.b_plus.tolist(),
            "b_minus": sbm.b_minus.tolist(),
            "k": sbm.k,
            "class_sizes": sbm.class_sizes.tolist(),
            "mock": False,
        })
    except Exception as exc:
        log.exception("sbm_matrices error")
        return jsonify({"error": "Failed to read SBM", "code": 500,
                        "detail": str(exc)}), 500

"""
api/server.py
=============
InfoShield Flask API server.

Startup sequence
----------------
1. Load SBM from data/processed/sbm_matrices/
2. Load BiGCN Predictor (fold=0, split=twitter15)
   — falls back to mock classifier if checkpoint absent
3. Load CounterNarrativeGenerator + PostFormatter
   — falls back to mock if ANTHROPIC_API_KEY absent
4. Start Telegram monitor in background thread
   — runs in mock mode if credentials absent
5. Register all route blueprints
6. Start Flask on --port (default 5000)

Launch
------
    python -m api.server                       # live mode
    python -m api.server --mock                # all mocks
    python -m api.server --port 8080
    python -m api.server --telegram-channels "@azertag,@aztv_official"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
from pathlib import Path

# ── Make project root importable ──────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

from api.routes.classify import classify_bp
from api.routes.simulate import simulate_bp
from api.routes.rebuttal import rebuttal_bp
from api.routes.monitoring import monitoring_bp

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Component loaders — each returns None on failure (never raise)
# ---------------------------------------------------------------------------

def _load_sbm() -> object | None:
    try:
        import numpy as np
        from graph_engine.network_model import SBM

        sbm_dir = ROOT / "data" / "processed" / "sbm_matrices"
        b_plus_path  = sbm_dir / "b_plus.npy"
        b_minus_path = sbm_dir / "b_minus.npy"

        if not b_plus_path.exists() or not b_minus_path.exists():
            log.warning("SBM matrices not found in %s — SBM disabled", sbm_dir)
            return None

        b_plus  = np.load(str(b_plus_path))
        b_minus = np.load(str(b_minus_path))
        k = b_plus.shape[0]

        # Try to load partition and class sizes
        partition_path   = sbm_dir / "partition.npy"
        class_sizes_path = sbm_dir / "class_sizes.npy"

        if partition_path.exists():
            import pickle
            with open(partition_path, "rb") as f:
                partition = pickle.load(f)
        else:
            partition = {}

        if class_sizes_path.exists():
            class_sizes = np.load(str(class_sizes_path))
        else:
            # Use paper-reported values as fallback
            class_sizes = np.array([38055, 55057, 8551, 12572], dtype=float)
            if k != 4:
                class_sizes = np.ones(k) * (114235 / k)

        sbm = SBM(
            b_plus=b_plus,
            b_minus=b_minus,
            k=k,
            partition=partition,
            class_sizes=class_sizes,
        )
        log.info("SBM loaded: k=%d, partition nodes=%d", k, len(partition))
        return sbm

    except Exception as exc:
        log.warning("SBM load failed: %s — SBM disabled", exc)
        return None


def _load_predictor() -> object | None:
    try:
        from gnn.predict import Predictor
        predictor = Predictor(fold=0, split="twitter15")
        # Trigger model load to catch missing checkpoint early
        predictor._load_model()
        log.info("BiGCN Predictor loaded (fold=0, twitter15)")
        return predictor
    except FileNotFoundError as exc:
        log.warning("BiGCN checkpoint not found: %s — using mock classifier", exc)
        return None
    except Exception as exc:
        log.warning("BiGCN load failed: %s — using mock classifier", exc)
        return None


def _load_counter_narrative_generator() -> tuple[object | None, object | None]:
    """Returns (generator, formatter) or (None, None) on failure."""
    try:
        from pipeline.counter_narrative.generator import CounterNarrativeGenerator
        from pipeline.counter_narrative.formatter import PostFormatter

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            log.warning("ANTHROPIC_API_KEY not set — counter-narrative generator disabled")
            return None, None

        sources_dir = ROOT / "pipeline" / "counter_narrative" / "sources"
        generator = CounterNarrativeGenerator(sources_dir=sources_dir, api_key=api_key)
        formatter = PostFormatter()
        log.info("CounterNarrativeGenerator loaded")
        return generator, formatter

    except Exception as exc:
        log.warning("CounterNarrativeGenerator load failed: %s — disabled", exc)
        return None, None


def _start_telegram_monitor(
    channels: list[str],
    mock: bool,
    app: Flask,
) -> object | None:
    """Start Telegram monitor in a background daemon thread."""
    try:
        from pipeline.ingestors.telegram_ingestor import TelegramMonitor  # type: ignore

        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if not token and not mock:
            log.warning("TELEGRAM_BOT_TOKEN not set — Telegram monitor in mock mode")
            mock = True

        monitor = TelegramMonitor(channels=channels, token=token, mock=mock)

        def _run():
            try:
                monitor.start()
            except Exception as exc:
                log.warning("Telegram monitor thread error: %s", exc)

        t = threading.Thread(target=_run, name="telegram-monitor", daemon=True)
        t.start()
        log.info("Telegram monitor started (mock=%s, channels=%s)", mock, channels)
        return monitor

    except ImportError:
        log.warning("TelegramMonitor not found — monitoring disabled")
        return None
    except Exception as exc:
        log.warning("Telegram monitor startup error: %s — disabled", exc)
        return None


# ---------------------------------------------------------------------------
# WhatsApp webhook (Twilio)
# ---------------------------------------------------------------------------

def _register_whatsapp_webhook(app: Flask) -> None:
    try:
        from pipeline.whatsapp_bot import whatsapp_bp  # type: ignore
        app.register_blueprint(whatsapp_bp)
        log.info("WhatsApp webhook registered at /whatsapp")
    except ImportError:
        log.debug("whatsapp_bot.py not found — /whatsapp endpoint not registered")
    except Exception as exc:
        log.warning("WhatsApp webhook registration failed: %s", exc)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(
    mock: bool = False,
    telegram_channels: list[str] | None = None,
) -> Flask:
    app = Flask(
        __name__,
        static_folder=str(ROOT / "dashboard"),
        static_url_path="",
    )
    CORS(app)

    # ── Extension slot (avoids circular import via current_app.extensions) ──
    app.extensions["sbm"] = None
    app.extensions["predictor"] = None
    app.extensions["counter_narrative_generator"] = None
    app.extensions["post_formatter"] = None
    app.extensions["telegram_monitor"] = None
    app.extensions["cascade_store"] = {}    # cascade_id → nx.DiGraph

    # ── Load components ──────────────────────────────────────────────────────
    if not mock:
        app.extensions["sbm"] = _load_sbm()
        app.extensions["predictor"] = _load_predictor()
        gen, fmt = _load_counter_narrative_generator()
        app.extensions["counter_narrative_generator"] = gen
        app.extensions["post_formatter"] = fmt
    else:
        log.info("Mock mode enabled — all real components skipped")

    channels = telegram_channels or []
    app.extensions["telegram_monitor"] = _start_telegram_monitor(
        channels=channels,
        mock=mock,
        app=app,
    )

    # ── Register blueprints ──────────────────────────────────────────────────
    app.register_blueprint(classify_bp)
    app.register_blueprint(simulate_bp)
    app.register_blueprint(rebuttal_bp)
    app.register_blueprint(monitoring_bp)
    _register_whatsapp_webhook(app)

    # ── Dashboard static route ────────────────────────────────────────────────
    @app.route("/")
    def index():
        dashboard_dir = ROOT / "dashboard"
        if (dashboard_dir / "index.html").exists():
            return send_from_directory(str(dashboard_dir), "index.html")
        return jsonify({"status": "InfoShield API running", "version": "1.0.0"})

    # ── Health check ─────────────────────────────────────────────────────────
    @app.route("/api/v1/health", methods=["GET"])
    def health():
        return jsonify({
            "status": "ok",
            "components": {
                "sbm":                  app.extensions["sbm"] is not None,
                "predictor":            app.extensions["predictor"] is not None,
                "counter_narrative":    app.extensions["counter_narrative_generator"] is not None,
                "telegram_monitor":     app.extensions["telegram_monitor"] is not None,
            },
            "mock_mode": mock,
        })

    # ── Global error handlers ────────────────────────────────────────────────
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found", "code": 404,
                        "detail": str(e)}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        return jsonify({"error": "Method not allowed", "code": 405,
                        "detail": str(e)}), 405

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal server error", "code": 500,
                        "detail": str(e)}), 500

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="InfoShield API server")
    parser.add_argument("--mock",    action="store_true",
                        help="Run all ingestors in mock mode (no real credentials needed)")
    parser.add_argument("--port",    type=int, default=5000,
                        help="Port to listen on (default: 5000)")
    parser.add_argument("--host",    default="0.0.0.0",
                        help="Host to bind (default: 0.0.0.0)")
    parser.add_argument("--telegram-channels", default="",
                        help="Comma-separated Telegram channel names, e.g. '@azertag,@aztv_official'")
    args = parser.parse_args()

    channels = [c.strip() for c in args.telegram_channels.split(",") if c.strip()]

    app = create_app(mock=args.mock, telegram_channels=channels)

    log.info(
        "Starting InfoShield API on %s:%d (mock=%s, channels=%s)",
        args.host, args.port, args.mock, channels or "none",
    )
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()

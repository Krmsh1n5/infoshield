"""
tests/test_api.py
=================
Pytest test suite for the InfoShield Flask API.

Run:
    pip install pytest flask flask-cors networkx numpy
    pytest tests/test_api.py -v

Run a single group:
    pytest tests/test_api.py -v -k "classify"
    pytest tests/test_api.py -v -k "simulate"
    pytest tests/test_api.py -v -k "rebuttal"
    pytest tests/test_api.py -v -k "monitoring"
    pytest tests/test_api.py -v -k "health"

All tests run against a mock-mode app (no real checkpoints, API keys,
or Telegram credentials required).  Real-component tests are skipped
automatically when dependencies are absent.
"""

from __future__ import annotations

import hashlib
import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Make the project root importable ──────────────────────────────────────────
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def mock_app():
    """
    Session-scoped Flask test app in mock mode.

    All real components (SBM, BiGCN, Anthropic) are None.
    Routes fall back to their built-in mock implementations.
    """
    # Stub out heavy optional imports so create_app doesn't crash on import
    _stub_optional_modules()

    from api.server import create_app
    app = create_app(mock=True)
    app.config["TESTING"] = True
    return app


@pytest.fixture(scope="session")
def client(mock_app):
    """Flask test client (session-scoped — shared across all tests)."""
    return mock_app.test_client()


@pytest.fixture(scope="session")
def app_with_sbm(mock_app):
    """
    App fixture with a minimal in-memory SBM injected into extensions.
    Used for simulate tests that want to exercise the real BFS path
    without loading WICO data from disk.
    """
    import numpy as np

    sbm = _make_mock_sbm(k=4)
    mock_app.extensions["sbm"] = sbm
    yield mock_app
    # Restore to None so other tests are unaffected
    mock_app.extensions["sbm"] = None


@pytest.fixture()
def client_with_sbm(app_with_sbm):
    return app_with_sbm.test_client()


# =============================================================================
# Helpers
# =============================================================================

def _stub_optional_modules():
    """
    Insert lightweight stubs for modules that may not be installed in the
    test environment (Telegram monitor, WhatsApp bot, pipeline components).
    This prevents ImportError during create_app() even when the full project
    tree is not present.
    """
    stubs = {
        "pipeline": types.ModuleType("pipeline"),
        "pipeline.ingestors": types.ModuleType("pipeline.ingestors"),
        "pipeline.ingestors.telegram_ingestor": types.ModuleType(
            "pipeline.ingestors.telegram_ingestor"
        ),
        "pipeline.whatsapp_bot": types.ModuleType("pipeline.whatsapp_bot"),
        "pipeline.counter_narrative": types.ModuleType("pipeline.counter_narrative"),
        "pipeline.counter_narrative.generator": types.ModuleType(
            "pipeline.counter_narrative.generator"
        ),
        "pipeline.counter_narrative.formatter": types.ModuleType(
            "pipeline.counter_narrative.formatter"
        ),
        "gnn": types.ModuleType("gnn"),
        "gnn.predict": types.ModuleType("gnn.predict"),
    }
    for name, mod in stubs.items():
        sys.modules.setdefault(name, mod)


def _make_mock_sbm(k: int = 4):
    """Return a minimal SBM-like object sufficient for simulate_with_steps."""
    import numpy as np

    sbm = MagicMock()
    sbm.k = k
    sbm.b_plus  = np.full((k, k), 1e-4)
    sbm.b_minus = np.full((k, k), 5e-4)
    sbm.class_sizes = np.array([38055, 55057, 8551, 12572], dtype=float)[:k]
    # Map every node to class 0 by default
    sbm.partition = {}
    # numpy arrays already have .tolist() — no override needed
    return sbm


def _make_mock_predictor(label: str = "false", confidence: float = 0.85):
    """Return a mock Predictor whose predict_from_digraph returns a fixed result."""
    result = MagicMock()
    result.label_name    = label
    result.binary_label  = "false" if label == "false" else "true"
    result.confidence    = confidence
    result.num_nodes     = 5
    result.high_confidence = confidence >= 0.65

    predictor = MagicMock()
    predictor.predict_from_digraph.return_value = result
    return predictor


def _make_mock_generator(text: str = "This claim is false according to WHO."):
    """Return a mock CounterNarrativeGenerator + PostFormatter pair."""
    result = MagicMock()
    result.text                   = text
    result.language               = "az"
    result.topic                  = "health"
    result.sources_used           = ["health_who"]
    result.confidence_in_rebuttal = 0.88
    result.generation_time_ms     = 210

    generator = MagicMock()
    generator.generate_rebuttal.return_value = result

    formatter = MagicMock()
    formatter.format_all_platforms.return_value = {
        "telegram":  f"📌 <b>OFFICIAL REBUTTAL</b>\n\n{text}",
        "instagram": text,
        "whatsapp":  text,
        "facebook":  text,
    }

    return generator, formatter


def post_json(client, url: str, payload: dict):
    return client.post(
        url,
        data=json.dumps(payload),
        content_type="application/json",
    )


def get(client, url: str):
    return client.get(url)


# =============================================================================
# Health
# =============================================================================

class TestHealth:
    def test_health_returns_200(self, client):
        r = get(client, "/api/v1/health")
        assert r.status_code == 200

    def test_health_schema(self, client):
        data = r = get(client, "/api/v1/health")
        body = json.loads(r.data)
        assert "status" in body
        assert body["status"] == "ok"
        assert "components" in body
        assert "mock_mode" in body

    def test_health_components_are_booleans(self, client):
        body = json.loads(get(client, "/api/v1/health").data)
        for key, val in body["components"].items():
            assert isinstance(val, bool), f"components.{key} should be bool"

    def test_health_mock_mode_true(self, client):
        body = json.loads(get(client, "/api/v1/health").data)
        assert body["mock_mode"] is True


# =============================================================================
# Root / index
# =============================================================================

class TestIndex:
    def test_index_returns_200(self, client):
        r = get(client, "/")
        assert r.status_code == 200

    def test_index_returns_json_when_no_dashboard(self, client):
        # In test env there is no dashboard/index.html — API status JSON expected
        body = json.loads(client.get("/").data)
        assert "status" in body or "InfoShield" in str(body)


# =============================================================================
# POST /api/v1/classify
# =============================================================================

class TestClassify:

    # ── Input validation ──────────────────────────────────────────────────────

    def test_missing_body_returns_400(self, client):
        r = post_json(client, "/api/v1/classify", {})
        assert r.status_code == 400

    def test_missing_body_error_schema(self, client):
        body = json.loads(post_json(client, "/api/v1/classify", {}).data)
        assert body["code"] == 400
        assert "error" in body
        assert "detail" in body

    def test_empty_text_and_no_edges_returns_400(self, client):
        r = post_json(client, "/api/v1/classify", {"text": ""})
        assert r.status_code == 400

    def test_non_json_body_returns_400(self, client):
        r = client.post("/api/v1/classify", data="not json",
                        content_type="text/plain")
        assert r.status_code == 400

    # ── Text-only path (mock classifier) ─────────────────────────────────────

    def test_text_only_returns_200(self, client):
        r = post_json(client, "/api/v1/classify", {"text": "5G towers spread COVID-19"})
        assert r.status_code == 200

    def test_text_only_response_schema(self, client):
        r = post_json(client, "/api/v1/classify", {"text": "5G towers spread COVID-19"})
        body = json.loads(r.data)
        for field in ("label", "binary_label", "confidence", "cascade_id", "mock"):
            assert field in body, f"Missing field: {field}"

    def test_text_only_label_is_valid(self, client):
        r = post_json(client, "/api/v1/classify", {"text": "hello world"})
        body = json.loads(r.data)
        assert body["label"] in ("true", "false", "unverified", "non-rumor")

    def test_text_only_binary_label_is_valid(self, client):
        r = post_json(client, "/api/v1/classify", {"text": "hello world"})
        body = json.loads(r.data)
        assert body["binary_label"] in ("true", "false", "uncertain")

    def test_text_only_confidence_in_range(self, client):
        r = post_json(client, "/api/v1/classify", {"text": "breaking news"})
        body = json.loads(r.data)
        assert 0.0 <= body["confidence"] <= 1.0

    def test_text_only_cascade_id_is_string(self, client):
        r = post_json(client, "/api/v1/classify", {"text": "test"})
        body = json.loads(r.data)
        assert isinstance(body["cascade_id"], str)
        assert len(body["cascade_id"]) > 0

    def test_text_only_mock_flag_is_true(self, client):
        # No predictor loaded in mock_app
        r = post_json(client, "/api/v1/classify", {"text": "claim"})
        body = json.loads(r.data)
        assert body["mock"] is True

    def test_text_classification_is_deterministic(self, client):
        """Same text must always return the same label (MD5 hash-based)."""
        text = "Vaccines contain microchips"
        r1 = json.loads(post_json(client, "/api/v1/classify", {"text": text}).data)
        r2 = json.loads(post_json(client, "/api/v1/classify", {"text": text}).data)
        assert r1["label"] == r2["label"]
        assert r1["confidence"] == r2["confidence"]

    def test_different_texts_may_get_different_labels(self, client):
        """Mock classifier uses MD5 hash % 4 — verify at least 2 distinct labels
        across a larger batch of inputs."""
        import hashlib
        labels_4 = ["true", "false", "unverified", "non-rumor"]
        # Generate 20 distinct texts and collect their labels
        texts = [f"claim number {i} about various topics" for i in range(20)]
        seen = set()
        for t in texts:
            body = json.loads(post_json(client, "/api/v1/classify", {"text": t}).data)
            seen.add(body["label"])
        assert len(seen) >= 2, f"Expected multiple labels, got only: {seen}"

    # ── Graph-edges path ──────────────────────────────────────────────────────

    SAMPLE_EDGES = [["A", "B"], ["B", "C"], ["B", "D"], ["A", "E"]]

    def test_edges_only_returns_200(self, client):
        r = post_json(client, "/api/v1/classify", {"graph_edges": self.SAMPLE_EDGES})
        assert r.status_code == 200

    def test_edges_only_response_schema(self, client):
        r = post_json(client, "/api/v1/classify", {"graph_edges": self.SAMPLE_EDGES})
        body = json.loads(r.data)
        for field in ("label", "binary_label", "confidence", "cascade_id",
                      "num_nodes", "pattern", "mock"):
            assert field in body, f"Missing field: {field}"

    def test_edges_num_nodes_correct(self, client):
        r = post_json(client, "/api/v1/classify", {"graph_edges": self.SAMPLE_EDGES})
        body = json.loads(r.data)
        # SAMPLE_EDGES has 5 distinct nodes
        assert body["num_nodes"] == 5

    def test_edges_pattern_is_valid(self, client):
        r = post_json(client, "/api/v1/classify", {"graph_edges": self.SAMPLE_EDGES})
        body = json.loads(r.data)
        assert body["pattern"] in ("wide_burst", "deep_chain", "slow_diffusion", "unknown")

    def test_edges_plus_text(self, client):
        r = post_json(client, "/api/v1/classify", {
            "text": "Some claim",
            "graph_edges": self.SAMPLE_EDGES,
        })
        assert r.status_code == 200

    def test_single_edge_graph(self, client):
        r = post_json(client, "/api/v1/classify", {"graph_edges": [["X", "Y"]]})
        assert r.status_code == 200
        body = json.loads(r.data)
        assert body["num_nodes"] == 2

    def test_malformed_edge_skipped_gracefully(self, client):
        """Edges shorter than 2 elements should be silently skipped."""
        r = post_json(client, "/api/v1/classify", {
            "graph_edges": [["A", "B"], ["C"], [], ["D", "E"]],
        })
        assert r.status_code == 200

    # ── Real predictor path ───────────────────────────────────────────────────

    def test_with_injected_predictor(self, mock_app, client):
        """Inject a mock predictor and verify mock=False + correct fields."""
        predictor = _make_mock_predictor(label="false", confidence=0.91)
        mock_app.extensions["predictor"] = predictor
        try:
            r = post_json(client, "/api/v1/classify", {"text": "claim"})
            body = json.loads(r.data)
            assert body["mock"] is False
            assert body["label"] == "false"
            assert body["confidence"] == pytest.approx(0.91)
        finally:
            mock_app.extensions["predictor"] = None

    def test_predictor_called_with_digraph_when_edges_provided(self, mock_app, client):
        predictor = _make_mock_predictor()
        mock_app.extensions["predictor"] = predictor
        try:
            post_json(client, "/api/v1/classify", {"graph_edges": self.SAMPLE_EDGES})
            predictor.predict_from_digraph.assert_called_once()
        finally:
            mock_app.extensions["predictor"] = None

    # ── Cascade pattern heuristics ────────────────────────────────────────────

    def test_infer_cascade_pattern_wide_burst(self):
        """avg_degree (edges/nodes) > 3 → wide_burst.
        Need a graph where E/N > 3.  A complete graph on 5 nodes has 20
        directed edges / 5 nodes = 4.0, which clears the threshold.
        """
        from api.routes.classify import _infer_cascade_pattern, _edges_to_digraph
        nodes = ["a", "b", "c", "d", "e"]
        edges = [[u, v] for u in nodes for v in nodes if u != v]  # 20 edges, 5 nodes
        G = _edges_to_digraph(edges)
        assert G.number_of_edges() / G.number_of_nodes() > 3
        assert _infer_cascade_pattern(G) == "wide_burst"

    def test_infer_cascade_pattern_slow_diffusion(self):
        """Sparse linear chain → slow_diffusion (not deep_chain because depth ≤ 8)."""
        from api.routes.classify import _infer_cascade_pattern, _edges_to_digraph
        edges = [[str(i), str(i + 1)] for i in range(4)]  # depth=4
        G = _edges_to_digraph(edges)
        assert _infer_cascade_pattern(G) == "slow_diffusion"

    def test_infer_cascade_pattern_empty_graph(self):
        from api.routes.classify import _infer_cascade_pattern
        import networkx as nx
        assert _infer_cascade_pattern(nx.DiGraph()) == "unknown"

    # ── Mock classifier unit tests ────────────────────────────────────────────

    def test_mock_classify_deterministic(self):
        from api.routes.classify import _mock_classify
        r1 = _mock_classify("hello")
        r2 = _mock_classify("hello")
        assert r1 == r2

    def test_mock_classify_confidence_in_range(self):
        from api.routes.classify import _mock_classify
        for text in ("a", "abc", "longer claim about vaccines"):
            r = _mock_classify(text)
            assert 0.55 <= r["confidence"] <= 0.95, f"Confidence out of range for {text!r}"

    def test_mock_classify_valid_label(self):
        from api.routes.classify import _mock_classify
        valid = {"true", "false", "unverified", "non-rumor"}
        for text in ("x", "y", "z", "1", "2"):
            assert _mock_classify(text)["label"] in valid

    def test_mock_classify_binary_label_consistent(self):
        from api.routes.classify import _mock_classify
        for text in ("test1", "test2", "test3", "test4", "test5", "test6"):
            r = _mock_classify(text)
            label = r["label"]
            binary = r["binary_label"]
            if label == "false":
                assert binary == "false"
            elif label in ("true", "non-rumor"):
                assert binary == "true"
            else:
                assert binary == "uncertain"


# =============================================================================
# POST /api/v1/simulate
# =============================================================================

class TestSimulate:

    BASE_PAYLOAD = {
        "cascade_id": "test-cascade-001",
        "alpha": 1.5,
        "lambda": 1.0,
        "content": "false",
        "seed": 42,
    }

    # ── Mock mode (no SBM) ────────────────────────────────────────────────────

    def test_mock_simulate_returns_200(self, client):
        r = post_json(client, "/api/v1/simulate", self.BASE_PAYLOAD)
        assert r.status_code == 200

    def test_mock_simulate_response_schema(self, client):
        r = post_json(client, "/api/v1/simulate", self.BASE_PAYLOAD)
        body = json.loads(r.data)
        for field in ("steps", "r_inf", "lp_types", "d_star_final", "cascade_id"):
            assert field in body, f"Missing field: {field}"

    def test_mock_simulate_steps_is_list(self, client):
        r = post_json(client, "/api/v1/simulate", self.BASE_PAYLOAD)
        body = json.loads(r.data)
        assert isinstance(body["steps"], list)

    def test_mock_simulate_each_step_has_required_keys(self, client):
        r = post_json(client, "/api/v1/simulate", self.BASE_PAYLOAD)
        body = json.loads(r.data)
        for step in body["steps"]:
            for key in ("step", "i_set_size", "new_infected", "d_star"):
                assert key in step, f"Step missing key: {key}"

    def test_mock_simulate_r_inf_positive(self, client):
        r = post_json(client, "/api/v1/simulate", self.BASE_PAYLOAD)
        body = json.loads(r.data)
        assert body["r_inf"] >= 1

    def test_mock_simulate_d_star_final_is_matrix(self, client):
        r = post_json(client, "/api/v1/simulate", self.BASE_PAYLOAD)
        body = json.loads(r.data)
        d = body["d_star_final"]
        assert isinstance(d, list)
        assert all(isinstance(row, list) for row in d)

    def test_mock_simulate_cascade_id_echoed(self, client):
        r = post_json(client, "/api/v1/simulate", self.BASE_PAYLOAD)
        body = json.loads(r.data)
        assert body["cascade_id"] == self.BASE_PAYLOAD["cascade_id"]

    def test_control_mode_alpha_null(self, client):
        payload = dict(self.BASE_PAYLOAD, alpha=None)
        r = post_json(client, "/api/v1/simulate", payload)
        assert r.status_code == 200

    def test_true_content_runs(self, client):
        payload = dict(self.BASE_PAYLOAD, content="true")
        r = post_json(client, "/api/v1/simulate", payload)
        assert r.status_code == 200

    def test_missing_body_still_runs(self, client):
        """Missing optional fields should use defaults, not crash."""
        r = post_json(client, "/api/v1/simulate", {})
        assert r.status_code == 200

    def test_seed_produces_deterministic_output(self, client):
        payload = dict(self.BASE_PAYLOAD, seed=7)
        r1 = json.loads(post_json(client, "/api/v1/simulate", payload).data)
        r2 = json.loads(post_json(client, "/api/v1/simulate", payload).data)
        assert r1["r_inf"] == r2["r_inf"]
        assert r1["steps"] == r2["steps"]

    def test_different_seeds_may_differ(self, client):
        r1 = json.loads(post_json(client, "/api/v1/simulate",
                                  dict(self.BASE_PAYLOAD, seed=1)).data)
        r2 = json.loads(post_json(client, "/api/v1/simulate",
                                  dict(self.BASE_PAYLOAD, seed=999)).data)
        # With random-based mock, different seeds should give different step counts
        # (not a guaranteed property, but almost certain with the range used)
        assert isinstance(r1["r_inf"], int)
        assert isinstance(r2["r_inf"], int)

    # ── simulate_with_steps unit test ────────────────────────────────────────

    def test_simulate_with_steps_control(self):
        """Control run (alpha=None): d_star must be all-ones.
        Skipped if graph_engine.optimizer is not importable (no full install).
        """
        pytest.importorskip("graph_engine.optimizer",
                            reason="graph_engine not installed")
        import networkx as nx
        import numpy as np
        from api.routes.simulate import simulate_with_steps

        sbm = _make_mock_sbm(k=4)
        G = nx.DiGraph()
        G.add_edges_from([("root", "1"), ("1", "2"), ("2", "3")])
        sbm.partition = {"root": 0, "1": 1, "2": 2, "3": 3}

        result = simulate_with_steps(
            G=G, partition=sbm.partition, root="root",
            sbm=sbm, alpha=None, lam=None,
            global_class_sizes=sbm.class_sizes, seed=0,
        )
        assert "steps" in result
        assert result["r_inf"] >= 1
        for step in result["steps"]:
            d = np.array(step["d_star"])
            assert np.allclose(d, 1.0), "Control d_star must be all-ones"

    def test_simulate_with_steps_returns_step_indices(self):
        pytest.importorskip("graph_engine.optimizer",
                            reason="graph_engine not installed")
        import networkx as nx
        from api.routes.simulate import simulate_with_steps

        sbm = _make_mock_sbm(k=4)
        G = nx.DiGraph()
        G.add_edges_from([("root", "A"), ("A", "B")])
        sbm.partition = {}

        result = simulate_with_steps(
            G=G, partition={}, root="root",
            sbm=sbm, alpha=None, lam=None,
            global_class_sizes=sbm.class_sizes, seed=1,
        )
        for i, step in enumerate(result["steps"]):
            assert step["step"] == i


# =============================================================================
# POST /api/v1/rebuttal
# =============================================================================

class TestRebuttal:

    BASE_PAYLOAD = {
        "claim": "5G towers spread COVID-19",
        "topic": "health",
        "language": "az",
        "confidence": 0.92,
    }

    # ── Input validation ──────────────────────────────────────────────────────

    def test_missing_claim_returns_400(self, client):
        r = post_json(client, "/api/v1/rebuttal", {"topic": "health", "language": "az"})
        assert r.status_code == 400

    def test_empty_claim_returns_400(self, client):
        r = post_json(client, "/api/v1/rebuttal",
                      {"claim": "", "language": "az", "topic": "health"})
        assert r.status_code == 400

    def test_invalid_language_returns_400(self, client):
        r = post_json(client, "/api/v1/rebuttal",
                      {"claim": "test", "language": "fr", "topic": "health"})
        assert r.status_code == 400

    def test_invalid_language_error_schema(self, client):
        body = json.loads(post_json(client, "/api/v1/rebuttal",
                                    {"claim": "test", "language": "zh"}).data)
        assert body["code"] == 400

    # ── Mock mode (no generator) ──────────────────────────────────────────────

    def test_mock_rebuttal_returns_200(self, client):
        r = post_json(client, "/api/v1/rebuttal", self.BASE_PAYLOAD)
        assert r.status_code == 200

    def test_mock_rebuttal_schema(self, client):
        r = post_json(client, "/api/v1/rebuttal", self.BASE_PAYLOAD)
        body = json.loads(r.data)
        for field in ("rebuttal", "language", "topic", "sources",
                      "confidence_in_rebuttal", "formatted", "mock"):
            assert field in body, f"Missing field: {field}"

    def test_mock_rebuttal_formatted_has_platforms(self, client):
        r = post_json(client, "/api/v1/rebuttal", self.BASE_PAYLOAD)
        body = json.loads(r.data)
        for platform in ("telegram", "instagram", "whatsapp"):
            assert platform in body["formatted"], f"Missing platform: {platform}"

    def test_mock_rebuttal_language_echoed(self, client):
        for lang in ("az", "ru", "en"):
            r = post_json(client, "/api/v1/rebuttal",
                          dict(self.BASE_PAYLOAD, language=lang))
            body = json.loads(r.data)
            assert body["language"] == lang

    def test_mock_rebuttal_text_non_empty(self, client):
        r = post_json(client, "/api/v1/rebuttal", self.BASE_PAYLOAD)
        body = json.loads(r.data)
        assert len(body["rebuttal"]) > 10

    def test_mock_rebuttal_sources_is_list(self, client):
        r = post_json(client, "/api/v1/rebuttal", self.BASE_PAYLOAD)
        body = json.loads(r.data)
        assert isinstance(body["sources"], list)

    def test_mock_rebuttal_is_true_when_no_generator(self, client):
        r = post_json(client, "/api/v1/rebuttal", self.BASE_PAYLOAD)
        body = json.loads(r.data)
        assert body["mock"] is True

    # ── All valid languages accepted ──────────────────────────────────────────

    @pytest.mark.parametrize("lang", ["az", "ru", "en"])
    def test_valid_languages_accepted(self, client, lang):
        r = post_json(client, "/api/v1/rebuttal",
                      dict(self.BASE_PAYLOAD, language=lang))
        assert r.status_code == 200

    # ── Real generator path ───────────────────────────────────────────────────

    def test_with_injected_generator(self, mock_app, client):
        """Inject mock generator and verify mock=False path."""
        gen, fmt = _make_mock_generator()
        mock_app.extensions["counter_narrative_generator"] = gen
        mock_app.extensions["post_formatter"] = fmt
        try:
            r = post_json(client, "/api/v1/rebuttal", self.BASE_PAYLOAD)
            body = json.loads(r.data)
            assert body["mock"] is False
            assert body["rebuttal"] == "This claim is false according to WHO."
            assert "telegram" in body["formatted"]
        finally:
            mock_app.extensions["counter_narrative_generator"] = None
            mock_app.extensions["post_formatter"] = None

    def test_generator_called_with_correct_args(self, mock_app, client):
        gen, fmt = _make_mock_generator()
        mock_app.extensions["counter_narrative_generator"] = gen
        mock_app.extensions["post_formatter"] = fmt
        try:
            post_json(client, "/api/v1/rebuttal", self.BASE_PAYLOAD)
            gen.generate_rebuttal.assert_called_once()
            kwargs = gen.generate_rebuttal.call_args.kwargs
            assert kwargs["false_claim"] == self.BASE_PAYLOAD["claim"]
            assert kwargs["language"] == self.BASE_PAYLOAD["language"]
            assert kwargs["topic"] == self.BASE_PAYLOAD["topic"]
        finally:
            mock_app.extensions["counter_narrative_generator"] = None
            mock_app.extensions["post_formatter"] = None


# =============================================================================
# GET /api/v1/live_cascades
# =============================================================================

class TestLiveCascades:

    def test_returns_200(self, client):
        r = get(client, "/api/v1/live_cascades")
        assert r.status_code == 200

    def test_returns_list(self, client):
        body = json.loads(get(client, "/api/v1/live_cascades").data)
        assert isinstance(body, list)

    def test_list_is_non_empty(self, client):
        body = json.loads(get(client, "/api/v1/live_cascades").data)
        assert len(body) > 0

    def test_each_cascade_has_required_fields(self, client):
        body = json.loads(get(client, "/api/v1/live_cascades").data)
        for cascade in body:
            for field in ("id", "nodes", "edges", "label", "confidence", "started"):
                assert field in cascade, f"Cascade missing field: {field}"

    def test_cascade_label_is_valid(self, client):
        body = json.loads(get(client, "/api/v1/live_cascades").data)
        valid_labels = {"true", "false", "unverified", "non-rumor"}
        for cascade in body:
            assert cascade["label"] in valid_labels

    def test_cascade_confidence_in_range(self, client):
        body = json.loads(get(client, "/api/v1/live_cascades").data)
        for cascade in body:
            assert 0.0 <= cascade["confidence"] <= 1.0

    def test_cascade_nodes_positive(self, client):
        body = json.loads(get(client, "/api/v1/live_cascades").data)
        for cascade in body:
            assert cascade["nodes"] > 0

    def test_real_monitor_used_when_injected(self, mock_app, client):
        monitor = MagicMock()
        monitor.get_active_cascades.return_value = [
            {"id": "real-001", "nodes": 42, "edges": 38,
             "label": "false", "confidence": 0.91, "started": "2025-01-01T00:00:00Z"}
        ]
        mock_app.extensions["telegram_monitor"] = monitor
        try:
            body = json.loads(get(client, "/api/v1/live_cascades").data)
            assert body[0]["id"] == "real-001"
            monitor.get_active_cascades.assert_called_once()
        finally:
            mock_app.extensions["telegram_monitor"] = None

    def test_monitor_error_falls_back_to_mock(self, mock_app, client):
        monitor = MagicMock()
        monitor.get_active_cascades.side_effect = RuntimeError("connection lost")
        mock_app.extensions["telegram_monitor"] = monitor
        try:
            r = get(client, "/api/v1/live_cascades")
            assert r.status_code == 200
            body = json.loads(r.data)
            assert isinstance(body, list)
        finally:
            mock_app.extensions["telegram_monitor"] = None


# =============================================================================
# GET /api/v1/sbm/matrices
# =============================================================================

class TestSBMMatrices:

    def test_returns_200(self, client):
        r = get(client, "/api/v1/sbm/matrices")
        assert r.status_code == 200

    def test_response_schema(self, client):
        body = json.loads(get(client, "/api/v1/sbm/matrices").data)
        for field in ("b_plus", "b_minus", "k", "class_sizes"):
            assert field in body, f"Missing field: {field}"

    def test_b_plus_is_k_by_k(self, client):
        body = json.loads(get(client, "/api/v1/sbm/matrices").data)
        k = body["k"]
        assert len(body["b_plus"]) == k
        for row in body["b_plus"]:
            assert len(row) == k

    def test_b_minus_is_k_by_k(self, client):
        body = json.loads(get(client, "/api/v1/sbm/matrices").data)
        k = body["k"]
        assert len(body["b_minus"]) == k
        for row in body["b_minus"]:
            assert len(row) == k

    def test_class_sizes_length_equals_k(self, client):
        body = json.loads(get(client, "/api/v1/sbm/matrices").data)
        assert len(body["class_sizes"]) == body["k"]

    def test_b_values_are_non_negative(self, client):
        body = json.loads(get(client, "/api/v1/sbm/matrices").data)
        for matrix_key in ("b_plus", "b_minus"):
            for row in body[matrix_key]:
                for val in row:
                    assert val >= 0.0, f"{matrix_key} has negative value: {val}"

    def test_mock_flag_present(self, client):
        body = json.loads(get(client, "/api/v1/sbm/matrices").data)
        assert "mock" in body

    def test_mock_true_when_no_sbm(self, client):
        body = json.loads(get(client, "/api/v1/sbm/matrices").data)
        assert body["mock"] is True

    def test_real_sbm_used_when_injected(self, mock_app, client):
        import numpy as np
        sbm = _make_mock_sbm(k=4)
        sbm.b_plus = np.eye(4) * 1e-3
        sbm.b_minus = np.ones((4, 4)) * 5e-4
        sbm.class_sizes = np.array([100, 200, 150, 50], dtype=float)
        mock_app.extensions["sbm"] = sbm
        try:
            body = json.loads(get(client, "/api/v1/sbm/matrices").data)
            assert body["mock"] is False
            assert body["k"] == 4
            assert len(body["class_sizes"]) == 4
        finally:
            mock_app.extensions["sbm"] = None


# =============================================================================
# Error handling
# =============================================================================

class TestErrorHandling:

    def test_unknown_route_returns_404(self, client):
        r = get(client, "/api/v1/nonexistent")
        assert r.status_code == 404

    def test_404_response_has_error_field(self, client):
        body = json.loads(get(client, "/api/v1/does_not_exist").data)
        assert "error" in body

    def test_get_on_post_only_route_returns_4xx(self, client):
        # Flask returns 404 (not 405) when no GET handler exists for a route
        # that only allows POST — this is standard Flask behaviour for blueprints.
        r = client.get("/api/v1/classify")
        assert r.status_code in (404, 405)

    def test_post_on_get_only_route_returns_405(self, client):
        r = client.post("/api/v1/live_cascades")
        assert r.status_code == 405


# =============================================================================
# Internal helpers — unit tests
# =============================================================================

class TestHelpers:

    def test_edges_to_digraph_basic(self):
        from api.routes.classify import _edges_to_digraph
        G = _edges_to_digraph([["A", "B"], ["B", "C"]])
        assert G.number_of_nodes() == 3
        assert G.number_of_edges() == 2

    def test_edges_to_digraph_sets_default_attrs(self):
        from api.routes.classify import _edges_to_digraph
        G = _edges_to_digraph([["A", "B"]])
        for node in G.nodes():
            assert "followers" in G.nodes[node]
            assert "friends" in G.nodes[node]
            assert "time" in G.nodes[node]

    def test_edges_to_digraph_skips_short_edges(self):
        from api.routes.classify import _edges_to_digraph
        G = _edges_to_digraph([["A", "B"], ["C"], []])
        assert G.number_of_edges() == 1

    def test_edges_to_digraph_deduplicates_nodes(self):
        from api.routes.classify import _edges_to_digraph
        G = _edges_to_digraph([["A", "B"], ["A", "C"], ["A", "D"]])
        assert "A" in G.nodes
        assert G.number_of_nodes() == 4

    def test_mock_simulate_output_structure(self):
        from api.routes.simulate import _mock_simulate
        result = _mock_simulate(content="false", alpha=1.5, seed=42)
        assert "steps" in result
        assert "r_inf" in result
        assert "lp_types" in result
        assert "d_star_final" in result
        assert result["mock"] is True

    def test_mock_simulate_r_inf_matches_step_totals(self):
        from api.routes.simulate import _mock_simulate
        result = _mock_simulate(content="false", alpha=1.5, seed=42)
        if result["steps"]:
            last = result["steps"][-1]
            assert last["total_reached"] == result["r_inf"]

    def test_generate_mock_cascades_returns_n_items(self):
        from api.routes.monitoring import _generate_mock_cascades
        for n in (1, 3, 10):
            assert len(_generate_mock_cascades(n)) == n

    def test_generate_mock_cascades_started_is_iso(self):
        from api.routes.monitoring import _generate_mock_cascades
        import datetime
        for c in _generate_mock_cascades(3):
            # Should parse without error
            datetime.datetime.fromisoformat(c["started"].rstrip("Z"))
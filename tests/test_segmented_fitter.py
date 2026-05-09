"""
tests/test_segmented_fitter.py
================================
Unit and integration tests for pipeline/segmented_sbm_fitter.py.

Test strategy
-------------
All tests use synthetic NetworkX DiGraphs so the full WICO dataset is never
required.  The synthetic graphs are constructed to have a known structure:

  "true" content graphs  : predominantly within-class edges (nodes 0-4 ↔ 0-4)
  "false" content graphs : predominantly cross-class edges  (nodes 0-4 ↔ 5-9)

This gives a predictable b_plus (diagonal-dominant) and b_minus (off-diagonal-
dominant) that we can assert against.

Running
-------
    pytest tests/test_segmented_fitter.py -v
    pytest tests/test_segmented_fitter.py -v -k "test_compare"
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Make the project root importable without an installed package
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "graph_engine"))
sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))

from network_model import SBM, SBMFitter, make_synthetic_sbm


# ---------------------------------------------------------------------------
# Helpers — build synthetic cascades
# ---------------------------------------------------------------------------

def _make_within_class_graph(
    class_a: list[str],
    class_b: list[str],
    seed: int = 0,
) -> nx.DiGraph:
    """
    Return a DiGraph where edges are *within* class_a and *within* class_b.
    Represents true-content behaviour (stays in community).
    """
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    for cls in (class_a, class_b):
        nodes = list(cls)
        for src in nodes:
            for dst in rng.choice(nodes, size=min(2, len(nodes)), replace=False):
                if src != dst:
                    G.add_edge(src, dst)
    return G


def _make_cross_class_graph(
    class_a: list[str],
    class_b: list[str],
    seed: int = 0,
) -> nx.DiGraph:
    """
    Return a DiGraph where edges cross from class_a to class_b.
    Represents false-content behaviour (jumps between communities).
    """
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    for src in class_a:
        for dst in rng.choice(class_b, size=min(2, len(class_b)), replace=False):
            G.add_edge(src, dst)
    return G


# Node pools for two synthetic polarisation classes
CLASS_A = [f"a{i}" for i in range(20)]
CLASS_B = [f"b{i}" for i in range(20)]
ALL_NODES = CLASS_A + CLASS_B


def _synthetic_true_graphs(n: int = 10) -> list[nx.DiGraph]:
    return [_make_within_class_graph(CLASS_A, CLASS_B, seed=i) for i in range(n)]


def _synthetic_false_graphs(n: int = 10) -> list[nx.DiGraph]:
    return [_make_cross_class_graph(CLASS_A, CLASS_B, seed=i) for i in range(n)]


def _fixed_partition() -> tuple[dict, int, np.ndarray]:
    """Return a hand-crafted 2-class partition matching CLASS_A / CLASS_B."""
    partition = {n: 0 for n in CLASS_A}
    partition.update({n: 1 for n in CLASS_B})
    k = 2
    class_sizes = np.array([len(CLASS_A), len(CLASS_B)], dtype=np.int64)
    return partition, k, class_sizes


@pytest.fixture
def tmp_dir() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


# ---------------------------------------------------------------------------
# Fixtures — fake WICO-like cascade data
# ---------------------------------------------------------------------------

def _build_fake_wico_cascades() -> list[tuple[nx.DiGraph, str, str, str]]:
    """
    Produce a list of (G, cascade_id, binary_label, class_dir) tuples that
    mimic the output of load_wico_all_cascades().

    5G_Conspiracy_Graphs  → false (aggressive cross-class)
    Other_Graphs          → false (mild cross-class)
    Non_Conspiracy_Graphs → true  (within-class)
    """
    cascades: list[tuple[nx.DiGraph, str, str, str]] = []

    # 5G: very aggressive cross-class (alternate nodes, maximise cross edges)
    for i in range(15):
        G = nx.DiGraph()
        for a in CLASS_A[:8]:
            for b in CLASS_B[:8]:
                G.add_edge(a, b)  # every node in A → every node in B
        cascades.append((G, f"5g_{i}", "false", "5G_Conspiracy_Graphs"))

    # Other: milder cross-class (only a few edges cross)
    for i in range(15):
        G = _make_cross_class_graph(CLASS_A[:4], CLASS_B[:4], seed=i + 100)
        cascades.append((G, f"other_{i}", "false", "Other_Graphs"))

    # Non-conspiracy: pure within-class
    for i in range(15):
        G = _make_within_class_graph(CLASS_A, CLASS_B, seed=i + 200)
        cascades.append((G, f"nc_{i}", "true", "Non_Conspiracy_Graphs"))

    return cascades


# ---------------------------------------------------------------------------
# Unit tests — _BEstimator
# ---------------------------------------------------------------------------

class TestBEstimator:
    """Tests for the thin _BEstimator wrapper."""

    def setup_method(self):
        from segmented_sbm_fitter import _BEstimator  # noqa: PLC0415
        self._BEstimator = _BEstimator

    def test_estimate_produces_valid_matrix(self):
        partition, k, class_sizes = _fixed_partition()
        estimator = self._BEstimator(
            k=k,
            partition=partition,
            class_sizes=class_sizes,
            fitter_kwargs={},
        )
        graphs = _synthetic_within_class_graphs = _synthetic_true_graphs(5)
        b = estimator.estimate(graphs, label="true")
        assert b.shape == (k, k)
        assert np.all(b >= 0)
        assert np.all(b <= 1)

    def test_true_b_diagonal_dominant(self):
        """Within-class graphs → b_plus diagonal should dominate."""
        partition, k, class_sizes = _fixed_partition()
        from segmented_sbm_fitter import _BEstimator
        estimator = _BEstimator(k=k, partition=partition,
                                class_sizes=class_sizes, fitter_kwargs={})
        b_plus = estimator.estimate(_synthetic_true_graphs(20), label="true")
        diag_mean    = np.diag(b_plus).mean()
        offdiag_mean = b_plus[~np.eye(k, dtype=bool)].mean()
        assert diag_mean > offdiag_mean, (
            f"Expected diagonal > off-diagonal for within-class graphs. "
            f"Got diag={diag_mean:.4e}, offdiag={offdiag_mean:.4e}"
        )

    def test_false_b_offdiag_dominant(self):
        """Cross-class graphs → b_minus off-diagonal should dominate."""
        partition, k, class_sizes = _fixed_partition()
        from segmented_sbm_fitter import _BEstimator
        estimator = _BEstimator(k=k, partition=partition,
                                class_sizes=class_sizes, fitter_kwargs={})
        b_minus = estimator.estimate(_synthetic_false_graphs(20), label="false")
        diag_mean    = np.diag(b_minus).mean()
        offdiag_mean = b_minus[~np.eye(k, dtype=bool)].mean()
        assert offdiag_mean > diag_mean, (
            f"Expected off-diagonal > diagonal for cross-class graphs. "
            f"Got diag={diag_mean:.4e}, offdiag={offdiag_mean:.4e}"
        )


# ---------------------------------------------------------------------------
# Unit tests — SegmentedSBMFitter (mocked WICO loader)
# ---------------------------------------------------------------------------

class TestSegmentedSBMFitter:
    """Tests for SegmentedSBMFitter using mocked cascade data."""

    @pytest.fixture(autouse=True)
    def _patch_wico_loader(self, monkeypatch):
        """Replace load_wico_all_cascades with a fake that returns synthetic data."""
        fake_cascades = _build_fake_wico_cascades()
        monkeypatch.setattr(
            "segmented_sbm_fitter.load_wico_all_cascades",
            lambda *args, **kwargs: fake_cascades,
        )
        self._fake_cascades = fake_cascades

    @pytest.fixture(autouse=True)
    def _patch_global_sbm(self, monkeypatch, tmp_dir):
        """
        Prevent the fitter from loading a real global SBM.
        Instead, inject our fixed synthetic partition.
        """
        # No pre-existing partition — force Louvain path
        # but patch cfg.paths.sbm_matrices to a non-existent dir
        monkeypatch.setattr(
            "segmented_sbm_fitter.DEFAULT_SEGMENTS_DIR",
            tmp_dir / "sbm_segments",
        )
        self._seg_dir = tmp_dir / "sbm_segments"

    def _make_fitter(self, tmp_dir: Path):
        from segmented_sbm_fitter import SegmentedSBMFitter
        fitter = SegmentedSBMFitter(
            wico_graph_dir = tmp_dir,  # dummy — loader is mocked
            output_dir     = self._seg_dir,
        )
        # Inject the fixed partition directly (skip Louvain)
        partition, k, class_sizes = _fixed_partition()
        fitter._cascades     = self._fake_cascades
        fitter._partition    = partition
        fitter._k            = k
        fitter._class_sizes  = class_sizes
        return fitter

    # ── fit_segment ──────────────────────────────────────────────────────────

    def test_fit_segment_returns_sbm(self, tmp_dir):
        fitter = self._make_fitter(tmp_dir)
        sbm = fitter.fit_segment("conspiracy_5g", force_refit=True)
        assert isinstance(sbm, SBM)
        assert sbm.k == 2

    def test_fit_segment_saves_files(self, tmp_dir):
        fitter = self._make_fitter(tmp_dir)
        fitter.fit_segment("conspiracy_5g", force_refit=True)
        seg_dir = self._seg_dir / "conspiracy_5g"
        for fname in ("b_plus.npy", "b_minus.npy", "k.npy",
                      "class_sizes.npy", "metadata.json"):
            assert (seg_dir / fname).exists(), f"Missing {fname}"

    def test_fit_segment_metadata_correct(self, tmp_dir):
        fitter = self._make_fitter(tmp_dir)
        fitter.fit_segment("other_conspiracy", force_refit=True)
        meta_path = self._seg_dir / "other_conspiracy" / "metadata.json"
        with open(meta_path) as fh:
            meta = json.load(fh)
        assert meta["segment"] == "other_conspiracy"
        assert "Other_Graphs" in meta["false_dirs"]
        assert "Non_Conspiracy_Graphs" in meta["true_dirs"]
        assert meta["k"] == 2
        assert meta["n_false_cascades"] == 15
        assert meta["n_true_cascades"]  == 15

    def test_fit_segment_reuse_cache(self, tmp_dir):
        """Second call to fit_segment should not re-fit (returns cached SBM)."""
        fitter = self._make_fitter(tmp_dir)
        sbm1 = fitter.fit_segment("conspiracy_5g", force_refit=True)
        sbm2 = fitter.fit_segment("conspiracy_5g", force_refit=False)
        # Both should have identical b_plus arrays
        np.testing.assert_array_equal(sbm1.b_plus, sbm2.b_plus)

    def test_fit_segment_force_refit(self, tmp_dir):
        """force_refit=True should overwrite the cached SBM."""
        fitter = self._make_fitter(tmp_dir)
        sbm1 = fitter.fit_segment("conspiracy_5g", force_refit=True)
        sbm2 = fitter.fit_segment("conspiracy_5g", force_refit=True)
        assert isinstance(sbm2, SBM)

    def test_unknown_segment_raises(self, tmp_dir):
        fitter = self._make_fitter(tmp_dir)
        with pytest.raises(ValueError, match="Unknown segment"):
            fitter.fit_segment("nonexistent_segment")

    # ── fit_all_segments ─────────────────────────────────────────────────────

    def test_fit_all_segments_returns_all_keys(self, tmp_dir):
        from segmented_sbm_fitter import SEGMENTS
        fitter = self._make_fitter(tmp_dir)
        results = fitter.fit_all_segments(force_refit=True)
        assert set(results.keys()) == set(SEGMENTS.keys())

    def test_fit_all_segments_shared_partition(self, tmp_dir):
        """All segments must use the same partition (cross-comparability)."""
        fitter = self._make_fitter(tmp_dir)
        results = fitter.fit_all_segments(force_refit=True)
        partitions = [sbm.partition for sbm in results.values()]
        # Compare against the first
        ref = partitions[0]
        for p in partitions[1:]:
            assert set(p.keys()) == set(ref.keys()), (
                "Segment partitions have different node sets — "
                "cross-comparability violated."
            )
            assert p == ref, "Partition assignments differ between segments."

    # ── b-matrix properties ──────────────────────────────────────────────────

    def test_5g_cross_class_higher_than_other(self, tmp_dir):
        """
        Key hypothesis: 5G conspiracy b_minus off-diagonal > other_conspiracy
        b_minus off-diagonal (because 5G graphs have ALL class_A → class_B edges).
        """
        fitter = self._make_fitter(tmp_dir)
        results = fitter.fit_all_segments(force_refit=True)

        def _offdiag_mean(sbm: SBM) -> float:
            k = sbm.k
            if k == 1:
                return 0.0
            mask = ~np.eye(k, dtype=bool)
            return float(sbm.b_minus[mask].mean())

        offdiag_5g    = _offdiag_mean(results["conspiracy_5g"])
        offdiag_other = _offdiag_mean(results["other_conspiracy"])

        assert offdiag_5g > offdiag_other, (
            f"Expected 5G b_minus off-diagonal ({offdiag_5g:.4e}) > "
            f"other_conspiracy ({offdiag_other:.4e}). "
            "5G cascades were constructed with ALL cross-class edges."
        )

    def test_sbm_matrices_in_valid_range(self, tmp_dir):
        fitter = self._make_fitter(tmp_dir)
        results = fitter.fit_all_segments(force_refit=True)
        for name, sbm in results.items():
            assert np.all(sbm.b_plus  >= 0), f"[{name}] b_plus has negative values"
            assert np.all(sbm.b_plus  <= 1), f"[{name}] b_plus > 1"
            assert np.all(sbm.b_minus >= 0), f"[{name}] b_minus has negative values"
            assert np.all(sbm.b_minus <= 1), f"[{name}] b_minus > 1"


# ---------------------------------------------------------------------------
# Unit tests — load_segment
# ---------------------------------------------------------------------------

class TestLoadSegment:
    def test_load_segment_roundtrip(self, tmp_dir):
        from segmented_sbm_fitter import load_segment
        # Save a synthetic SBM into the expected location
        sbm = make_synthetic_sbm(k=2, n_users=40, seed=7)
        seg_dir = tmp_dir / "test_seg"
        sbm.save(seg_dir)

        loaded = load_segment("test_seg", segments_dir=tmp_dir)
        np.testing.assert_array_almost_equal(loaded.b_plus,  sbm.b_plus)
        np.testing.assert_array_almost_equal(loaded.b_minus, sbm.b_minus)
        assert loaded.k == sbm.k

    def test_load_segment_missing_raises(self, tmp_dir):
        from segmented_sbm_fitter import load_segment
        with pytest.raises(FileNotFoundError, match="not found"):
            load_segment("does_not_exist", segments_dir=tmp_dir)


# ---------------------------------------------------------------------------
# Unit tests — compare_segments
# ---------------------------------------------------------------------------

class TestCompareSegments:
    def _save_synthetic_segments(self, base_dir: Path) -> None:
        """Save three synthetic SBMs with known asymmetry ordering."""
        # high asymmetry: more off-diagonal in b_minus
        sbm_high = make_synthetic_sbm(k=2, x=0.001, y=0.004, n_users=40)
        sbm_high.save(base_dir / "conspiracy_5g")

        # lower asymmetry
        sbm_low = make_synthetic_sbm(k=2, x=0.003, y=0.001, n_users=40)
        sbm_low.save(base_dir / "other_conspiracy")

        # middle
        sbm_mid = make_synthetic_sbm(k=2, x=0.002, y=0.002, n_users=40)
        sbm_mid.save(base_dir / "all_conspiracy")

    def test_compare_returns_dataframe(self, tmp_dir):
        from segmented_sbm_fitter import compare_segments, SEGMENTS
        self._save_synthetic_segments(tmp_dir)
        df = compare_segments(segments_dir=tmp_dir)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_compare_columns_present(self, tmp_dir):
        from segmented_sbm_fitter import compare_segments
        self._save_synthetic_segments(tmp_dir)
        df = compare_segments(segments_dir=tmp_dir)
        required = {
            "segment", "k", "b_plus_diag_mean", "b_plus_offdiag_mean",
            "b_minus_diag_mean", "b_minus_offdiag_mean",
            "cross_class_asymmetry",
        }
        assert required.issubset(df.columns), (
            f"Missing columns: {required - set(df.columns)}"
        )

    def test_compare_sorted_by_asymmetry(self, tmp_dir):
        from segmented_sbm_fitter import compare_segments
        self._save_synthetic_segments(tmp_dir)
        df = compare_segments(segments_dir=tmp_dir)
        asymmetries = df["cross_class_asymmetry"].dropna().tolist()
        assert asymmetries == sorted(asymmetries, reverse=True), (
            "compare_segments should return rows sorted by cross_class_asymmetry descending."
        )

    def test_compare_skips_missing_segments(self, tmp_dir):
        from segmented_sbm_fitter import compare_segments
        # Only save one segment
        sbm = make_synthetic_sbm(k=2, n_users=40)
        sbm.save(tmp_dir / "conspiracy_5g")

        df = compare_segments(
            segment_names=["conspiracy_5g", "other_conspiracy"],
            segments_dir=tmp_dir,
        )
        assert len(df) == 1
        assert df.iloc[0]["segment"] == "conspiracy_5g"

    def test_compare_asymmetry_values_positive(self, tmp_dir):
        from segmented_sbm_fitter import compare_segments
        self._save_synthetic_segments(tmp_dir)
        df = compare_segments(segments_dir=tmp_dir)
        assert (df["cross_class_asymmetry"].dropna() > 0).all()


# ---------------------------------------------------------------------------
# Integration test — end-to-end with SegmentedSBMFitter (no mocks)
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """
    Builds real SBMFitter objects using synthetic graphs.
    Exercises the full SegmentedSBMFitter pipeline without mock.
    """

    def test_full_pipeline_no_wico(self, tmp_dir):
        """
        Simulate the full pipeline using synthetic cascades injected directly
        into the fitter (bypasses load_wico_all_cascades and Louvain).
        """
        from segmented_sbm_fitter import SegmentedSBMFitter, SEGMENTS

        fake_cascades = _build_fake_wico_cascades()
        partition, k, class_sizes = _fixed_partition()

        fitter = SegmentedSBMFitter(
            wico_graph_dir = tmp_dir,
            output_dir     = tmp_dir / "segs",
        )
        fitter._cascades    = fake_cascades
        fitter._partition   = partition
        fitter._k           = k
        fitter._class_sizes = class_sizes

        results = fitter.fit_all_segments(force_refit=True)

        # All three segments should be fitted
        assert set(results.keys()) == {"conspiracy_5g", "other_conspiracy", "all_conspiracy"}

        # Matrices should be valid
        for name, sbm in results.items():
            assert sbm.k == 2, f"[{name}] expected k=2, got {sbm.k}"
            assert sbm.b_plus.shape  == (2, 2)
            assert sbm.b_minus.shape == (2, 2)

        # 5G should have higher off-diagonal b_minus than "other"
        mask = ~np.eye(2, dtype=bool)
        offdiag_5g    = results["conspiracy_5g"].b_minus[mask].mean()
        offdiag_other = results["other_conspiracy"].b_minus[mask].mean()
        assert offdiag_5g > offdiag_other, (
            "End-to-end: 5G b_minus off-diag should exceed other_conspiracy"
        )

    def test_compare_segments_after_full_pipeline(self, tmp_dir):
        from segmented_sbm_fitter import SegmentedSBMFitter, compare_segments

        fake_cascades = _build_fake_wico_cascades()
        partition, k, class_sizes = _fixed_partition()

        fitter = SegmentedSBMFitter(
            wico_graph_dir = tmp_dir,
            output_dir     = tmp_dir / "segs",
        )
        fitter._cascades    = fake_cascades
        fitter._partition   = partition
        fitter._k           = k
        fitter._class_sizes = class_sizes
        fitter.fit_all_segments(force_refit=True)

        df = compare_segments(segments_dir=tmp_dir / "segs")
        assert not df.empty
        assert "cross_class_asymmetry" in df.columns
        # conspiracy_5g should rank first
        assert df.iloc[0]["segment"] == "conspiracy_5g", (
            f"Expected conspiracy_5g to rank first by asymmetry. "
            f"Got: {df['segment'].tolist()}"
        )


# ---------------------------------------------------------------------------
# run_pipeline integration: segment parameter (smoke test)
# ---------------------------------------------------------------------------

class TestRunPipelineSegmentParam:
    """
    Smoke-tests that run_pipeline.py accepts --segment and loads the
    appropriate SBM.  Uses monkeypatching to avoid real data.
    """

    def test_load_segment_sbm_in_pipeline(self, tmp_dir, monkeypatch):
        """
        Verify that simulate_cascade_following can run with an SBM loaded
        from a segment directory.
        """
        # Save a minimal SBM for "conspiracy_5g"
        partition, k, class_sizes = _fixed_partition()
        sbm = make_synthetic_sbm(k=2, n_users=40, seed=99)
        # Override partition to match CLASS_A / CLASS_B
        sbm = SBM(
            b_plus      = sbm.b_plus,
            b_minus     = sbm.b_minus,
            k           = k,
            partition   = partition,
            class_sizes = class_sizes,
        )

        # Build a minimal cascade graph that uses nodes from partition
        G = nx.DiGraph()
        for a in CLASS_A[:5]:
            for b in CLASS_B[:5]:
                G.add_edge(a, b)
        root = CLASS_A[0]

        # Import simulate_cascade_following
        sys.path.insert(0, str(PROJECT_ROOT / "pipeline"))
        try:
            from run_pipeline import simulate_cascade_following
        except ImportError:
            pytest.skip("run_pipeline not importable in this environment.")

        size, lp_types = simulate_cascade_following(
            G                  = G,
            partition          = partition,
            root               = root,
            sbm                = sbm,
            alpha              = 1.5,
            lam                = 1.0,
            global_class_sizes = class_sizes.astype(float),
            seed               = 0,
        )
        assert isinstance(size, int)
        assert size >= 1
        assert isinstance(lp_types, list)
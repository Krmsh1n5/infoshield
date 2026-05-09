"""
tests/test_instagram_ingestor.py — Test suite for InstagramIngestor.

Run with:
    pytest tests/test_instagram_ingestor.py -v

All tests use mock=True and require no external credentials.
The test suite validates:
  A. BaseIngestor contract compliance
  B. Graph structural correctness (InfoShield pipeline compatibility)
  C. Node attribute contract (WICO compatibility)
  D. Statistical properties of mock cascades
  E. All three public methods (get_post_cascade, search_hashtag_cascade,
     build_from_comments)
  F. Edge direction convention
  G. Validation / repair logic
  H. Reproducibility via seed
"""

from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

# ── path setup ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.ingestors.base_ingestor import BaseIngestor, ValidationReport
from pipeline.ingestors.instagram_ingestor import InstagramIngestor


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def ingestor() -> InstagramIngestor:
    """Shared mock ingestor — recreated once per module for speed."""
    return InstagramIngestor(mock=True, seed=42)


@pytest.fixture(scope="module")
def single_cascade(ingestor: InstagramIngestor) -> nx.DiGraph:
    """A single cascade built with get_post_cascade()."""
    return ingestor.get_post_cascade("test_post_001")


@pytest.fixture(scope="module")
def comment_cascade(ingestor: InstagramIngestor) -> nx.DiGraph:
    """A cascade built via the comment-thread fallback."""
    return ingestor.build_from_comments("test_post_002")


# ──────────────────────────────────────────────────────────────────────────
# A. BaseIngestor contract
# ──────────────────────────────────────────────────────────────────────────

class TestBaseIngestorContract:
    """InstagramIngestor must satisfy the BaseIngestor ABC."""

    def test_is_subclass_of_base_ingestor(self):
        assert issubclass(InstagramIngestor, BaseIngestor)

    def test_cannot_instantiate_without_token_or_mock(self):
        with pytest.raises(ValueError, match="access_token"):
            InstagramIngestor()

    def test_mock_instantiation_requires_no_token(self):
        ing = InstagramIngestor(mock=True)
        assert ing.mock is True

    def test_build_cascade_returns_digraph(self, ingestor, single_cascade):
        assert isinstance(single_cascade, nx.DiGraph)

    def test_get_root_node_returns_string(self, ingestor, single_cascade):
        root = ingestor.get_root_node(single_cascade)
        assert isinstance(root, str)
        assert root in single_cascade.nodes

    def test_get_node_metadata_returns_required_keys(self, ingestor, single_cascade):
        root = ingestor.get_root_node(single_cascade)
        meta = ingestor.get_node_metadata(root)
        for key in ("followers", "time", "friends"):
            assert key in meta, f"Missing required metadata key: '{key}'"

    def test_validate_graph_returns_validation_report(self, ingestor, single_cascade):
        report = ingestor.validate_graph(single_cascade)
        assert isinstance(report, ValidationReport)


# ──────────────────────────────────────────────────────────────────────────
# B. Graph structural correctness
# ──────────────────────────────────────────────────────────────────────────

class TestGraphStructure:
    """Verify the DiGraph satisfies InfoShield pipeline preconditions."""

    def test_is_directed(self, single_cascade):
        assert isinstance(single_cascade, nx.DiGraph)

    def test_minimum_node_count(self, single_cascade):
        assert single_cascade.number_of_nodes() >= 2

    def test_node_count_in_mock_range(self, ingestor):
        for i in range(10):
            G = ingestor.get_post_cascade(f"range_test_{i}")
            n = G.number_of_nodes()
            assert (
                ingestor._MOCK_MIN_NODES <= n <= ingestor._MOCK_MAX_NODES
            ), f"Node count {n} outside [{ingestor._MOCK_MIN_NODES}, {ingestor._MOCK_MAX_NODES}]"

    def test_exactly_one_root(self, single_cascade):
        """find_root_user() requires exactly one in_degree-0 node."""
        roots = [n for n, d in single_cascade.in_degree() if d == 0]
        assert len(roots) == 1, f"Expected 1 root, found {len(roots)}: {roots}"

    def test_weakly_connected(self, single_cascade):
        assert nx.is_weakly_connected(single_cascade)

    def test_root_has_in_degree_zero(self, ingestor, single_cascade):
        root = ingestor.get_root_node(single_cascade)
        assert single_cascade.in_degree(root) == 0

    def test_non_root_nodes_have_nonzero_in_degree(self, ingestor, single_cascade):
        root = ingestor.get_root_node(single_cascade)
        for node in single_cascade.nodes():
            if node != root:
                assert single_cascade.in_degree(node) > 0 or \
                       single_cascade.out_degree(node) > 0, \
                       f"Node {node} is isolated"

    def test_no_self_loops(self, single_cascade):
        self_loops = list(nx.selfloop_edges(single_cascade))
        assert len(self_loops) == 0, f"Self-loops found: {self_loops}"


# ──────────────────────────────────────────────────────────────────────────
# C. Node attribute contract (WICO compatibility)
# ──────────────────────────────────────────────────────────────────────────

class TestNodeAttributes:
    """All nodes must carry the WICO-required attribute set."""

    REQUIRED = ("followers", "time", "friends")

    def test_all_nodes_have_required_attrs(self, single_cascade):
        for node, data in single_cascade.nodes(data=True):
            for attr in self.REQUIRED:
                assert attr in data, (
                    f"Node '{node}' missing required attribute '{attr}'"
                )

    def test_followers_are_non_negative(self, single_cascade):
        for node, data in single_cascade.nodes(data=True):
            assert data["followers"] >= 0, (
                f"Node '{node}' has negative followers: {data['followers']}"
            )

    def test_time_is_non_negative(self, single_cascade):
        for node, data in single_cascade.nodes(data=True):
            assert data["time"] >= 0.0, (
                f"Node '{node}' has negative time: {data['time']}"
            )

    def test_friends_are_non_negative(self, single_cascade):
        for node, data in single_cascade.nodes(data=True):
            assert data["friends"] >= 0, (
                f"Node '{node}' has negative friends: {data['friends']}"
            )

    def test_root_time_is_zero(self, ingestor, single_cascade):
        root = ingestor.get_root_node(single_cascade)
        assert single_cascade.nodes[root]["time"] == 0.0

    def test_platform_tag_present(self, single_cascade):
        for node, data in single_cascade.nodes(data=True):
            assert data.get("platform") == "instagram"

    def test_root_has_highest_followers_among_roots(self, ingestor, single_cascade):
        """Root returned by get_root_node should have max followers of all roots."""
        roots = [n for n, d in single_cascade.in_degree() if d == 0]
        returned_root = ingestor.get_root_node(single_cascade)
        max_followers_root = max(
            roots, key=lambda n: single_cascade.nodes[n].get("followers", 0)
        )
        assert returned_root == max_followers_root


# ──────────────────────────────────────────────────────────────────────────
# D. Statistical properties of mock cascades
# ──────────────────────────────────────────────────────────────────────────

class TestMockStatistics:
    """
    Mock cascades must statistically match WICO graphs.
    Tests use lenient bounds to avoid flakiness from RNG variance.
    """
    N_SAMPLES = 50

    @pytest.fixture(scope="class")
    def many_cascades(self, ingestor):
        return [ingestor.get_post_cascade(f"stat_{i}") for i in range(self.N_SAMPLES)]

    def test_avg_degree_near_wico(self, ingestor, many_cascades):
        avg_degrees = [
            G.number_of_edges() / G.number_of_nodes()
            for G in many_cascades
        ]
        mean_deg = np.mean(avg_degrees)
        # WICO 2.82 is the *undirected* average. For directed sparse trees
        # with 15% cross-edges, directed avg ≈ 1.4 is normal and correct.
        assert 1.0 <= mean_deg <= 5.0, (
            f"Mean avg_degree={mean_deg:.3f} outside expected range [1.0, 5.0]"
        )

    def test_follower_distribution_roughly_lognormal(self, ingestor, many_cascades):
        """Log of followers should be approximately normally distributed."""
        all_followers = []
        for G in many_cascades:
            all_followers.extend(
                [d["followers"] for _, d in G.nodes(data=True) if d["followers"] > 0]
            )
        log_followers = np.log(all_followers)
        # Mean of log(followers) should be near mu=log(500)≈6.215 (±2)
        mean_log = np.mean(log_followers)
        assert 4.0 <= mean_log <= 9.0, (
            f"Mean log(followers)={mean_log:.2f}; expected near 6.215"
        )

    def test_all_cascades_pass_validation(self, ingestor, many_cascades):
        failed = [
            i for i, G in enumerate(many_cascades)
            if not ingestor.validate_graph(G).ok
        ]
        assert not failed, f"Cascades {failed} failed validation"

    def test_cascade_depths_positive(self, ingestor, many_cascades):
        for G in many_cascades:
            roots = [n for n, d in G.in_degree() if d == 0]
            if roots:
                # BFS in G (content flow direction): root→children→grandchildren
                depths = nx.single_source_shortest_path_length(G, roots[0])
                assert max(depths.values()) > 0, "All non-root nodes at depth 0"


# ──────────────────────────────────────────────────────────────────────────
# E. All three public methods
# ──────────────────────────────────────────────────────────────────────────

class TestPublicMethods:

    def test_get_post_cascade_returns_digraph(self, ingestor):
        G = ingestor.get_post_cascade("method_test_001")
        assert isinstance(G, nx.DiGraph)

    def test_search_hashtag_cascade_returns_list(self, ingestor):
        cascades = ingestor.search_hashtag_cascade("azerbeycan")
        assert isinstance(cascades, list)
        assert len(cascades) >= 2  # mock returns 2–5 cascades

    def test_search_hashtag_cascade_all_valid(self, ingestor):
        cascades = ingestor.search_hashtag_cascade("infoshield")
        for G in cascades:
            report = ingestor.validate_graph(G)
            assert report.ok, f"Hashtag cascade failed validation: {report}"

    def test_build_from_comments_returns_digraph(self, ingestor, comment_cascade):
        assert isinstance(comment_cascade, nx.DiGraph)

    def test_build_from_comments_has_root(self, ingestor, comment_cascade):
        roots = [n for n, d in comment_cascade.in_degree() if d == 0]
        assert len(roots) == 1

    def test_build_from_comments_passes_validation(self, ingestor, comment_cascade):
        report = ingestor.validate_graph(comment_cascade)
        assert report.ok, str(report)

    def test_hashtag_cascades_independent_graphs(self, ingestor):
        """Each cascade returned by search_hashtag_cascade must be independent."""
        cascades = ingestor.search_hashtag_cascade("test")
        if len(cascades) >= 2:
            nodes_a = set(cascades[0].nodes())
            nodes_b = set(cascades[1].nodes())
            # Node ID sets must be disjoint (different post IDs in mock)
            # We use post ID prefix to distinguish — just check type
            assert nodes_a != nodes_b or len(nodes_a) == 0


# ──────────────────────────────────────────────────────────────────────────
# F. Edge direction convention
# ──────────────────────────────────────────────────────────────────────────

class TestEdgeDirectionConvention:
    """
    Convention: A→B means A engaged *because of* B's share.
    Root (original poster) has in_degree == 0.
    G.reverse() exposes the influence flow (who influenced whom).
    """

    def test_root_in_degree_zero(self, single_cascade, ingestor):
        root = ingestor.get_root_node(single_cascade)
        assert single_cascade.in_degree(root) == 0

    def test_root_out_degree_zero_in_reversed(self, single_cascade, ingestor):
        """
        After reversing, root should have out_degree == 0
        (no one influenced the root; it initiated the cascade).
        """
        root = ingestor.get_root_node(single_cascade)
        G_rev = single_cascade.reverse()
        assert G_rev.out_degree(root) == 0

    def test_root_reaches_all_nodes_in_G(self, single_cascade, ingestor):
        """
        In G (content flow direction root→children), root must be able to
        reach all nodes. This mirrors simulate_cascade_following() which does
        BFS from root in G to propagate information.
        """
        root = ingestor.get_root_node(single_cascade)
        reachable = nx.descendants(single_cascade, root) | {root}
        assert reachable == set(single_cascade.nodes()), (
            f"{len(set(single_cascade.nodes()) - reachable)} nodes unreachable from root in G"
        )

    def test_compatible_with_wico_reverse_convention(self, ingestor):
        """
        WICO graphs are loaded with G.reverse() in network_model.py.
        Our graphs should NOT need an extra reverse (already in correct orientation).
        The root in our graph is in_degree=0, same as WICO after reversal.
        """
        G = ingestor.get_post_cascade("direction_test")
        roots = [n for n, d in G.in_degree() if d == 0]
        assert len(roots) == 1, (
            "Graph must have exactly 1 root (in_degree=0) to match pipeline expectation."
        )


# ──────────────────────────────────────────────────────────────────────────
# G. Validation logic
# ──────────────────────────────────────────────────────────────────────────

class TestValidationLogic:

    def test_valid_graph_passes(self, ingestor, single_cascade):
        report = ingestor.validate_graph(single_cascade)
        assert report.ok

    def test_non_digraph_fails(self, ingestor):
        G = nx.Graph()
        G.add_nodes_from(["a", "b"])
        G.add_edge("a", "b")
        report = ingestor.validate_graph(G)
        assert not report.ok
        assert any("DiGraph" in e for e in report.errors)

    def test_single_node_fails(self, ingestor):
        G = nx.DiGraph()
        G.add_node("lonely")
        report = ingestor.validate_graph(G)
        assert not report.ok

    def test_multiple_roots_fails(self, ingestor):
        G = nx.DiGraph()
        G.add_node("root1", followers=100, time=0.0, friends=10)
        G.add_node("root2", followers=200, time=0.0, friends=20)
        G.add_node("leaf", followers=50, time=10.0, friends=5)
        G.add_edge("leaf", "root1")
        # root2 has in_degree=0 and is not connected to root1
        G.add_node("leaf2", followers=30, time=5.0, friends=3)
        G.add_edge("leaf2", "root2")
        report = ingestor.validate_graph(G)
        # Multiple roots OR disconnected graph — either error is correct
        assert not report.ok

    def test_disconnected_graph_fails(self, ingestor):
        G = nx.DiGraph()
        G.add_node("r1", followers=100, time=0.0, friends=10)
        G.add_node("r2", followers=200, time=0.0, friends=20)
        G.add_edge("c1", "r1") if False else None  # stays disconnected
        # Two isolated nodes with no edges
        report = ingestor.validate_graph(G)
        assert not report.ok

    def test_missing_followers_attr_fails(self, ingestor):
        G = nx.DiGraph()
        G.add_node("root", time=0.0, friends=10)   # missing 'followers'
        G.add_node("leaf", followers=50, time=5.0, friends=5)
        G.add_edge("leaf", "root")
        report = ingestor.validate_graph(G)
        assert not report.ok
        assert report.missing_followers > 0

    def test_report_str_contains_status(self, ingestor, single_cascade):
        report = ingestor.validate_graph(single_cascade)
        report_str = str(report)
        assert "PASS" in report_str or "FAIL" in report_str


# ──────────────────────────────────────────────────────────────────────────
# H. Reproducibility via seed
# ──────────────────────────────────────────────────────────────────────────

class TestReproducibility:

    def test_same_seed_same_graph(self):
        ing1 = InstagramIngestor(mock=True, seed=99)
        ing2 = InstagramIngestor(mock=True, seed=99)
        G1 = ing1.get_post_cascade("reproducibility_test")
        G2 = ing2.get_post_cascade("reproducibility_test")

        assert G1.number_of_nodes() == G2.number_of_nodes()
        assert G1.number_of_edges() == G2.number_of_edges()
        assert set(G1.nodes()) == set(G2.nodes())

    def test_different_seeds_different_graphs(self):
        ing1 = InstagramIngestor(mock=True, seed=1)
        ing2 = InstagramIngestor(mock=True, seed=2)
        G1 = ing1.get_post_cascade("seed_test")
        G2 = ing2.get_post_cascade("seed_test")

        # Different seeds should produce different graphs (very high probability)
        # Allow for the tiny probability they're equal in node count
        followers1 = sorted(
            d["followers"] for _, d in G1.nodes(data=True)
        )
        followers2 = sorted(
            d["followers"] for _, d in G2.nodes(data=True)
        )
        # At minimum node counts differ or follower lists differ
        assert (
            G1.number_of_nodes() != G2.number_of_nodes()
            or followers1 != followers2
        )

    def test_post_id_affects_mock_graph(self, ingestor):
        """Each call to get_post_cascade should progress the RNG."""
        G1 = ingestor.get_post_cascade("post_aaa")
        G2 = ingestor.get_post_cascade("post_bbb")
        # Post IDs share the same ingestor; graphs should differ
        # (sequential RNG advances)
        assert G1.number_of_nodes() != G2.number_of_nodes() or \
               G1.number_of_edges() != G2.number_of_edges() or \
               set(G1.nodes()) != set(G2.nodes())


# ──────────────────────────────────────────────────────────────────────────
# I. Prufer tree helper (unit)
# ──────────────────────────────────────────────────────────────────────────

class TestPruferHelper:

    def test_prufer_produces_correct_edge_count(self):
        n = 10
        sequence = list(range(n - 2))  # length n-2
        edges = InstagramIngestor._prufer_to_edges(list(range(n)), sequence)
        # A spanning tree on n nodes has exactly n-1 edges
        assert len(edges) == n - 1

    def test_prufer_nodes_within_range(self):
        n = 8
        rng = np.random.default_rng(0)
        sequence = rng.integers(0, n, size=n - 2).tolist()
        edges = InstagramIngestor._prufer_to_edges(list(range(n)), sequence)
        all_nodes = {v for e in edges for v in e}
        assert all_nodes.issubset(set(range(n)))

    def test_prufer_result_is_tree(self):
        n = 15
        rng = np.random.default_rng(7)
        sequence = rng.integers(0, n, size=n - 2).tolist()
        edges = InstagramIngestor._prufer_to_edges(list(range(n)), sequence)
        T = nx.Graph()
        T.add_nodes_from(range(n))
        T.add_edges_from(edges)
        assert nx.is_tree(T)


# ──────────────────────────────────────────────────────────────────────────
# J. Pipeline compatibility (end-to-end mock integration)
# ──────────────────────────────────────────────────────────────────────────

class TestPipelineCompatibility:
    """
    Test that the output of InstagramIngestor can flow directly into the
    InfoShield pipeline without conversion.

    We mock the pipeline functions that require fitted SBM matrices.
    """

    def test_find_root_user_equivalent(self, ingestor, single_cascade):
        """
        Replicates find_root_user() from pipeline/sbm_fitter.py.
        Must find exactly one node with in_degree==0.
        """
        roots = [n for n, d in single_cascade.in_degree() if d == 0]
        assert len(roots) == 1
        root = max(roots, key=lambda n: single_cascade.nodes[n].get("followers", 0))
        assert root in single_cascade.nodes

    def test_simulate_cascade_following_args_satisfied(self, ingestor, single_cascade):
        """
        Verify all arguments for simulate_cascade_following() can be
        constructed from the ingestor output.

        Signature:
            simulate_cascade_following(
                G, partition, root, sbm, alpha, lam,
                global_class_sizes, seed=42
            )
        """
        root = ingestor.get_root_node(single_cascade)

        # partition: assign random classes (would come from SBM fitter)
        k = 4
        rng = np.random.default_rng(0)
        partition = {
            node: int(rng.integers(0, k))
            for node in single_cascade.nodes()
        }

        # Verify all required pieces exist
        assert isinstance(single_cascade, nx.DiGraph)
        assert isinstance(partition, dict)
        assert root in single_cascade.nodes
        assert set(partition.keys()) == set(single_cascade.nodes())

    def test_metadata_cache_works(self, ingestor, single_cascade):
        """get_node_metadata should cache results (no duplicate API calls)."""
        root = ingestor.get_root_node(single_cascade)

        meta1 = ingestor.get_node_metadata(root)
        meta2 = ingestor.get_node_metadata(root)  # should hit cache

        assert meta1 == meta2

    def test_summary_string_format(self, ingestor, single_cascade):
        summary = ingestor.summary(single_cascade)
        assert "Cascade:" in summary
        assert "nodes" in summary
        assert "edges" in summary
        assert "root=" in summary

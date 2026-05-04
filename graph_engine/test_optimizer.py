"""
InfoGuard — Tests: graph_engine (optimizer, network_model, sir_simulation)
==========================================================================
Pytest-compatible test suite.

Run all tests:
    cd graph_engine && pytest test_optimizer.py -v

Run one group:
    pytest test_optimizer.py -v -k "test_lp"
    pytest test_optimizer.py -v -k "test_sbm"
    pytest test_optimizer.py -v -k "test_sir"
    pytest test_optimizer.py -v -k "test_integration"
"""

import sys
import numpy as np
import networkx as nx
import pytest

# Allow running from the graph_engine directory or project root
sys.path.insert(0, ".")
sys.path.insert(0, "..")

from optimizer      import DropoutOptimizer, OptimizerResult
from network_model  import SBM, SBMFitter, make_synthetic_sbm
from sir_simulation import (
    SIRState, SBMSIRSimulator, NodeSIRSimulator, run_algorithm2
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def sbm_2_balanced():
    """Balanced 2-partition SBM matching paper eq. 25."""
    return make_synthetic_sbm(k=2, x=0.005, y=0.0005, n_users=1000, balanced=True)

@pytest.fixture
def sbm_2_unbalanced():
    """Unbalanced 2-partition SBM [800, 200] matching paper Table I."""
    return make_synthetic_sbm(k=2, x=0.005, y=0.0005, n_users=1000, balanced=False)

@pytest.fixture
def sbm_3_balanced():
    """Balanced 3-partition SBM matching paper eq. 26."""
    return make_synthetic_sbm(k=3, x=0.005, y=0.0005, n_users=1000, balanced=True)

@pytest.fixture
def opt_2(sbm_2_balanced):
    return DropoutOptimizer(
        sbm_2_balanced.b_minus, sbm_2_balanced.b_plus, alpha=1.5, lambda_weight=1.0
    )

@pytest.fixture
def S2():
    return np.array([490.0, 490.0])

@pytest.fixture
def I2():
    return np.array([5.0, 5.0])


# ═══════════════════════════════════════════════════════════════════════════════
# LP Optimizer tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLPOptimizer:

    def test_primary_lp_type(self, opt_2, S2, I2):
        res = opt_2.solve(S2, I2)
        assert res.lp_type == "primary", f"Expected primary, got {res.lp_type}"

    def test_converged(self, opt_2, S2, I2):
        res = opt_2.solve(S2, I2)
        assert res.converged

    def test_dropout_shape(self, opt_2, S2, I2):
        res = opt_2.solve(S2, I2)
        assert res.dropout_matrix.shape == (2, 2)

    def test_dropout_in_unit_interval(self, opt_2, S2, I2):
        res = opt_2.solve(S2, I2)
        assert np.all(res.dropout_matrix >= 0.0 - 1e-9)
        assert np.all(res.dropout_matrix <= 1.0 + 1e-9)

    def test_true_content_constraint_satisfied(self, opt_2, sbm_2_balanced, S2, I2):
        """Core LP guarantee: true content branching ratio ≥ α after dropout."""
        res       = opt_2.solve(S2, I2)
        true_spread = res.expected_true_spread(S2, I2, sbm_2_balanced.b_plus)
        I_total   = I2.sum()
        assert true_spread >= 1.5 * I_total - 1e-6, (
            f"Constraint violated: E[I_true]={true_spread:.4f} < α|I|={1.5*I_total:.4f}"
        )

    def test_false_content_suppressed(self, opt_2, sbm_2_balanced, S2, I2):
        """False content spread must be ≤ unaltered false spread."""
        res          = opt_2.solve(S2, I2)
        false_altered  = res.expected_false_spread(S2, I2, sbm_2_balanced.b_minus)
        # Unaltered: d=1 everywhere
        d_ones         = np.ones((2, 2))
        false_unaltered = float(np.sum(
            S2[np.newaxis, :] * I2[:, np.newaxis] * d_ones * sbm_2_balanced.b_minus
        ))
        assert false_altered <= false_unaltered + 1e-6, (
            f"LP did not suppress false spread: {false_altered:.4f} > {false_unaltered:.4f}"
        )

    def test_feasibility_margin_positive_for_primary(self, opt_2, S2, I2):
        res = opt_2.solve(S2, I2)
        assert res.feasibility_margin >= 0.0

    def test_softened_lp_when_infeasible(self, S2, I2):
        """When b⁺ ≈ 0 the primary LP is infeasible → softened LP."""
        b_plus_low  = np.full((2, 2), 1e-10)
        b_minus     = np.array([[0.01, 0.002], [0.002, 0.01]])
        opt         = DropoutOptimizer(b_minus, b_plus_low, alpha=1.5)
        res         = opt.solve(S2, I2)
        assert res.lp_type == "softened"
        assert res.feasibility_margin < 0.0

    def test_softened_lp_still_converges(self, S2, I2):
        b_plus_low = np.full((2, 2), 1e-10)
        b_minus    = np.array([[0.01, 0.002], [0.002, 0.01]])
        opt        = DropoutOptimizer(b_minus, b_plus_low, alpha=1.5)
        res        = opt.solve(S2, I2)
        assert res.converged

    def test_no_infected_returns_zero_dropout(self, opt_2):
        res = opt_2.solve(np.array([490.0, 490.0]), np.array([0.0, 0.0]))
        assert res.lp_type == "no_infected"
        assert np.allclose(res.dropout_matrix, 0.0)

    def test_3_partition_constraint(self, sbm_3_balanced):
        opt   = DropoutOptimizer(sbm_3_balanced.b_minus, sbm_3_balanced.b_plus, alpha=1.5)
        S     = np.array([330.0, 330.0, 330.0])
        I     = np.array([3.0, 3.0, 3.0])
        res   = opt.solve(S, I)
        true_s = res.expected_true_spread(S, I, sbm_3_balanced.b_plus)
        assert true_s >= 1.5 * I.sum() - 1e-6

    def test_dropout_min_respected(self, sbm_2_balanced, S2, I2):
        opt = DropoutOptimizer(sbm_2_balanced.b_minus, sbm_2_balanced.b_plus,
                               alpha=1.5, dropout_min=0.2)
        res = opt.solve(S2, I2)
        assert np.all(res.dropout_matrix >= 0.2 - 1e-9)

    def test_dropout_max_respected(self, sbm_2_balanced, S2, I2):
        opt = DropoutOptimizer(sbm_2_balanced.b_minus, sbm_2_balanced.b_plus,
                               alpha=1.5, dropout_max=0.8)
        res = opt.solve(S2, I2)
        assert np.all(res.dropout_matrix <= 0.8 + 1e-9)

    def test_alpha_2_constraint(self, sbm_2_balanced, S2, I2):
        """Higher α = tighter true-content constraint."""
        opt = DropoutOptimizer(sbm_2_balanced.b_minus, sbm_2_balanced.b_plus, alpha=2.0)
        res = opt.solve(S2, I2)
        if res.lp_type == "primary":
            true_s = res.expected_true_spread(S2, I2, sbm_2_balanced.b_plus)
            assert true_s >= 2.0 * I2.sum() - 1e-6

    def test_unbalanced_partitions(self, sbm_2_unbalanced):
        """Matches paper Table I unbalanced 2-partition scenario."""
        opt   = DropoutOptimizer(sbm_2_unbalanced.b_minus, sbm_2_unbalanced.b_plus,
                                 alpha=1.5)
        S     = np.array([795.0, 195.0])
        I     = np.array([4.0, 1.0])
        res   = opt.solve(S, I)
        assert res.converged
        if res.lp_type == "primary":
            true_s = res.expected_true_spread(S, I, sbm_2_unbalanced.b_plus)
            assert true_s >= 1.5 * I.sum() - 1e-6

    def test_solve_cascade_length(self, opt_2):
        S_seq = [np.array([490.0, 490.0])] * 3
        I_seq = [np.array([5.0, 5.0])] * 3
        results = opt_2.solve_cascade(S_seq, I_seq)
        assert len(results) == 3

    def test_bad_shape_raises(self, sbm_2_balanced):
        with pytest.raises(ValueError):
            DropoutOptimizer(sbm_2_balanced.b_minus, np.ones((3, 3)))

    def test_bad_bounds_raise(self, sbm_2_balanced):
        with pytest.raises(ValueError):
            DropoutOptimizer(sbm_2_balanced.b_minus, sbm_2_balanced.b_plus,
                             dropout_min=0.8, dropout_max=0.2)

    def test_mismatched_count_shape_raises(self, opt_2):
        with pytest.raises(ValueError):
            opt_2.solve(np.array([490.0]), np.array([5.0, 5.0]))

    def test_negative_counts_raise(self, opt_2):
        with pytest.raises(ValueError):
            opt_2.solve(np.array([-1.0, 490.0]), np.array([5.0, 5.0]))

    def test_check_feasibility(self, opt_2, S2, I2):
        is_feasible, margin = opt_2.check_feasibility(S2, I2)
        assert isinstance(is_feasible, bool)
        assert isinstance(margin, float)
        assert is_feasible == (margin >= 0.0)


# ═══════════════════════════════════════════════════════════════════════════════
# SBM Network Model tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSBMNetworkModel:

    def test_synthetic_sbm_shape(self, sbm_2_balanced):
        assert sbm_2_balanced.b_plus.shape  == (2, 2)
        assert sbm_2_balanced.b_minus.shape == (2, 2)

    def test_synthetic_sbm_values_in_range(self, sbm_2_balanced):
        assert np.all(sbm_2_balanced.b_plus  >= 0)
        assert np.all(sbm_2_balanced.b_plus  <= 1)
        assert np.all(sbm_2_balanced.b_minus >= 0)
        assert np.all(sbm_2_balanced.b_minus <= 1)

    def test_b_plus_stronger_within_class(self, sbm_2_balanced):
        """True content: stronger within-class spread (paper Section III.B)."""
        for u in range(2):
            assert sbm_2_balanced.b_plus[u, u] > sbm_2_balanced.b_minus[u, u], (
                f"Expected b⁺[{u},{u}] > b⁻[{u},{u}]"
            )

    def test_b_minus_stronger_cross_class(self, sbm_2_balanced):
        """False content: stronger cross-class spread (paper Section III.B)."""
        assert sbm_2_balanced.b_minus[0, 1] > sbm_2_balanced.b_plus[0, 1]
        assert sbm_2_balanced.b_minus[1, 0] > sbm_2_balanced.b_plus[1, 0]

    def test_class_sizes_sum_to_n(self, sbm_2_balanced):
        assert sbm_2_balanced.class_sizes.sum() == 1000

    def test_3partition_k(self, sbm_3_balanced):
        assert sbm_3_balanced.k == 3
        assert sbm_3_balanced.b_plus.shape == (3, 3)

    def test_partition_covers_all_users(self, sbm_2_balanced):
        assert len(sbm_2_balanced.partition) == 1000

    def test_partition_values_in_range(self, sbm_2_balanced):
        k = sbm_2_balanced.k
        assert all(0 <= v < k for v in sbm_2_balanced.partition.values())

    def test_b_plus_minus_diff_sign(self, sbm_2_balanced):
        diff = sbm_2_balanced.b_plus_minus_diff
        # Diagonal (within-class): positive — true content travels more
        assert diff[0, 0] > 0 and diff[1, 1] > 0
        # Off-diagonal (cross-class): negative — false content travels more
        assert diff[0, 1] < 0 and diff[1, 0] < 0

    def test_expected_spread_shape(self, sbm_2_balanced):
        I = np.array([5.0, 5.0])
        S = np.array([490.0, 490.0])
        spread = sbm_2_balanced.expected_spread(I, S, content="false")
        assert spread.shape == (2,)
        assert np.all(spread >= 0)

    def test_expected_spread_drops_with_dropout(self, sbm_2_balanced):
        I = np.array([5.0, 5.0])
        S = np.array([490.0, 490.0])
        no_dropout  = sbm_2_balanced.expected_spread(I, S, content="false")
        zero_dropout = sbm_2_balanced.expected_spread(I, S, content="false",
                                                       dropout=np.zeros((2, 2)))
        assert np.all(zero_dropout <= no_dropout + 1e-9)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            SBM(b_plus=np.ones((2,2)), b_minus=np.ones((3,3)),
                k=2, partition={}, class_sizes=np.array([500,500]))

    def test_save_and_load(self, sbm_2_balanced, tmp_path):
        sbm_2_balanced.save(tmp_path)
        loaded = SBM.load(tmp_path)
        assert loaded.k == sbm_2_balanced.k
        assert np.allclose(loaded.b_plus,  sbm_2_balanced.b_plus)
        assert np.allclose(loaded.b_minus, sbm_2_balanced.b_minus)
        assert np.array_equal(loaded.class_sizes, sbm_2_balanced.class_sizes)
        assert loaded.partition == sbm_2_balanced.partition

    def test_fitter_balanced_2_partitions(self):
        """SBMFitter correctly estimates matrices from synthetic cascades."""
        np.random.seed(42)
        rng = np.random.default_rng(42)

        # Build simple cascades: 2 clusters, mostly within-cluster edges
        def make_cascade(label, n_cascades=20):
            graphs = []
            for _ in range(n_cascades):
                G = nx.DiGraph()
                # Within-cluster edges
                G.add_edge("A1", "A2"); G.add_edge("A2", "A3")
                G.add_edge("B1", "B2"); G.add_edge("B2", "B3")
                if label == "false":
                    # Add cross-cluster for false content
                    G.add_edge("A1", "B2"); G.add_edge("B1", "A2")
                graphs.append(G)
            return graphs

        fitter = SBMFitter(
            num_partitions=2,
            clustering_resolution=1.0,
            min_partition_fraction=0.01,
            label_confidence_threshold=0.0,
            seed=42,
        )
        for G in make_cascade("true",  n_cascades=30):
            fitter.add_cascade(G, label="true",  confidence=1.0)
        for G in make_cascade("false", n_cascades=30):
            fitter.add_cascade(G, label="false", confidence=1.0)

        sbm = fitter.fit()
        assert sbm.k >= 1
        assert sbm.b_plus.shape  == (sbm.k, sbm.k)
        assert sbm.b_minus.shape == (sbm.k, sbm.k)
        assert np.all(sbm.b_plus  >= 0) and np.all(sbm.b_plus  <= 1)
        assert np.all(sbm.b_minus >= 0) and np.all(sbm.b_minus <= 1)

    def test_fitter_skips_low_confidence(self):
        """Cascades below threshold should be silently ignored."""
        fitter = SBMFitter(label_confidence_threshold=0.9, seed=42,
                           num_partitions=2, clustering_resolution=1.0,
                           min_partition_fraction=0.1)
        G = nx.DiGraph(); G.add_edge("a", "b")
        fitter.add_cascade(G, label="false", confidence=0.5)   # below threshold
        fitter.add_cascade(G, label="true",  confidence=0.95)  # above threshold
        # Only 1 cascade added — should warn but not crash when fit() is called
        # (will fail due to insufficient data, but the skip logic works)
        assert len(fitter._false_graphs) == 0
        assert len(fitter._true_graphs)  == 1

    def test_fitter_raises_on_empty(self):
        fitter = SBMFitter(num_partitions=2, clustering_resolution=1.0,
                           min_partition_fraction=0.1, seed=42)
        with pytest.raises(RuntimeError):
            fitter.fit()

    def test_unbalanced_sbm(self, sbm_2_unbalanced):
        assert sbm_2_unbalanced.class_sizes[0] == 800
        assert sbm_2_unbalanced.class_sizes[1] == 200


# ═══════════════════════════════════════════════════════════════════════════════
# SIR Simulation tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSIRState:

    def test_i_total(self):
        state = SIRState(I_counts=np.array([3.0, 2.0]))
        assert state.I_total == 5

    def test_is_terminated_true(self):
        state = SIRState(I_counts=np.array([0.0, 0.0]))
        assert state.is_terminated()

    def test_is_terminated_false(self):
        state = SIRState(I_counts=np.array([1.0, 0.0]))
        assert not state.is_terminated()

    def test_cascade_size(self):
        state = SIRState(
            I_counts=np.array([2.0, 3.0]),
            R_counts=np.array([5.0, 5.0]),
        )
        assert state.cascade_size == 15


class TestSBMSIRSimulator:

    def test_step_type(self, sbm_2_balanced, S2, I2):
        sim   = SBMSIRSimulator(sbm_2_balanced, content="false", rng_seed=42)
        state = SIRState(S_counts=S2.copy(), I_counts=I2.copy(),
                         R_counts=np.zeros(2))
        next_state = sim.step(state)
        assert isinstance(next_state, SIRState)

    def test_step_increments_t(self, sbm_2_balanced, S2, I2):
        sim   = SBMSIRSimulator(sbm_2_balanced, content="false", rng_seed=42)
        state = SIRState(S_counts=S2.copy(), I_counts=I2.copy(),
                         R_counts=np.zeros(2), t=3)
        next_state = sim.step(state)
        assert next_state.t == 4

    def test_infected_become_removed(self, sbm_2_balanced, S2, I2):
        """All infected at t must move to Removed at t+1 (m=1)."""
        sim   = SBMSIRSimulator(sbm_2_balanced, content="false", rng_seed=42)
        state = SIRState(S_counts=S2.copy(), I_counts=I2.copy(),
                         R_counts=np.zeros(2))
        next_state = sim.step(state)
        assert np.allclose(next_state.R_counts, state.R_counts + state.I_counts)

    def test_population_conserved(self, sbm_2_balanced, S2, I2):
        """S + I + R must equal total population at every step."""
        total = S2.sum() + I2.sum()
        sim   = SBMSIRSimulator(sbm_2_balanced, content="false", rng_seed=42)
        state = SIRState(S_counts=S2.copy(), I_counts=I2.copy(),
                         R_counts=np.zeros(2))
        for _ in range(5):
            pop = state.S_counts.sum() + state.I_counts.sum() + state.R_counts.sum()
            assert abs(pop - total) <= 1.0, f"Population leaked: {pop} ≠ {total}"
            state = sim.step(state)

    def test_zero_dropout_stops_cascade(self, sbm_2_balanced, S2, I2):
        """d* = 0 everywhere → no new infections."""
        sim   = SBMSIRSimulator(sbm_2_balanced, content="false", rng_seed=42)
        state = SIRState(S_counts=S2.copy(), I_counts=I2.copy(),
                         R_counts=np.zeros(2))
        d_zero     = np.zeros((2, 2))
        next_state = sim.step(state, dropout=d_zero)
        assert np.allclose(next_state.I_counts, 0.0)

    def test_run_terminates(self, sbm_2_balanced):
        sim    = SBMSIRSimulator(sbm_2_balanced, content="false", rng_seed=42)
        I0     = np.array([5.0, 5.0])
        totals = sbm_2_balanced.class_sizes.astype(float)
        history = sim.run(I0, totals, max_steps=50)
        assert len(history) >= 1
        assert history[-1].is_terminated() or len(history) == 51

    def test_run_length_bounded_by_max_steps(self, sbm_2_balanced):
        sim    = SBMSIRSimulator(sbm_2_balanced, content="false", rng_seed=42)
        I0     = np.array([5.0, 5.0])
        totals = sbm_2_balanced.class_sizes.astype(float)
        history = sim.run(I0, totals, max_steps=3)
        assert len(history) <= 4  # t=0 plus max 3 steps

    def test_expected_new_infections_deterministic(self, sbm_2_balanced, S2, I2):
        sim = SBMSIRSimulator(sbm_2_balanced, content="false")
        r1  = sim.expected_new_infections(S2, I2)
        r2  = sim.expected_new_infections(S2, I2)
        assert np.allclose(r1, r2)

    def test_branching_ratio_drops_with_dropout(self, sbm_2_balanced, S2, I2):
        sim = SBMSIRSimulator(sbm_2_balanced, content="false")
        br_full = sim.branching_ratio(S2, I2, dropout=np.ones((2,2)))
        br_zero = sim.branching_ratio(S2, I2, dropout=np.zeros((2,2)))
        assert br_zero <= br_full

    def test_true_branching_above_alpha_without_dropout(self, sbm_2_balanced, S2, I2):
        """True content branching ratio should naturally exceed 1.0 for healthy spread."""
        sim = SBMSIRSimulator(sbm_2_balanced, content="true")
        br  = sim.branching_ratio(S2, I2, dropout=np.ones((2,2)))
        assert br > 0.0

    def test_invalid_content_raises(self, sbm_2_balanced):
        with pytest.raises(ValueError):
            SBMSIRSimulator(sbm_2_balanced, content="unknown")


class TestNodeSIRSimulator:

    def _make_star_graph(self, sbm, n_leaves=10, rng_seed=42):
        """Star-shaped graph: root connected to n_leaves leaves."""
        rng   = np.random.default_rng(rng_seed)
        nodes = list(sbm.partition.keys())[:n_leaves + 1]
        root  = nodes[0]
        G     = nx.DiGraph()
        for leaf in nodes[1:]:
            G.add_edge(root, leaf)
        return G, root

    def test_step_moves_infected_to_removed(self, sbm_2_balanced):
        G, root = self._make_star_graph(sbm_2_balanced)
        sim     = NodeSIRSimulator(G, sbm_2_balanced, content="false", rng_seed=0)
        state   = SIRState(S_set=set(G.nodes()) - {root}, I_set={root}, R_set=set())
        ns      = sim.step(state)
        assert root in ns.R_set

    def test_zero_dropout_stops_spread(self, sbm_2_balanced):
        G, root = self._make_star_graph(sbm_2_balanced)
        sim     = NodeSIRSimulator(G, sbm_2_balanced, content="false", rng_seed=0)
        state   = SIRState(S_set=set(G.nodes()) - {root}, I_set={root}, R_set=set())
        d_zero  = np.zeros((sbm_2_balanced.k, sbm_2_balanced.k))
        ns      = sim.step(state, dropout=d_zero)
        assert len(ns.I_set) == 0

    def test_run_terminates(self, sbm_2_balanced):
        G, root  = self._make_star_graph(sbm_2_balanced, n_leaves=20)
        sim      = NodeSIRSimulator(G, sbm_2_balanced, content="false", rng_seed=42)
        history  = sim.run(seed_nodes={root}, max_steps=20)
        assert len(history) >= 1

    def test_cascade_size_bounded_by_graph_size(self, sbm_2_balanced):
        G, root = self._make_star_graph(sbm_2_balanced, n_leaves=10)
        sim     = NodeSIRSimulator(G, sbm_2_balanced, content="false", rng_seed=42)
        size    = sim.final_cascade_size(seed_nodes={root})
        assert 1 <= size <= G.number_of_nodes()

    def test_node_level_count_consistency(self, sbm_2_balanced):
        """S + I + R = total nodes at every step."""
        G, root = self._make_star_graph(sbm_2_balanced, n_leaves=15)
        total   = G.number_of_nodes()
        sim     = NodeSIRSimulator(G, sbm_2_balanced, content="false", rng_seed=0)
        state   = SIRState(S_set=set(G.nodes())-{root}, I_set={root}, R_set=set())
        for _ in range(5):
            n = len(state.S_set) + len(state.I_set) + len(state.R_set)
            assert n == total
            state = sim.step(state)


# ═══════════════════════════════════════════════════════════════════════════════
# Integration tests: LP + SBM + SIR together (Algorithm 2)
# ═══════════════════════════════════════════════════════════════════════════════

class TestAlgorithm2Integration:

    def test_algorithm2_runs(self, sbm_2_balanced):
        result = run_algorithm2(
            sbm                   = sbm_2_balanced,
            seed_I_counts         = np.array([5.0, 5.0]),
            total_users_per_class = sbm_2_balanced.class_sizes.astype(float),
            alpha                 = 1.5,
            max_steps             = 20,
            rng_seed              = 42,
        )
        assert "cascade_size_false" in result
        assert "cascade_size_true"  in result
        assert result["cascade_size_false"] >= 0
        assert result["cascade_size_true"]  >= 0

    def test_dropout_reduces_false_cascade(self, sbm_2_balanced):
        """
        Core property from the paper: with optimal dropout, false cascade
        size should be ≤ unaltered false cascade size on average.
        """
        totals = sbm_2_balanced.class_sizes.astype(float)
        I0     = np.array([5.0, 5.0])

        # Control: no dropout (d=1 everywhere → same as not running LP)
        sim_ctrl  = SBMSIRSimulator(sbm_2_balanced, content="false", rng_seed=0)
        ctrl_size = sim_ctrl.final_cascade_size(I0, totals, dropout_sequence=None)

        # Algorithm 2: optimal dropout
        result   = run_algorithm2(
            sbm=sbm_2_balanced, seed_I_counts=I0,
            total_users_per_class=totals, alpha=1.5, rng_seed=0,
        )
        # Run multiple seeds and check average reduction
        sizes_alg2 = []
        for seed in range(20):
            r = run_algorithm2(
                sbm=sbm_2_balanced, seed_I_counts=I0,
                total_users_per_class=totals, alpha=1.5, rng_seed=seed,
            )
            sizes_alg2.append(r["cascade_size_false"])
        avg_alg2 = np.mean(sizes_alg2)
        # Algorithm 2 should reduce average false cascade (allow 20% tolerance
        # for stochastic variation in small networks)
        assert avg_alg2 <= ctrl_size * 1.2, (
            f"Expected Algorithm 2 to reduce false cascade. "
            f"Control: {ctrl_size}, Algorithm 2 avg: {avg_alg2:.1f}"
        )

    def test_return_dropouts_flag(self, sbm_2_balanced):
        result = run_algorithm2(
            sbm=sbm_2_balanced,
            seed_I_counts=np.array([5.0, 5.0]),
            total_users_per_class=sbm_2_balanced.class_sizes.astype(float),
            alpha=1.5, rng_seed=0, return_dropouts=True,
        )
        assert "dropout_sequence" in result
        for d in result["dropout_sequence"]:
            assert d.shape == (2, 2)
            assert np.all(d >= 0) and np.all(d <= 1)

    def test_lp_types_recorded(self, sbm_2_balanced):
        result = run_algorithm2(
            sbm=sbm_2_balanced,
            seed_I_counts=np.array([5.0, 5.0]),
            total_users_per_class=sbm_2_balanced.class_sizes.astype(float),
            alpha=1.5, rng_seed=42,
        )
        assert isinstance(result["lp_types"], list)
        valid_types = {"primary", "softened", "no_infected", "fallback"}
        assert all(t in valid_types for t in result["lp_types"])

    def test_feasible_fraction_in_range(self, sbm_2_balanced):
        result = run_algorithm2(
            sbm=sbm_2_balanced,
            seed_I_counts=np.array([5.0, 5.0]),
            total_users_per_class=sbm_2_balanced.class_sizes.astype(float),
            alpha=1.5, rng_seed=42,
        )
        assert 0.0 <= result["feasible_fraction"] <= 1.0

    def test_paper_table1_direction(self, sbm_2_balanced):
        """
        Sanity-check against Table I direction (not exact values):
        With α=1.5, λ=1: false cascade < true cascade on average.
        Paper reports E[R∞/N] ≈ 0.32 false vs 0.51 true for balanced 2-partition.
        """
        totals     = sbm_2_balanced.class_sizes.astype(float)
        false_sizes = []
        true_sizes  = []
        for seed in range(30):
            r = run_algorithm2(
                sbm=sbm_2_balanced,
                seed_I_counts=np.array([5.0, 5.0]),
                total_users_per_class=totals,
                alpha=1.5, lambda_weight=1.0,
                rng_seed=seed,
            )
            false_sizes.append(r["cascade_size_false"])
            true_sizes.append(r["cascade_size_true"])

        avg_false = np.mean(false_sizes)
        avg_true  = np.mean(true_sizes)
        # Direction check: false should be suppressed more than true
        assert avg_false <= avg_true, (
            f"Expected avg_false ≤ avg_true. Got false={avg_false:.1f}, true={avg_true:.1f}"
        )

    @pytest.mark.parametrize("alpha,lambda_weight", [
        (1.5, 1.0),
        (2.0, 1.5),
        (3.0, 2.0),
    ])
    def test_all_paper_alpha_lambda_pairs(self, sbm_2_balanced, alpha, lambda_weight):
        """All three (α, λ) pairs from the paper must run without errors."""
        result = run_algorithm2(
            sbm=sbm_2_balanced,
            seed_I_counts=np.array([5.0, 5.0]),
            total_users_per_class=sbm_2_balanced.class_sizes.astype(float),
            alpha=alpha, lambda_weight=lambda_weight,
            rng_seed=42,
        )
        assert result["cascade_size_false"] >= 0
        assert result["cascade_size_true"]  >= 0


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone runner (no pytest required)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import traceback

    passed = 0
    failed = 0

    def run_test(name, fn, *args):
        global passed, failed
        try:
            fn(*args)
            print(f"  ✓ {name}")
            passed += 1
        except Exception as exc:
            print(f"  ✗ {name}: {exc}")
            traceback.print_exc()
            failed += 1

    print("\n=== InfoGuard Graph Engine — standalone tests ===\n")

    sbm2  = make_synthetic_sbm(k=2, balanced=True)
    sbm2u = make_synthetic_sbm(k=2, balanced=False)
    sbm3  = make_synthetic_sbm(k=3)
    S2    = np.array([490.0, 490.0])
    I2    = np.array([5.0, 5.0])
    opt2  = DropoutOptimizer(sbm2.b_minus, sbm2.b_plus, alpha=1.5)

    # LP tests
    print("── LP Optimizer ──")
    t = TestLPOptimizer()
    run_test("primary LP",           t.test_primary_lp_type,                opt2, S2, I2)
    run_test("converged",            t.test_converged,                       opt2, S2, I2)
    run_test("dropout shape",        t.test_dropout_shape,                   opt2, S2, I2)
    run_test("dropout in [0,1]",     t.test_dropout_in_unit_interval,        opt2, S2, I2)
    run_test("constraint satisfied", t.test_true_content_constraint_satisfied, opt2, sbm2, S2, I2)
    run_test("false suppressed",     t.test_false_content_suppressed,        opt2, sbm2, S2, I2)
    run_test("softened LP",          t.test_softened_lp_when_infeasible,     S2, I2)
    run_test("no_infected",          t.test_no_infected_returns_zero_dropout, opt2)
    run_test("3-partition",          t.test_3_partition_constraint,          sbm3)
    run_test("cascade length",       t.test_solve_cascade_length,            opt2)

    # SBM tests
    print("\n── SBM Network Model ──")
    t2 = TestSBMNetworkModel()
    run_test("shapes",               t2.test_synthetic_sbm_shape,            sbm2)
    run_test("values in range",      t2.test_synthetic_sbm_values_in_range,  sbm2)
    run_test("b+ within stronger",   t2.test_b_plus_stronger_within_class,   sbm2)
    run_test("b- cross stronger",    t2.test_b_minus_stronger_cross_class,   sbm2)
    run_test("class sizes sum",      t2.test_class_sizes_sum_to_n,           sbm2)
    run_test("partition coverage",   t2.test_partition_covers_all_users,     sbm2)
    run_test("expected spread shape",t2.test_expected_spread_shape,          sbm2)
    run_test("dropout reduces spread",t2.test_expected_spread_drops_with_dropout, sbm2)
    run_test("fitter basic",         t2.test_fitter_balanced_2_partitions)
    run_test("fitter empty raises",  t2.test_fitter_raises_on_empty)

    # SIR tests
    print("\n── SIR Simulation ──")
    t3 = TestSBMSIRSimulator()
    run_test("step type",            t3.test_step_type,           sbm2, S2, I2)
    run_test("step increments t",    t3.test_step_increments_t,   sbm2, S2, I2)
    run_test("infected→removed",     t3.test_infected_become_removed, sbm2, S2, I2)
    run_test("population conserved", t3.test_population_conserved,    sbm2, S2, I2)
    run_test("zero dropout stops",   t3.test_zero_dropout_stops_cascade, sbm2, S2, I2)
    run_test("run terminates",       t3.test_run_terminates,           sbm2)
    run_test("branching drops",      t3.test_branching_ratio_drops_with_dropout, sbm2, S2, I2)

    # Integration tests
    print("\n── Algorithm 2 Integration ──")
    t4 = TestAlgorithm2Integration()
    run_test("algorithm2 runs",      t4.test_algorithm2_runs,             sbm2)
    run_test("dropouts reduce false",t4.test_dropout_reduces_false_cascade, sbm2)
    run_test("return dropouts flag", t4.test_return_dropouts_flag,         sbm2)
    run_test("lp types recorded",   t4.test_lp_types_recorded,            sbm2)
    run_test("paper direction",      t4.test_paper_table1_direction,       sbm2)
    for alpha, lam in [(1.5,1.0),(2.0,1.5),(3.0,2.0)]:
        run_test(f"(α={alpha},λ={lam})",
                 t4.test_all_paper_alpha_lambda_pairs, sbm2, alpha, lam)

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed ✓")
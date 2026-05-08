"""
InfoGuard — Graph Engine: SIR Cascade Simulator
================================================
Implements the discrete-time SIR propagation model from the paper
(Section II.B and III.B) at two levels of resolution:

1.  SBMSIRSimulator  — aggregate simulation over class counts.
    Fast. Matches the LP optimizer's approximation exactly.
    Used for: LP validation, feasibility checks, Table I/II reproduction.

2.  NodeSIRSimulator — node-level simulation on real graphs.
    Accurate. Uses actual user-to-user edges and partition assignments.
    Used for: WICO cascade evaluation, pipeline integration.

─── Paper model (Section II.B) ─────────────────────────────────────────────

Infectious period m = 1 (equivalent to independent cascade process).

At each step t:
    P(j ∈ I_{t+1} | j ∈ S_t) = 1 − ∏_{i ∈ I_t} (1 − p̃_ij)     eq. (1)

where p̃_ij = d_uv * b_uv for users i ∈ Cu, j ∈ Cv.

Large-N asymptotic approximation (eq. 14):
    P(j ∈ I_{t+1} | j ∈ S_t) ≈ 1 − exp(−∑_{i ∈ I_t} p̃_ij)

Aggregate version (eq. 15/16):
    E[|I^v_{t+1}|] = |S^v_t| * (1 − exp(−∑_u |I^u_t| * d_uv * b_uv))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import networkx as nx
import numpy as np

from network_model import SBM


# ── State container ───────────────────────────────────────────────────────────

@dataclass
class SIRState:
    """
    State of the SIR process at time t.

    Both representations are kept in sync:
      - Sets (S_set, I_set, R_set): used by NodeSIRSimulator
      - Class counts (S_counts, I_counts, R_counts): used by SBMSIRSimulator
        and the LP optimizer.
    """
    # ── Set representation (node IDs) ─────────────────────────────────────
    S_set: set = field(default_factory=set)   # susceptible
    I_set: set = field(default_factory=set)   # infected
    R_set: set = field(default_factory=set)   # removed

    # ── Aggregate class counts (shape: k) ─────────────────────────────────
    # S_counts[v] = |S^v_t|, etc.
    S_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    I_counts: np.ndarray = field(default_factory=lambda: np.array([]))
    R_counts: np.ndarray = field(default_factory=lambda: np.array([]))

    # Current step
    t: int = 0

    @property
    def I_total(self) -> int:
        return len(self.I_set) if self.I_set else int(self.I_counts.sum())

    @property
    def S_total(self) -> int:
        return len(self.S_set) if self.S_set else int(self.S_counts.sum())

    @property
    def R_total(self) -> int:
        return len(self.R_set) if self.R_set else int(self.R_counts.sum())

    @property
    def cascade_size(self) -> int:
        """R∞ — total users ever infected (R_set ∪ I_set at end)."""
        return self.R_total + self.I_total

    def is_terminated(self) -> bool:
        """Cascade ends when no infected users remain."""
        return self.I_total == 0


# ── SBM-level simulator (fast, LP-facing) ────────────────────────────────────

class SBMSIRSimulator:
    """
    Simulate SIR cascades at the class-count level using the paper's
    aggregate equations (15/16).

    This is the simulator the LP optimizer is designed for: the LP
    minimises the expected output of this simulator for false content
    while constraining its output for true content.

    At each step the simulator:
    1.  Takes current class counts (S^v_t, I^u_t) from the LP optimizer's
        perspective.
    2.  Applies the dropout matrix d*.
    3.  Draws new infections stochastically from Poisson(λ) ≈ Binomial(N, p)
        — the exact eq. (1) for finite N, or the asymptotic eq. (14) for
        large N (controlled by `use_asymptotic`).

    Parameters
    ----------
    sbm             : fitted SBM with b⁺ and b⁻ matrices
    content         : "true" or "false" — which SBM matrix to use
    use_asymptotic  : if True, use exp approximation (eq. 14); if False,
                      use exact product formula (eq. 1). Default True.
    rng_seed        : random seed for reproducible simulations
    """

    def __init__(
        self,
        sbm: SBM,
        content: str = "false",
        use_asymptotic: bool = True,
        rng_seed: int = 42,
    ) -> None:
        if content not in ("true", "false"):
            raise ValueError(f"content must be 'true' or 'false', got {content!r}")
        self.sbm            = sbm
        self.content        = content
        self.b              = sbm.b_plus if content == "true" else sbm.b_minus
        self.use_asymptotic = use_asymptotic
        self.rng            = np.random.default_rng(rng_seed)

    # ── Core step ──────────────────────────────────────────────────────────

    def step(
        self,
        state: SIRState,
        dropout: Optional[np.ndarray] = None,
    ) -> SIRState:
        """
        Advance the SIR model by one step using class-count dynamics.

        New infections are drawn per-class using the aggregate formula.
        All currently infected nodes move to Removed (m=1).

        Parameters
        ----------
        state   : current SIRState (uses S_counts and I_counts)
        dropout : (k, k) dropout matrix d*; None means d=1 (no dropout)

        Returns
        -------
        New SIRState at t+1
        """
        k = self.sbm.k
        d = dropout if dropout is not None else np.ones((k, k))
        S = state.S_counts.copy()
        I = state.I_counts.copy()
        R = state.R_counts.copy()

        if I.sum() == 0:
            return SIRState(S_counts=S, I_counts=I, R_counts=R, t=state.t + 1)

        # Expected new infections per class v (eq. 15/16)
        new_I_counts = self._draw_new_infections(S, I, d)

        # All infected at t move to Removed (m=1)
        new_R = R + I
        # New susceptible = old susceptible minus newly infected
        new_S = np.maximum(S - new_I_counts, 0)

        return SIRState(
            S_counts=new_S,
            I_counts=new_I_counts,
            R_counts=new_R,
            t=state.t + 1,
        )

    def run(
        self,
        initial_I_counts: np.ndarray,
        total_users_per_class: np.ndarray,
        dropout_sequence: Optional[list[np.ndarray]] = None,
        max_steps: int = 50,
    ) -> list[SIRState]:
        """
        Run a complete SIR cascade from seed infections.

        Parameters
        ----------
        initial_I_counts      : (k,) infected users per class at t=0
        total_users_per_class : (k,) total users per class (|Cu|)
        dropout_sequence      : list of (k,k) dropout matrices, one per step.
                                If shorter than the cascade, the last matrix
                                is repeated. None = no dropout.
        max_steps             : safety cap on cascade length

        Returns
        -------
        List of SIRState objects, one per step (including t=0).
        """
        S0 = total_users_per_class - initial_I_counts
        I0 = initial_I_counts.copy()
        R0 = np.zeros_like(I0)

        state = SIRState(S_counts=S0, I_counts=I0, R_counts=R0, t=0)
        history = [state]

        for step_idx in range(max_steps):
            if state.is_terminated():
                break
            if dropout_sequence is not None:
                idx = min(step_idx, len(dropout_sequence) - 1)
                d   = dropout_sequence[idx]
            else:
                d = None
            state = self.step(state, dropout=d)
            history.append(state)

        return history

    def final_cascade_size(
        self,
        initial_I_counts: np.ndarray,
        total_users_per_class: np.ndarray,
        dropout_sequence: Optional[list[np.ndarray]] = None,
        max_steps: int = 50,
    ) -> int:
        """Run cascade and return R∞ (total users ever infected)."""
        history = self.run(
            initial_I_counts, total_users_per_class,
            dropout_sequence, max_steps,
        )
        last = history[-1]
        return int(last.R_counts.sum() + last.I_counts.sum())

    # ── LP integration helper ───────────────────────────────────────────────

    def expected_new_infections(
        self,
        S_counts: np.ndarray,
        I_counts: np.ndarray,
        dropout: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Return E[|I^v_{t+1}|] for each class v (deterministic, no sampling).

        This is the quantity the LP optimizer minimises/constrains.
        Uses the asymptotic formula (eq. 15/16) regardless of use_asymptotic.
        """
        k = self.sbm.k
        d = dropout if dropout is not None else np.ones((k, k))
        # effective_rate[v] = ∑_u I^u * d_uv * b_uv
        effective_rate = (I_counts[:, np.newaxis] * d * self.b).sum(axis=0)
        return S_counts * (1.0 - np.exp(-effective_rate))

    def branching_ratio(
        self,
        S_counts: np.ndarray,
        I_counts: np.ndarray,
        dropout: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute E[|I_{t+1}|] / |I_t| — the branching ratio at this step.

        This is the quantity the LP constrains to be ≥ α for true content.
        """
        I_total = float(I_counts.sum())
        if I_total == 0:
            return 0.0
        new_I = self.expected_new_infections(S_counts, I_counts, dropout)
        return float(new_I.sum()) / I_total

    # ── Private ─────────────────────────────────────────────────────────────

    def _draw_new_infections(
        self,
        S: np.ndarray,
        I: np.ndarray,
        d: np.ndarray,
    ) -> np.ndarray:
        """
        Stochastically draw new infections per class from the SBM dynamics.

        For use_asymptotic=True (large N, default):
            infection_prob[v] = 1 - exp(- ∑_u I^u * d_uv * b_uv)
            new_I[v] ~ Binomial(S[v], infection_prob[v])

        For use_asymptotic=False (exact, eq. 1):
            Each individual in S[v] is infected independently with
            P = 1 - (1-d_uv*b_uv)^{I^u} per source class u, combined
            as 1 - prod_u (1-p_u). Modelled as Binomial with exact prob.
        """
        k = self.sbm.k
        new_I = np.zeros(k, dtype=np.float64)

        for v in range(k):
            if S[v] <= 0:
                continue
            if self.use_asymptotic:
                # Asymptotic: eq. (14) / (15)
                lam    = float(np.dot(I, d[:, v] * self.b[:, v]))
                p_infect = 1.0 - np.exp(-lam)
            else:
                # Exact: eq. (1)
                p_survive = 1.0
                for u in range(k):
                    if I[u] > 0:
                        p_survive *= (1.0 - d[u, v] * self.b[u, v]) ** I[u]
                p_infect = 1.0 - p_survive

            p_infect = float(np.clip(p_infect, 0.0, 1.0))
            n_susceptible = int(S[v])
            if n_susceptible > 0 and p_infect > 0:
                new_I[v] = float(self.rng.binomial(n_susceptible, p_infect))

        return new_I


# ── Node-level simulator (accurate, evaluation) ───────────────────────────────

class NodeSIRSimulator:
    """
    Simulate SIR cascades at the individual user level on a real graph.

    Used for WICO evaluation in Phase 4 — runs Algorithm 2 at each step
    by calling the LP optimizer, then applies the resulting dropout matrix
    to the actual edge-level infection probabilities.

    Parameters
    ----------
    G         : DiGraph of the real social network (WICO cascade graph).
                Nodes are user IDs; edges represent possible content transfers.
    sbm       : fitted SBM — provides b_uv as base infection probabilities
                and partition assignments.
    content   : "true" or "false"
    rng_seed  : reproducibility seed
    """

    def __init__(
        self,
        G: nx.DiGraph,
        sbm: SBM,
        content: str = "false",
        rng_seed: int = 42,
    ) -> None:
        self.G       = G
        self.sbm     = sbm
        self.content = content
        self.b       = sbm.b_plus if content == "true" else sbm.b_minus
        self.rng     = np.random.default_rng(rng_seed)

    def step(
        self,
        state: SIRState,
        dropout: Optional[np.ndarray] = None,
    ) -> SIRState:
        """
        Advance one SIR step at node level using eq. (1).

        For each susceptible node j:
            P(j infected) = 1 - ∏_{i ∈ I_t, (i,j) ∈ E} (1 - d*[u,v] * b_uv)
        where u = class(i), v = class(j).

        All infected nodes at t move to Removed (m=1).
        """
        k = self.sbm.k
        d = dropout if dropout is not None else np.ones((k, k))
        partition = self.sbm.partition

        new_I_set = set()
        for j in state.S_set:
            v = partition.get(j)
            if v is None or not (0 <= v < k):
                continue
            # Probability j survives (is NOT infected)
            p_survive = 1.0
            for i in self.G.predecessors(j):
                if i not in state.I_set:
                    continue
                u = partition.get(i)
                if u is None or not (0 <= u < k):
                    continue
                p_edge = float(np.clip(d[u, v] * self.b[u, v], 0.0, 1.0))
                p_survive *= (1.0 - p_edge)
            p_infect = 1.0 - p_survive
            if p_infect > 0 and self.rng.random() < p_infect:
                new_I_set.add(j)

        new_R_set = state.R_set | state.I_set
        new_S_set = state.S_set - new_I_set

        # Update class counts from sets
        new_I_counts, new_S_counts, new_R_counts = (
            self._counts_from_set(new_I_set, k),
            self._counts_from_set(new_S_set, k),
            self._counts_from_set(new_R_set, k),
        )

        return SIRState(
            S_set=new_S_set,    I_set=new_I_set,    R_set=new_R_set,
            S_counts=new_S_counts, I_counts=new_I_counts, R_counts=new_R_counts,
            t=state.t + 1,
        )

    def run(
        self,
        seed_nodes: set,
        dropout_sequence: Optional[list[np.ndarray]] = None,
        max_steps: int = 50,
    ) -> list[SIRState]:
        """
        Run a complete node-level SIR cascade from a set of seed nodes.

        Parameters
        ----------
        seed_nodes       : initial I_0 (set of node IDs)
        dropout_sequence : list of (k,k) dropout matrices per step
        max_steps        : safety cap

        Returns
        -------
        List of SIRState, one per step (including t=0).
        """
        k = self.sbm.k
        all_nodes = set(self.G.nodes())
        I0 = seed_nodes & all_nodes
        S0 = all_nodes - I0

        state = SIRState(
            S_set=S0, I_set=I0, R_set=set(),
            S_counts=self._counts_from_set(S0, k),
            I_counts=self._counts_from_set(I0, k),
            R_counts=np.zeros(k, dtype=np.float64),
            t=0,
        )
        history = [state]

        for step_idx in range(max_steps):
            if state.is_terminated():
                break
            if dropout_sequence is not None:
                idx = min(step_idx, len(dropout_sequence) - 1)
                d   = dropout_sequence[idx]
            else:
                d = None
            state = self.step(state, dropout=d)
            history.append(state)

        return history

    def final_cascade_size(
        self,
        seed_nodes: set,
        dropout_sequence: Optional[list[np.ndarray]] = None,
        max_steps: int = 50,
    ) -> int:
        """Run cascade and return R∞."""
        history = self.run(seed_nodes, dropout_sequence, max_steps)
        last = history[-1]
        return len(last.R_set) + len(last.I_set)

    def _counts_from_set(self, node_set: set, k: int) -> np.ndarray:
        """Convert a set of node IDs to per-class counts."""
        counts = np.zeros(k, dtype=np.float64)
        for node in node_set:
            cls = self.sbm.partition.get(node)
            if cls is not None and 0 <= cls < k:
                counts[cls] += 1
        return counts


# ── Algorithm 2 integration: run LP + SIR in a loop ─────────────────────────

def run_algorithm2(
    sbm: SBM,
    seed_I_counts: np.ndarray,
    total_users_per_class: np.ndarray,
    alpha: float = 1.5,
    lambda_weight: float = 1.0,
    max_steps: int = 50,
    rng_seed: int = 42,
    return_dropouts: bool = False,
) -> dict:
    """
    Run the complete Algorithm 2 loop from the paper:
    at each step, solve the LP → get d* → simulate one SIR step.

    Runs BOTH false and true content simulators simultaneously using the
    SAME d* sequence, matching the paper's evaluation methodology.

    Parameters
    ----------
    sbm                   : fitted SBM
    seed_I_counts         : (k,) initial infected users per class
    total_users_per_class : (k,) |Cu| for each class
    alpha, lambda_weight  : LP parameters
    max_steps             : cascade length cap
    rng_seed              : reproducibility
    return_dropouts       : if True, include the d* sequence in output

    Returns
    -------
    dict with keys:
        cascade_size_false : R∞ for false content
        cascade_size_true  : R∞ for true content
        n_steps            : actual cascade length
        lp_types           : list of LP types used per step
        feasible_fraction  : fraction of steps where primary LP was used
        dropout_sequence   : list of d* matrices (only if return_dropouts)
    """
    # Import here to avoid circular imports when modules are split across files
    from optimizer import DropoutOptimizer

    opt        = DropoutOptimizer(sbm.b_minus, sbm.b_plus, alpha=alpha,
                                  lambda_weight=lambda_weight)
    sim_false  = SBMSIRSimulator(sbm, content="false", rng_seed=rng_seed)
    sim_true   = SBMSIRSimulator(sbm, content="true",  rng_seed=rng_seed + 1)

    S0 = total_users_per_class - seed_I_counts
    state_false = SIRState(S_counts=S0.copy(), I_counts=seed_I_counts.copy(),
                           R_counts=np.zeros_like(S0))
    state_true  = SIRState(S_counts=S0.copy(), I_counts=seed_I_counts.copy(),
                           R_counts=np.zeros_like(S0))

    lp_types       = []
    dropout_seq    = []
    primary_count  = 0

    for _ in range(max_steps):
        if state_false.is_terminated() and state_true.is_terminated():
            break

        # LP uses false-content state (the quantity being minimised)
        lp_result = opt.solve(state_false.S_counts, state_false.I_counts)
        d_star    = lp_result.dropout_matrix
        lp_types.append(lp_result.lp_type)
        if lp_result.lp_type == "primary":
            primary_count += 1
        if return_dropouts:
            dropout_seq.append(d_star.copy())

        state_false = sim_false.step(state_false, dropout=d_star)
        state_true  = sim_true.step(state_true,   dropout=d_star)

    n_steps = len(lp_types)
    result  = {
        "cascade_size_false": int(state_false.R_counts.sum() + state_false.I_counts.sum()),
        "cascade_size_true":  int(state_true.R_counts.sum()  + state_true.I_counts.sum()),
        "n_steps":            n_steps,
        "lp_types":           lp_types,
        "feasible_fraction":  primary_count / max(1, n_steps),
    }
    if return_dropouts:
        result["dropout_sequence"] = dropout_seq
    return result
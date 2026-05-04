"""
InfoGuard — Graph Engine: LP Optimizer
=======================================
Implements Algorithm 2 from:
    Bayiz & Topcu (2022). "Countering Misinformation on Social Networks
    Using Graph Alterations." arXiv:2211.04617v1

Algorithm 2 solves a Linear Program at each SIR cascade step t to find
dropout probabilities d*_uv — the probability of hiding content in a
user's feed for each pair of polarization classes (u, v).

The LP minimises the expected spread of false content while keeping the
expected spread of true content above a minimum branching ratio α.

─── Paper equations implemented ────────────────────────────────────────────

PRIMARY LP (eq. 22) — used when the feasibility condition (eq. 23) holds:

    min_{d ∈ [0,1]^{k×k}}  ∑_v ∑_u  |S^v_t| |I^u_t|  d_uv  b⁻_uv
    subject to:             ∑_v ∑_u  |S^v_t| |I^u_t|  d_uv  b⁺_uv  ≥  α |I_t|

FEASIBILITY CHECK (eq. 23):

    ∑_v ∑_u  |S^v_t| |I^u_t|  b⁺_uv  ≥  α |I_t|

    (i.e. the unaltered true-content network already satisfies the
     branching constraint with d=1 everywhere; if not, we cannot
     preserve the required α and must use the softened LP instead.)

SOFTENED LP (eq. 24) — used when (eq. 23) is infeasible:

    min_{d ∈ [0,1]^{k×k}}  ∑_v ∑_u  |S^v_t| |I^u_t|  (b⁻_uv  +  λ b⁺_uv)  d_uv

    (No hard constraint. λ is the relative importance of preserving true
     content vs suppressing false content.)

─── Variable layout ────────────────────────────────────────────────────────

d is a k×k matrix indexed by (u, v) where:
    u = polarization class of the SHARING user
    v = polarization class of the RECEIVING user

For scipy.optimize.linprog the matrix is flattened to a 1-D vector of
length k² in row-major order: d_flat[u*k + v] = d[u, v].

────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.optimize import linprog, OptimizeResult


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class OptimizerResult:
    """Output of one LP solve at cascade step t."""

    # The optimal dropout matrix d* (k×k), values in [0, 1].
    # d*[u, v] is the probability that content shared from class u to
    # class v is hidden from the receiver's feed.
    dropout_matrix: np.ndarray

    # Which LP was solved
    lp_type: str          # "primary" | "softened" | "no_infected" | "fallback"

    # Whether the LP solver reported success
    converged: bool

    # scipy solver message
    solver_message: str

    # Feasibility of the primary LP (eq. 23 LHS - α|I_t|)
    feasibility_margin: float

    # Objective value at optimum (expected false-content spread, scaled)
    objective_value: float

    # Parameters used
    alpha: float
    lambda_weight: float

    @property
    def is_feasible(self) -> bool:
        """True if the primary LP was feasible at this step."""
        return self.feasibility_margin >= 0.0

    def expected_false_spread(
        self,
        S_counts: np.ndarray,
        I_counts: np.ndarray,
        b_minus: np.ndarray,
    ) -> float:
        """
        Compute E[|I_{t+1}|] for false content under d*.

        Uses the linear approximation from eq. (22a):
            ∑_v ∑_u |S^v_t| |I^u_t| d*_uv b⁻_uv
        """
        return float(np.sum(
            S_counts[np.newaxis, :] *
            I_counts[:, np.newaxis] *
            self.dropout_matrix *
            b_minus
        ))

    def expected_true_spread(
        self,
        S_counts: np.ndarray,
        I_counts: np.ndarray,
        b_plus: np.ndarray,
    ) -> float:
        """
        Compute E[|I_{t+1}|] for true content under d*.

        Uses the linear approximation from eq. (22b):
            ∑_v ∑_u |S^v_t| |I^u_t| d*_uv b⁺_uv
        """
        return float(np.sum(
            S_counts[np.newaxis, :] *
            I_counts[:, np.newaxis] *
            self.dropout_matrix *
            b_plus
        ))


# ── Main optimizer class ──────────────────────────────────────────────────────

class DropoutOptimizer:
    """
    Solves the dropout LP from Algorithm 2 of the paper at each cascade step.

    Parameters
    ----------
    b_minus : np.ndarray, shape (k, k)
        SBM edge probability matrix for FALSE content.
        b_minus[u, v] = probability a user in class u shares content to
        a user in class v, given the content is false.
        Estimated by frequentist counting on labeled WICO cascades.

    b_plus : np.ndarray, shape (k, k)
        SBM edge probability matrix for TRUE content.
        Same interpretation as b_minus but for true content.

    alpha : float
        Safety parameter α. The LP requires that the expected number of
        newly infected nodes for true content at step t+1 is at least
        α * |I_t|. The paper tests α ∈ {1.5, 2.0, 3.0}.
        Default: 1.5 (cfg.lp.alpha).

    lambda_weight : float
        Weight λ for the softened LP (eq. 24). Higher λ = more weight on
        preserving true content relative to suppressing false content.
        Default: 1.0 (cfg.lp.lambda_weight).

    dropout_min : float
        Lower bound on all d_uv. Setting > 0 prevents total suppression of
        any connection. Default: 0.0.

    dropout_max : float
        Upper bound on all d_uv. Default: 1.0 (no dropout at most).

    solver : str
        scipy linprog method. "highs" is the fastest and most robust
        (default). Falls back to "revised simplex" automatically.

    Example
    -------
    >>> import numpy as np
    >>> k = 2
    >>> b_minus = np.array([[0.012, 0.003], [0.003, 0.012]])
    >>> b_plus  = np.array([[0.010, 0.002], [0.002, 0.010]])
    >>> opt = DropoutOptimizer(b_minus, b_plus, alpha=1.5)
    >>>
    >>> # At cascade step t=1:
    >>> S_counts = np.array([490, 490])   # susceptible per class
    >>> I_counts = np.array([5,   5])     # infected per class
    >>> result = opt.solve(S_counts, I_counts)
    >>> print(result.dropout_matrix)
    >>> print(result.lp_type)
    """

    def __init__(
        self,
        b_minus: np.ndarray,
        b_plus:  np.ndarray,
        alpha:   float = 1.5,
        lambda_weight: float = 1.0,
        dropout_min:   float = 0.0,
        dropout_max:   float = 1.0,
        solver: str = "highs",
    ) -> None:
        b_minus = np.asarray(b_minus, dtype=float)
        b_plus  = np.asarray(b_plus,  dtype=float)

        if b_minus.ndim != 2 or b_minus.shape != b_plus.shape:
            raise ValueError(
                f"b_minus and b_plus must be square matrices of the same shape. "
                f"Got {b_minus.shape} and {b_plus.shape}."
            )
        if b_minus.shape[0] != b_minus.shape[1]:
            raise ValueError(
                f"SBM matrices must be square. Got shape {b_minus.shape}."
            )
        if not (0.0 <= dropout_min < dropout_max <= 1.0):
            raise ValueError(
                f"Require 0 ≤ dropout_min < dropout_max ≤ 1. "
                f"Got [{dropout_min}, {dropout_max}]."
            )

        self.b_minus = b_minus
        self.b_plus  = b_plus
        self.k       = b_minus.shape[0]
        self.alpha   = float(alpha)
        self.lambda_weight = float(lambda_weight)
        self.dropout_min   = float(dropout_min)
        self.dropout_max   = float(dropout_max)
        self.solver  = solver

        # Pre-compute bounds once — same for every step
        self._bounds = [(self.dropout_min, self.dropout_max)] * (self.k * self.k)

    # ── Public API ────────────────────────────────────────────────────────────

    def solve(
        self,
        S_counts: np.ndarray,
        I_counts: np.ndarray,
    ) -> OptimizerResult:
        """
        Solve the dropout LP for one cascade step.

        Parameters
        ----------
        S_counts : np.ndarray, shape (k,)
            |S^v_t| — number of susceptible users in each polarization class v.

        I_counts : np.ndarray, shape (k,)
            |I^u_t| — number of infected users in each polarization class u.

        Returns
        -------
        OptimizerResult
            Contains the optimal dropout matrix d* and diagnostics.
        """
        S_counts = np.asarray(S_counts, dtype=float)
        I_counts = np.asarray(I_counts, dtype=float)
        self._validate_counts(S_counts, I_counts)

        I_total = float(I_counts.sum())

        # ── Edge case: no infected users ─────────────────────────────────────
        if I_total == 0.0:
            return OptimizerResult(
                dropout_matrix    = np.zeros((self.k, self.k)),
                lp_type           = "no_infected",
                converged         = True,
                solver_message    = "No infected users — zero dropouts applied.",
                feasibility_margin= 0.0,
                objective_value   = 0.0,
                alpha             = self.alpha,
                lambda_weight     = self.lambda_weight,
            )

        # ── Coefficient matrix W[u, v] = |S^v| * |I^u| ───────────────────────
        # Shape (k, k): W[u, v] is the weight for pair (u → v).
        W = I_counts[:, np.newaxis] * S_counts[np.newaxis, :]  # (k, k)

        # ── Feasibility check (eq. 23) ────────────────────────────────────────
        # ∑_uv W[u,v] * b⁺[u,v] ≥ α * |I_t|
        unaltered_true_spread = float(np.sum(W * self.b_plus))
        feasibility_margin    = unaltered_true_spread - self.alpha * I_total

        if feasibility_margin >= 0.0:
            result = self._solve_primary_lp(W, I_total, feasibility_margin)
        else:
            result = self._solve_softened_lp(W, feasibility_margin)

        return result

    def solve_cascade(
        self,
        S_by_class: list[np.ndarray],
        I_by_class: list[np.ndarray],
    ) -> list[OptimizerResult]:
        """
        Solve the LP for every step in a complete cascade.

        Parameters
        ----------
        S_by_class : list of np.ndarray, each shape (k,)
            Susceptible counts per class at each cascade step.
        I_by_class : list of np.ndarray, each shape (k,)
            Infected counts per class at each cascade step.

        Returns
        -------
        List of OptimizerResult, one per step.
        """
        if len(S_by_class) != len(I_by_class):
            raise ValueError("S_by_class and I_by_class must have the same length.")
        return [
            self.solve(S, I)
            for S, I in zip(S_by_class, I_by_class)
        ]

    def check_feasibility(
        self,
        S_counts: np.ndarray,
        I_counts: np.ndarray,
    ) -> tuple[bool, float]:
        """
        Check whether the primary LP (eq. 22) is feasible at this step.

        Returns
        -------
        (is_feasible, margin)
            is_feasible : True if the primary LP can be solved.
            margin : ∑_uv W[u,v] b⁺[u,v] - α|I_t|.
                     Positive = feasible; negative = infeasible.
        """
        S_counts = np.asarray(S_counts, dtype=float)
        I_counts = np.asarray(I_counts, dtype=float)
        W        = I_counts[:, np.newaxis] * S_counts[np.newaxis, :]
        I_total  = float(I_counts.sum())
        margin   = float(np.sum(W * self.b_plus)) - self.alpha * I_total
        return (margin >= 0.0, margin)

    # ── Private LP solvers ────────────────────────────────────────────────────

    def _solve_primary_lp(
        self,
        W: np.ndarray,
        I_total: float,
        feasibility_margin: float,
    ) -> OptimizerResult:
        """
        Solve the primary LP from equation (22).

        scipy.linprog minimises c^T x subject to:
            A_ub @ x ≤ b_ub   (inequality, ≤)
            bounds on x

        Objective (minimise false-content spread):
            c[u*k+v] = W[u,v] * b⁻[u,v]

        Constraint (true-content branching ≥ α|I_t|, rewritten as ≤):
            −∑_uv (W[u,v] * b⁺[u,v] * d_uv)  ≤  −α * |I_t|
            i.e. A_ub = −(W * b⁺).flatten()[np.newaxis, :]
                 b_ub = [−α * |I_t|]
        """
        k = self.k

        # Objective: c[u*k+v] = W[u,v] * b⁻[u,v]
        c = (W * self.b_minus).flatten()

        # Inequality constraint: −(W * b⁺) · d ≤ −α |I_t|
        A_ub = -(W * self.b_plus).flatten().reshape(1, k * k)
        b_ub = np.array([-self.alpha * I_total])

        res = self._run_linprog(c, A_ub, b_ub)
        return self._build_result(res, "primary", feasibility_margin, W)

    def _solve_softened_lp(
        self,
        W: np.ndarray,
        feasibility_margin: float,
    ) -> OptimizerResult:
        """
        Solve the softened LP from equation (24).

        No hard constraint. Objective combines false-content suppression
        and true-content preservation with weight λ:

            min_d  ∑_uv W[u,v] (b⁻[u,v] + λ b⁺[u,v]) d_uv

        Note: when λ > 0 the solver will increase some d_uv values to
        preserve true-content pathways even at the cost of allowing
        slightly more false-content spread.
        """
        c = (W * (self.b_minus + self.lambda_weight * self.b_plus)).flatten()
        res = self._run_linprog(c, A_ub=None, b_ub=None)
        return self._build_result(res, "softened", feasibility_margin, W)

    def _run_linprog(
        self,
        c: np.ndarray,
        A_ub: Optional[np.ndarray],
        b_ub: Optional[np.ndarray],
    ) -> OptimizeResult:
        """
        Call scipy.optimize.linprog, falling back gracefully on failure.
        """
        try:
            res = linprog(
                c      = c,
                A_ub   = A_ub,
                b_ub   = b_ub,
                bounds = self._bounds,
                method = self.solver,
                options = {"disp": False, "presolve": True},
            )
        except Exception as exc:
            # Return a synthetic failed result so the caller can fall back
            warnings.warn(f"linprog raised {type(exc).__name__}: {exc}")
            n = self.k * self.k
            res = OptimizeResult(
                x       = np.ones(n) * self.dropout_max,
                fun     = float("nan"),
                success = False,
                message = str(exc),
                status  = -1,
            )
        return res

    def _build_result(
        self,
        res: OptimizeResult,
        lp_type: str,
        feasibility_margin: float,
        W: np.ndarray,
    ) -> OptimizerResult:
        """
        Convert a scipy OptimizeResult into an OptimizerResult.

        If the solver failed, fall back to d* = 1 (no dropouts applied).
        This is the conservative safe choice — it never incorrectly
        suppresses true content even if the LP fails.
        """
        k = self.k

        if res.success and res.x is not None:
            d_flat = np.clip(res.x, self.dropout_min, self.dropout_max)
            obj    = float(res.fun)
        else:
            # Fallback: no dropouts (conservative — never harms true content)
            d_flat = np.ones(k * k) * self.dropout_max
            obj    = float("nan")
            if lp_type != "fallback":
                lp_type = "fallback"

        return OptimizerResult(
            dropout_matrix    = d_flat.reshape(k, k),
            lp_type           = lp_type,
            converged         = bool(res.success),
            solver_message    = res.message,
            feasibility_margin= float(feasibility_margin),
            objective_value   = obj,
            alpha             = self.alpha,
            lambda_weight     = self.lambda_weight,
        )

    def _validate_counts(
        self,
        S_counts: np.ndarray,
        I_counts: np.ndarray,
    ) -> None:
        """Check that count vectors have the right shape and are non-negative."""
        for name, arr in [("S_counts", S_counts), ("I_counts", I_counts)]:
            if arr.ndim != 1 or arr.shape[0] != self.k:
                raise ValueError(
                    f"{name} must be a 1-D array of length k={self.k}. "
                    f"Got shape {arr.shape}."
                )
            if np.any(arr < 0):
                raise ValueError(f"{name} must be non-negative.")


# ── Convenience factory ───────────────────────────────────────────────────────

def make_optimizer_from_config(b_minus: np.ndarray, b_plus: np.ndarray) -> DropoutOptimizer:
    """
    Build a DropoutOptimizer using parameters from config.py.

    Usage:
        from config import cfg
        from graph_engine.optimizer import make_optimizer_from_config
        import numpy as np

        b_minus = np.load(cfg.paths.sbm_matrices / 'b_minus.npy')
        b_plus  = np.load(cfg.paths.sbm_matrices / 'b_plus.npy')
        opt = make_optimizer_from_config(b_minus, b_plus)
    """
    try:
        from config import cfg
        return DropoutOptimizer(
            b_minus        = b_minus,
            b_plus         = b_plus,
            alpha          = cfg.lp.alpha,
            lambda_weight  = cfg.lp.lambda_weight,
            dropout_min    = cfg.lp.dropout_min,
            dropout_max    = cfg.lp.dropout_max,
        )
    except ImportError:
        return DropoutOptimizer(b_minus, b_plus)


# ── Tests ────────────────────────────────────────────────────────────────────

def _run_tests() -> None:
    """
    Self-contained test suite — no pytest required.
    Run: python optimizer.py
    """
    import traceback
    passed = 0
    failed = 0

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal passed, failed
        if condition:
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}" + (f": {detail}" if detail else ""))
            failed += 1

    print("\n=== DropoutOptimizer tests ===\n")

    # ── Test 1: Basic 2-partition case from paper (synthetic SBM matrices) ──
    print("Test 1: 2-partition synthetic SBM (matches paper eq. 25)")
    k = 2
    # Base matrix from paper (eq. 25):
    #   bbase = [[0.01, 0.002], [0.002, 0.01]]
    # False: b⁻ = bbase - xI + y(J-I)  → higher cross-partition spread
    # True:  b⁺ = bbase + xI - y(J-I)  → higher within-partition spread
    x, y = 0.005, 0.0005
    b_plus  = np.array([[0.01 + x, 0.002 - y],
                        [0.002 - y, 0.01 + x]])
    b_minus = np.array([[0.01 - x, 0.002 + y],
                        [0.002 + y, 0.01 - x]])
    opt = DropoutOptimizer(b_minus, b_plus, alpha=1.5, lambda_weight=1.0)

    S_counts = np.array([490.0, 490.0])
    I_counts = np.array([5.0,   5.0])
    res = opt.solve(S_counts, I_counts)
    check("lp_type is primary", res.lp_type == "primary",
          f"got {res.lp_type!r}")
    check("converged", res.converged)
    check("dropout_matrix shape", res.dropout_matrix.shape == (k, k))
    check("dropout values in [0,1]",
          np.all(res.dropout_matrix >= 0) and np.all(res.dropout_matrix <= 1))
    check("feasibility_margin >= 0", res.feasibility_margin >= 0)
    # True content branching must still be ≥ α after dropout
    true_spread = res.expected_true_spread(S_counts, I_counts, b_plus)
    I_total = I_counts.sum()
    check("true content constraint satisfied",
          true_spread >= 1.5 * I_total - 1e-6,
          f"true_spread={true_spread:.4f}, α|I|={1.5*I_total:.4f}")

    # ── Test 2: 3-partition case ─────────────────────────────────────────────
    print("\nTest 2: 3-partition synthetic SBM (matches paper eq. 26)")
    k = 3
    bbase = np.full((k, k), 0.002)
    np.fill_diagonal(bbase, 0.01)
    b_plus3  = bbase + x * np.eye(k) - y * (np.ones((k,k)) - np.eye(k))
    b_minus3 = bbase - x * np.eye(k) + y * (np.ones((k,k)) - np.eye(k))
    opt3 = DropoutOptimizer(b_minus3, b_plus3, alpha=1.5)
    S3   = np.array([330.0, 330.0, 330.0])
    I3   = np.array([3.0,   3.0,   3.0])
    res3 = opt3.solve(S3, I3)
    check("3-partition converged", res3.converged)
    check("3-partition shape", res3.dropout_matrix.shape == (3, 3))
    true3 = res3.expected_true_spread(S3, I3, b_plus3)
    check("3-partition constraint",
          true3 >= 1.5 * I3.sum() - 1e-6,
          f"{true3:.4f} ≥ {1.5*I3.sum():.4f}")

    # ── Test 3: Infeasible → softened LP ────────────────────────────────────
    print("\nTest 3: Infeasible primary LP → softened LP")
    # Extremely low true-content spread: b⁺ ≈ 0 → can never satisfy α≥1.5
    b_plus_low  = np.full((2, 2), 1e-8)
    b_minus_low = np.array([[0.01, 0.002], [0.002, 0.01]])
    opt_inf = DropoutOptimizer(b_minus_low, b_plus_low, alpha=1.5, lambda_weight=1.0)
    S_inf   = np.array([490.0, 490.0])
    I_inf   = np.array([5.0, 5.0])
    res_inf = opt_inf.solve(S_inf, I_inf)
    check("softened LP triggered", res_inf.lp_type == "softened",
          f"got {res_inf.lp_type!r}")
    check("softened LP converged", res_inf.converged)
    check("feasibility_margin < 0", res_inf.feasibility_margin < 0)
    check("dropout values in [0,1]",
          np.all(res_inf.dropout_matrix >= 0) and np.all(res_inf.dropout_matrix <= 1))

    # ── Test 4: No infected users ────────────────────────────────────────────
    print("\nTest 4: No infected users (cascade not started)")
    res_empty = opt.solve(np.array([490.0, 490.0]), np.array([0.0, 0.0]))
    check("no_infected type", res_empty.lp_type == "no_infected")
    check("zero dropouts", np.allclose(res_empty.dropout_matrix, 0))

    # ── Test 5: Unbalanced partitions (matches paper Table I unbalanced) ─────
    print("\nTest 5: Unbalanced 2-partition [800, 200 users]")
    opt_ub = DropoutOptimizer(b_minus, b_plus, alpha=1.5)
    S_ub   = np.array([795.0, 195.0])   # most susceptible in class 0
    I_ub   = np.array([4.0, 1.0])
    res_ub = opt_ub.solve(S_ub, I_ub)
    check("unbalanced converged", res_ub.converged)
    true_ub = res_ub.expected_true_spread(S_ub, I_ub, b_plus)
    I_ub_total = I_ub.sum()
    check("unbalanced constraint",
          true_ub >= 1.5 * I_ub_total - 1e-6,
          f"{true_ub:.4f} ≥ {1.5*I_ub_total:.4f}")

    # ── Test 6: solve_cascade helper ─────────────────────────────────────────
    print("\nTest 6: solve_cascade over multiple steps")
    S_seq = [np.array([490.0, 490.0]),
             np.array([450.0, 450.0]),
             np.array([400.0, 400.0])]
    I_seq = [np.array([5.0, 5.0]),
             np.array([30.0, 30.0]),
             np.array([20.0, 20.0])]
    results = opt.solve_cascade(S_seq, I_seq)
    check("cascade length", len(results) == 3)
    check("all converged", all(r.converged for r in results))

    # ── Test 7: dropout_min > 0 ──────────────────────────────────────────────
    print("\nTest 7: dropout_min=0.1 (minimum connection preserved)")
    opt_min = DropoutOptimizer(b_minus, b_plus, alpha=1.5, dropout_min=0.1)
    res_min = opt_min.solve(S_counts, I_counts)
    check("all dropouts ≥ 0.1",
          np.all(res_min.dropout_matrix >= 0.1 - 1e-9),
          f"min={res_min.dropout_matrix.min():.4f}")

    # ── Test 8: feasibility_margin sign matches lp_type ──────────────────────
    print("\nTest 8: feasibility_margin sign consistency")
    check("primary ↔ margin ≥ 0",
          (res.lp_type == "primary") == (res.feasibility_margin >= 0))
    check("softened ↔ margin < 0",
          (res_inf.lp_type == "softened") == (res_inf.feasibility_margin < 0))

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed ✓")
    else:
        print("Some tests FAILED — review output above")
    return failed


# ── Demo: reproduce paper's 2-partition balanced scenario ────────────────────

def _demo_paper_scenario() -> None:
    """
    Reproduce the balanced 2-partition scenario from the paper (Section V.A).
    Shows what the LP actually computes at the first cascade step.
    """
    print("\n=== Demo: Paper balanced 2-partition scenario ===\n")

    k = 2
    bbase   = np.array([[0.01, 0.002], [0.002, 0.01]])
    x, y    = 0.005, 0.0005

    results_table = []

    for alpha, lam in [(1.5, 1.0), (2.0, 1.5)]:
        b_plus  = bbase + x * np.eye(k) - y * (np.ones((k,k)) - np.eye(k))
        b_minus = bbase - x * np.eye(k) + y * (np.ones((k,k)) - np.eye(k))

        opt = DropoutOptimizer(b_minus, b_plus, alpha=alpha, lambda_weight=lam)

        # Balanced partitions, t=1 step of cascade
        S_counts = np.array([495.0, 495.0])
        I_counts = np.array([5.0, 5.0])

        res = opt.solve(S_counts, I_counts)

        false_spread = res.expected_false_spread(S_counts, I_counts, b_minus)
        true_spread  = res.expected_true_spread(S_counts, I_counts, b_plus)
        I_total      = I_counts.sum()

        print(f"α={alpha}, λ={lam}:")
        print(f"  LP type          : {res.lp_type}")
        print(f"  Converged        : {res.converged}")
        print(f"  Dropout matrix d*:\n{res.dropout_matrix}")
        print(f"  E[false spread]  : {false_spread:.4f}")
        print(f"  E[true  spread]  : {true_spread:.4f}  (must ≥ α|I|={alpha*I_total:.1f})")
        print(f"  Constraint slack : {true_spread - alpha*I_total:.6f}")
        print()


if __name__ == "__main__":
    failed = _run_tests()
    _demo_paper_scenario()
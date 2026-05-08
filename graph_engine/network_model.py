"""
InfoGuard — Graph Engine: Network Model (SBM)
=============================================
Builds and fits the two Stochastic Block Models that drive the LP optimizer:

    G⁺ = GSBM(C, [b⁺_uv])   ← true content propagation
    G⁻ = GSBM(C, [b⁻_uv])   ← false content propagation

where C = {C₁, C₂, ..., Cₖ} is the polarization partition of users and
b_uv is the probability of content transfer from a user in class u to a
user in class v.

─── Fitting procedure (paper Section III.B) ────────────────────────────────

1.  Build the union graph from all WICO cascades.
2.  Run modularity-based clustering (Louvain, resolution=cfg.sbm.clustering_resolution)
    to assign each user to a polarization class.
3.  Merge tiny partitions (< cfg.sbm.min_partition_fraction of total users)
    into the nearest class by edge density.
4.  For each WICO cascade labeled TRUE  : count transfers between class pairs → b⁺
    For each WICO cascade labeled FALSE : count transfers between class pairs → b⁻
    Normalise by the number of sharing users in the source class (frequentist MLE).

─── Paper equation reference ────────────────────────────────────────────────

    b⁺_uv = (1/|Cu|) * ∑_{i ∈ Cu} r⁺_i * c⁺_uv        eq. (6a)
    b⁻_uv = (1/|Cu|) * ∑_{i ∈ Cu} r⁻_i * c⁻_uv        eq. (6b)

In the frequentist estimate this simplifies to:
    b̂_uv = (observed transfers from Cu to Cv) / (observed sharing opportunities from Cu)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
from networkx.algorithms.community import louvain_communities


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SBM:
    """
    A fitted pair of Stochastic Block Models (G⁺ and G⁻).

    Attributes
    ----------
    b_plus : np.ndarray, shape (k, k)
        Edge probability matrix for TRUE content.
        b_plus[u, v] = P(user in class u shares true content to user in class v).

    b_minus : np.ndarray, shape (k, k)
        Edge probability matrix for FALSE content.
        b_minus[u, v] = P(user in class u shares false content to user in class v).

    k : int
        Number of polarization classes.

    partition : dict[node_id, int]
        Maps each user node ID to its class index (0 … k-1).

    class_sizes : np.ndarray, shape (k,)
        Number of users in each class.
    """
    b_plus:       np.ndarray
    b_minus:      np.ndarray
    k:            int
    partition:    dict
    class_sizes:  np.ndarray

    # ── Validation ─────────────────────────────────────────────────────────
    def __post_init__(self) -> None:
        for name, mat in [("b_plus", self.b_plus), ("b_minus", self.b_minus)]:
            if mat.shape != (self.k, self.k):
                raise ValueError(
                    f"{name} must be ({self.k},{self.k}). Got {mat.shape}."
                )
            if np.any(mat < 0) or np.any(mat > 1):
                raise ValueError(f"{name} values must be in [0, 1].")

    # ── Derived properties ──────────────────────────────────────────────────
    @property
    def b_plus_minus_diff(self) -> np.ndarray:
        """
        Element-wise difference b⁺ - b⁻.
        Positive entries: true content travels this path more than false.
        Negative entries: false content travels this path more than true.
        The LP exploits these differences.
        """
        return self.b_plus - self.b_minus

    def expected_spread(
        self,
        I_counts: np.ndarray,
        S_counts: np.ndarray,
        content: str = "false",
        dropout: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute E[|I^v_{t+1}|] for each class v using eq. (15)/(16).

        Uses the large-N asymptotic approximation from the paper:
            E[|I^v_{t+1}|] = |S^v_t| * (1 - exp(- ∑_u |I^u_t| * d_uv * b_uv))

        Parameters
        ----------
        I_counts : (k,) infected users per class
        S_counts : (k,) susceptible users per class
        content  : "true" or "false"
        dropout  : (k, k) dropout matrix d; defaults to all-ones (no dropout)

        Returns
        -------
        (k,) expected new infections per class
        """
        b = self.b_plus if content == "true" else self.b_minus
        d = dropout if dropout is not None else np.ones((self.k, self.k))
        # effective_rate[v] = ∑_u I^u * d_uv * b_uv  (sum over sharing classes u)
        effective_rate = (I_counts[:, np.newaxis] * d * b).sum(axis=0)   # (k,)
        return S_counts * (1.0 - np.exp(-effective_rate))

    def total_expected_spread(
        self,
        I_counts: np.ndarray,
        S_counts: np.ndarray,
        content: str = "false",
        dropout: Optional[np.ndarray] = None,
    ) -> float:
        """Scalar total E[|I_{t+1}|] across all classes."""
        return float(self.expected_spread(I_counts, S_counts, content, dropout).sum())

    # ── Persistence ────────────────────────────────────────────────────────
    def save(self, directory: Path) -> None:
        """
        Save SBM matrices and partition to cfg.paths.sbm_matrices.
        Creates the directory if it does not exist.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "b_plus.npy",      self.b_plus)
        np.save(directory / "b_minus.npy",     self.b_minus)
        np.save(directory / "class_sizes.npy", self.class_sizes)
        # Partition dict: keys may be strings or ints — save as two arrays
        keys   = np.array(list(self.partition.keys()),   dtype=object)
        values = np.array(list(self.partition.values()), dtype=np.int32)
        np.save(directory / "partition_keys.npy",   keys,   allow_pickle=True)
        np.save(directory / "partition_values.npy", values)
        np.save(directory / "k.npy", np.array([self.k]))
        print(f"✓ SBM saved to {directory}  (k={self.k})")

    @classmethod
    def load(cls, directory: Path) -> "SBM":
        """Load a previously saved SBM from cfg.paths.sbm_matrices."""
        directory = Path(directory)
        b_plus      = np.load(directory / "b_plus.npy")
        b_minus     = np.load(directory / "b_minus.npy")
        class_sizes = np.load(directory / "class_sizes.npy")
        keys        = np.load(directory / "partition_keys.npy",   allow_pickle=True)
        values      = np.load(directory / "partition_values.npy")
        k           = int(np.load(directory / "k.npy")[0])
        partition   = dict(zip(keys.tolist(), values.tolist()))
        return cls(b_plus=b_plus, b_minus=b_minus, k=k,
                   partition=partition, class_sizes=class_sizes)

    def __repr__(self) -> str:
        sizes = ", ".join(str(s) for s in self.class_sizes)
        return (f"SBM(k={self.k}, class_sizes=[{sizes}], "
                f"b_plus_range=[{self.b_plus.min():.4f}, {self.b_plus.max():.4f}], "
                f"b_minus_range=[{self.b_minus.min():.4f}, {self.b_minus.max():.4f}])")


# ── SBM Fitter ────────────────────────────────────────────────────────────────

class SBMFitter:
    """
    Fits the b⁺ and b⁻ SBM matrices from labeled WICO cascade graphs.

    Usage
    -----
    fitter = SBMFitter()                        # reads k, resolution from cfg
    fitter.add_cascade(G, label="false")        # add one WICO cascade
    fitter.add_cascade(G, label="true")
    ...
    sbm = fitter.fit()                          # returns fitted SBM
    sbm.save(cfg.paths.sbm_matrices)
    """

    def __init__(
        self,
        num_partitions:          Optional[int]   = None,
        clustering_resolution:   Optional[float] = None,
        min_partition_fraction:  Optional[float] = None,
        label_confidence_threshold: Optional[float] = None,
        seed: Optional[int]      = None,
    ) -> None:
        """
        Parameters default to cfg values when not provided.
        This lets tests pass explicit values without needing config.
        """
        try:
            from config import cfg
            self._k_target    = num_partitions         or cfg.sbm.num_partitions
            self._resolution  = clustering_resolution  or cfg.sbm.clustering_resolution
            self._min_frac    = min_partition_fraction or cfg.sbm.min_partition_fraction
            self._conf_thresh = label_confidence_threshold or cfg.sbm.label_confidence_threshold
            self._seed        = seed if seed is not None else cfg.seed
        except ImportError:
            self._k_target    = num_partitions         or 13
            self._resolution  = clustering_resolution  or 2.0
            self._min_frac    = min_partition_fraction or 0.01
            self._conf_thresh = label_confidence_threshold or 0.65
            self._seed        = seed if seed is not None else 42

        # Accumulated cascade data
        self._true_graphs:  list[nx.DiGraph] = []
        self._false_graphs: list[nx.DiGraph] = []

        # Set after fit() is called
        self._partition:    Optional[dict]       = None
        self._class_sizes:  Optional[np.ndarray] = None
        self._k:            Optional[int]        = None

    # ── Public API ──────────────────────────────────────────────────────────

    def add_cascade(
        self,
        G: nx.DiGraph,
        label: str,
        confidence: float = 1.0,
    ) -> None:
        """
        Register one WICO cascade graph for fitting.

        Parameters
        ----------
        G          : DiGraph where nodes are user IDs and edges represent
                     content transfers (source → target).
        label      : "true" or "false" (binary label from cfg.wico.binary_label_map).
        confidence : BiGCN confidence score. Cascades below
                     cfg.sbm.label_confidence_threshold are silently skipped.
        """
        if confidence < self._conf_thresh:
            return
        if label == "true":
            self._true_graphs.append(G)
        elif label == "false":
            self._false_graphs.append(G)
        else:
            warnings.warn(f"Unknown label {label!r} — skipped.")

    def add_cascades_from_df(self, df, graphs: dict) -> None:
        """
        Bulk-add cascades from a DataFrame (from propagation_tree_summary.csv).

        Parameters
        ----------
        df     : DataFrame with columns 'content_id', 'label', optionally
                 'branching_ratio' used as a confidence proxy.
        graphs : dict mapping content_id → nx.DiGraph
        """
        for _, row in df.iterrows():
            cid   = str(row.get("content_id", row.get("tweet_id", "")))
            label = str(row.get("label", "")).lower().strip()
            conf  = float(row.get("branching_ratio", 1.0)) if "branching_ratio" in row else 1.0
            if cid in graphs and label in ("true", "false"):
                self.add_cascade(graphs[cid], label, confidence=1.0)

    def fit(self) -> SBM:
        """
        Run the full fitting procedure and return a fitted SBM.

        Steps:
        1. Build union graph from all added cascades.
        2. Run Louvain clustering to get polarization classes.
        3. Merge tiny classes.
        4. Count transfers per class pair → estimate b⁺ and b⁻.
        """
        all_graphs = self._true_graphs + self._false_graphs
        if not all_graphs:
            raise RuntimeError(
                "No cascades have been added. Call add_cascade() first."
            )

        print(f"Fitting SBM from {len(self._true_graphs)} true and "
              f"{len(self._false_graphs)} false cascades.")

        # Step 1: build union graph
        union_G = self._build_union_graph(all_graphs)
        print(f"  Union graph: {union_G.number_of_nodes()} nodes, "
              f"{union_G.number_of_edges()} edges")

        # Step 2: Louvain clustering
        partition = self._cluster(union_G)

        # Step 3: merge tiny classes
        partition, k = self._merge_small_classes(partition, union_G)
        self._partition   = partition
        self._k           = k
        self._class_sizes = self._compute_class_sizes(partition, k)
        print(f"  Polarization classes: k={k}, "
              f"sizes={self._class_sizes.tolist()}")

        # Step 4: estimate b⁺ and b⁻
        b_plus  = self._estimate_b(self._true_graphs,  partition, k, label="true")
        b_minus = self._estimate_b(self._false_graphs, partition, k, label="false")

        return SBM(
            b_plus=b_plus, b_minus=b_minus,
            k=k, partition=partition,
            class_sizes=self._class_sizes,
        )

    # ── Private helpers ─────────────────────────────────────────────────────

    def _build_union_graph(self, graphs: list[nx.DiGraph]) -> nx.Graph:
        """
        Merge all cascade graphs into a single undirected union graph.
        Edge weights count how many times a connection appeared.
        """
        U = nx.Graph()
        for G in graphs:
            for u, v in G.edges():
                if U.has_edge(u, v):
                    U[u][v]["weight"] += 1
                else:
                    U.add_edge(u, v, weight=1)
        return U

    def _cluster(self, G: nx.Graph) -> dict:
        """
        Run Louvain clustering and return {node: class_index} dict.

        Uses cfg.sbm.clustering_resolution (paper uses 2.0 for WICO).
        Falls back to greedy modularity if Louvain fails.
        """
        try:
            communities = louvain_communities(
                G,
                resolution=self._resolution,
                seed=self._seed,
            )
        except Exception as exc:
            warnings.warn(f"Louvain failed ({exc}), falling back to greedy modularity.")
            from networkx.algorithms.community import greedy_modularity_communities
            communities = list(greedy_modularity_communities(G, resolution=self._resolution))

        partition = {}
        for class_idx, community in enumerate(communities):
            for node in community:
                partition[node] = class_idx
        return partition

    def _merge_small_classes(
        self,
        partition: dict,
        G: nx.Graph,
    ) -> tuple[dict, int]:
        """
        Merge classes that contain fewer than cfg.sbm.min_partition_fraction
        of total users into the nearest class (most shared edges).

        The paper merged all partitions with < 1% of users into their
        nearest neighbours. This keeps k manageable (paper got k=13 from
        63,914 raw Louvain partitions on WICO).
        """
        total_nodes = len(partition)
        min_size    = max(1, int(self._min_frac * total_nodes))

        # Count class sizes
        from collections import Counter
        size_map = Counter(partition.values())
        all_classes = set(size_map.keys())

        changed = True
        while changed:
            changed = False
            small = {c for c, s in size_map.items() if s < min_size}
            if not small:
                break

            for tiny_cls in sorted(small):
                # Find the class most connected to tiny_cls via union graph edges
                edge_counts: Counter = Counter()
                for node, cls in partition.items():
                    if cls != tiny_cls:
                        continue
                    for neighbor in G.neighbors(node):
                        nb_cls = partition.get(neighbor)
                        if nb_cls is not None and nb_cls != tiny_cls:
                            edge_counts[nb_cls] += G[node][neighbor].get("weight", 1)

                if edge_counts:
                    target_cls = edge_counts.most_common(1)[0][0]
                else:
                    # No edges to other classes — merge into class 0
                    target_cls = min(all_classes - {tiny_cls})

                for node in list(partition.keys()):
                    if partition[node] == tiny_cls:
                        partition[node] = target_cls

                size_map[target_cls] += size_map.pop(tiny_cls)
                all_classes.discard(tiny_cls)
                changed = True
                break   # restart after each merge

        # Re-index classes to 0 … k-1
        old_classes = sorted(all_classes)
        remap       = {old: new for new, old in enumerate(old_classes)}
        partition   = {node: remap[cls] for node, cls in partition.items()}
        k           = len(old_classes)
        return partition, k

    def _compute_class_sizes(self, partition: dict, k: int) -> np.ndarray:
        sizes = np.zeros(k, dtype=np.int64)
        for cls in partition.values():
            if 0 <= cls < k:
                sizes[cls] += 1
        return sizes

    def _estimate_b(
        self,
        graphs: list[nx.DiGraph],
        partition: dict,
        k: int,
        label: str,
    ) -> np.ndarray:
        """
        Frequentist MLE of the SBM transfer matrix from labeled cascades.

            b̂_uv = transfers(Cu → Cv) / sharing_opportunities(Cu)

        sharing_opportunities(Cu) = number of times a user in Cu shared
        content (i.e. had at least one out-edge) across all cascades.

        If a cell has zero observations a small Laplace smoothing (1e-8)
        is applied so b is never exactly 0 (avoids log(0) in the SIR model).
        """
        transfer_counts     = np.zeros((k, k), dtype=np.float64)  # numerator
        sharing_opportunity = np.zeros(k, dtype=np.float64)        # denominator

        for G in graphs:
            for src, dst in G.edges():
                u = partition.get(src)
                v = partition.get(dst)
                if u is None or v is None:
                    continue           # node not in partition (unseen user)
                if not (0 <= u < k and 0 <= v < k):
                    continue
                transfer_counts[u, v] += 1

            # Count sharing opportunities: nodes with at least one out-edge
            for node in G.nodes():
                if G.out_degree(node) > 0:
                    u = partition.get(node)
                    if u is not None and 0 <= u < k:
                        sharing_opportunity[u] += 1

        # Normalise: b̂_uv = transfers_uv / max(1, opportunity_u)
        b = np.zeros((k, k), dtype=np.float64)
        for u in range(k):
            n_sharers = max(1.0, sharing_opportunity[u])
            for v in range(k):
                # Divide by target class size: gives per-(u_user, v_user) pair probability
                # This is what b_uv means in the paper's SBM — the probability that
                # one infected user in Cu infects one specific susceptible user in Cv
                cv_size = max(1.0, float(self._class_sizes[v]))
                b[u, v] = transfer_counts[u, v] / (n_sharers * cv_size)

        # Smooth cells with fewer than min_count observations toward the row mean
        # This prevents b=1.0 from cells with 1 transfer / 1 opportunity (sparse pairs)
        min_count = 5
        for u in range(k):
            for v in range(k):
                if transfer_counts[u, v] < min_count and sharing_opportunity[u] > 0:
                    # Pull toward the row mean of observed cells
                    row_mean = float(transfer_counts[u, :].sum() /
                                     max(1.0, sharing_opportunity[u] * k))
                    weight = transfer_counts[u, v] / min_count
                    b[u, v] = weight * b[u, v] + (1 - weight) * max(row_mean, 1e-6)
        b = np.where(b < 1e-8, 1e-8, b)
        b = np.clip(b, 0.0, 1.0)

        if label == "true":
            print(f"  b⁺: range [{b.min():.2e}, {b.max():.2e}], "
                  f"mean={b.mean():.2e}")
        else:
            print(f"  b⁻: range [{b.min():.2e}, {b.max():.2e}], "
                  f"mean={b.mean():.2e}")
        return b


# ── Synthetic SBM factory (for testing / demo) ───────────────────────────────

def make_synthetic_sbm(
    k: int = 2,
    x: float = 0.005,
    y: float = 0.0005,
    n_users: int = 1000,
    balanced: bool = True,
    seed: int = 42,
) -> SBM:
    """
    Build a synthetic SBM matching the paper's Section V.A test configurations.

    Uses the paper's base matrix (eq. 25/26) with perturbation parameters x, y:
        b⁺ = bbase + x*I  - y*(J-I)   (true:  stronger within-class)
        b⁻ = bbase - x*I  + y*(J-I)   (false: stronger cross-class)

    Parameters
    ----------
    k        : number of polarization classes (2 or 3)
    x        : within-class spread difference (b⁺_uu - b⁻_uu = 2x)
    y        : cross-class spread difference  (b⁻_uv - b⁺_uv = 2y, u≠v)
    n_users  : total synthetic users
    balanced : if True, equal class sizes; if False, [80%, 20%] for k=2
    seed     : random seed for partition assignment
    """
    rng   = np.random.default_rng(seed)
    I_mat = np.eye(k)
    J_mat = np.ones((k, k))

    if k == 2:
        bbase = np.array([[0.010, 0.002], [0.002, 0.010]])
    elif k == 3:
        bbase = np.full((k, k), 0.002)
        np.fill_diagonal(bbase, 0.010)
    else:
        bbase = np.full((k, k), 0.001)
        np.fill_diagonal(bbase, 0.010)

    b_plus  = np.clip(bbase + x * I_mat - y * (J_mat - I_mat), 0, 1)
    b_minus = np.clip(bbase - x * I_mat + y * (J_mat - I_mat), 0, 1)

    # Build class sizes
    if balanced or k != 2:
        base_size   = n_users // k
        class_sizes = np.array([base_size] * k, dtype=np.int64)
        class_sizes[0] += n_users - class_sizes.sum()
    else:
        class_sizes = np.array([int(0.8 * n_users),
                                n_users - int(0.8 * n_users)], dtype=np.int64)

    # Build partition dict
    nodes     = [f"u{i}" for i in range(n_users)]
    partition = {}
    start = 0
    for cls, size in enumerate(class_sizes):
        for node in nodes[start:start + size]:
            partition[node] = cls
        start += size

    return SBM(
        b_plus=b_plus, b_minus=b_minus,
        k=k, partition=partition,
        class_sizes=class_sizes,
    )
"""
base_ingestor.py — Abstract base class for all InfoShield cascade ingestors.

Every ingestor (Instagram, Twitter, Telegram, etc.) must implement this
interface so the pipeline can call simulate_cascade_following() without
knowing the source platform.

WICO node attribute contract (must be present on every node):
    followers     : int   — follower count at ingestion time
    time          : float — seconds since cascade root posted (0.0 for root)
    friends       : int   — followee count (accounts this user follows)

Optional but recommended:
    account_age_days : int
    is_verified      : bool
    platform         : str   — e.g. "instagram", "twitter"
    username         : str   — human-readable handle (never used as node id)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data contract returned by validate_graph()
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Structured summary of graph compatibility checks."""

    ok: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # Graph statistics populated during validation
    n_nodes: int = 0
    n_edges: int = 0
    n_roots: int = 0          # nodes with in_degree == 0
    avg_degree: float = 0.0
    missing_followers: int = 0
    missing_time: int = 0
    missing_friends: int = 0

    def __str__(self) -> str:
        status = "PASS" if self.ok else "FAIL"
        lines = [
            f"ValidationReport [{status}]",
            f"  nodes={self.n_nodes}  edges={self.n_edges}  roots={self.n_roots}",
            f"  avg_degree={self.avg_degree:.2f}",
            f"  missing_followers={self.missing_followers}  "
            f"missing_time={self.missing_time}  missing_friends={self.missing_friends}",
        ]
        for e in self.errors:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings:
            lines.append(f"  WARN : {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseIngestor(ABC):
    """
    Abstract ingestor — convert platform-specific engagement data into a
    propagation DiGraph compatible with InfoShield's simulate_cascade_following.

    EDGE DIRECTION CONVENTION (matches WICO after G.reverse()):
        A → B  means A's engagement was *triggered by* B's share.
                     (B shared → A saw it → A engaged)

    This is the REVERSED direction vs. the raw "B retweeted A" encoding
    used in Twitter15/16 tree files.  The convention ensures that
    find_root_user() locates the original poster as in_degree=0.

    Subclasses must implement the three abstract methods below.
    """

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def build_cascade(self, content_id: str) -> nx.DiGraph:
        """
        Fetch engagement data for *content_id* and return a propagation
        DiGraph.

        The returned graph must satisfy:
          - Nodes carry at least {'followers', 'time', 'friends'} attrs.
          - Exactly one node has in_degree == 0 (the original poster /
            cascade root).
          - The graph is weakly connected (single cascade component).

        Parameters
        ----------
        content_id : str
            Platform-specific identifier for the post/content item.

        Returns
        -------
        nx.DiGraph
            Propagation graph ready for simulate_cascade_following().
        """

    @abstractmethod
    def get_root_node(self, G: nx.DiGraph) -> str:
        """
        Return the node ID of the original poster.

        Must be consistent with find_root_user() in pipeline/sbm_fitter.py,
        which selects the node with in_degree==0 and highest follower count.

        Parameters
        ----------
        G : nx.DiGraph
            Graph returned by build_cascade().

        Returns
        -------
        str
            Node ID (same string used as node key in G).
        """

    @abstractmethod
    def get_node_metadata(self, node_id: str) -> dict[str, Any]:
        """
        Return metadata for a single node.

        The dict must contain at minimum:
            {
                'followers': int,
                'time':      float,   # seconds since root posted
                'friends':   int,
            }

        Additional keys are allowed and will be stored on the node.

        Parameters
        ----------
        node_id : str
            Node identifier as used in the graph.

        Returns
        -------
        dict
            Metadata compatible with WICO nodes.csv structure.
        """

    # ------------------------------------------------------------------
    # Concrete helpers (may be overridden but rarely need to be)
    # ------------------------------------------------------------------

    def validate_graph(self, G: nx.DiGraph) -> ValidationReport:
        """
        Run compatibility checks against the InfoShield pipeline contract.

        Checks performed
        ----------------
        1. Graph is a DiGraph (not MultiDiGraph, not undirected).
        2. At least 2 nodes present.
        3. Exactly one root (in_degree == 0).
        4. Graph is weakly connected.
        5. All nodes carry 'followers', 'time', 'friends' attributes.
        6. Warn if avg_degree deviates significantly from WICO baseline (~2.82).
        7. Warn if any follower count is negative.

        Returns
        -------
        ValidationReport
        """
        report = ValidationReport()
        report.n_nodes = G.number_of_nodes()
        report.n_edges = G.number_of_edges()

        # --- structural type ---
        if not isinstance(G, nx.DiGraph):
            report.errors.append(
                f"Graph must be nx.DiGraph, got {type(G).__name__}"
            )
            report.ok = False
            return report  # cannot proceed with further checks

        # --- minimum size ---
        if report.n_nodes < 2:
            report.errors.append(
                f"Graph has {report.n_nodes} node(s); minimum is 2."
            )
            report.ok = False

        # --- root count ---
        roots = [n for n, d in G.in_degree() if d == 0]
        report.n_roots = len(roots)
        if report.n_roots == 0:
            report.errors.append(
                "No root node found (no node with in_degree == 0). "
                "Check edge direction convention."
            )
            report.ok = False
        elif report.n_roots > 1:
            report.errors.append(
                f"{report.n_roots} root nodes found: {roots[:5]}… "
                "Graph must be a single cascade with one original poster."
            )
            report.ok = False

        # --- connectivity ---
        if report.n_nodes >= 2 and not nx.is_weakly_connected(G):
            n_comp = nx.number_weakly_connected_components(G)
            report.errors.append(
                f"Graph has {n_comp} weakly connected components; must be 1."
            )
            report.ok = False

        # --- node attributes ---
        required_attrs = ("followers", "time", "friends")
        missing: dict[str, int] = {a: 0 for a in required_attrs}
        negative_followers = 0

        for node, data in G.nodes(data=True):
            for attr in required_attrs:
                if attr not in data:
                    missing[attr] += 1
            if data.get("followers", 0) < 0:
                negative_followers += 1

        report.missing_followers = missing["followers"]
        report.missing_time = missing["time"]
        report.missing_friends = missing["friends"]

        for attr in required_attrs:
            if missing[attr] > 0:
                report.errors.append(
                    f"{missing[attr]} node(s) missing attribute '{attr}'."
                )
                report.ok = False

        if negative_followers > 0:
            report.warnings.append(
                f"{negative_followers} node(s) have negative follower counts."
            )

        # --- degree statistics ---
        if report.n_nodes > 0:
            report.avg_degree = report.n_edges / report.n_nodes
            wico_avg = 2.82
            if report.avg_degree > wico_avg * 3:
                report.warnings.append(
                    f"avg_degree={report.avg_degree:.2f} is >3× WICO baseline "
                    f"({wico_avg}). Graph may be unusually dense."
                )

        return report

    def attach_metadata(
        self, G: nx.DiGraph, node_id: str, meta: dict[str, Any]
    ) -> None:
        """Convenience: write metadata dict onto an existing node."""
        if node_id not in G:
            raise KeyError(f"Node '{node_id}' not in graph.")
        G.nodes[node_id].update(meta)

    def ensure_required_attrs(
        self,
        G: nx.DiGraph,
        default_followers: int = 0,
        default_friends: int = 0,
    ) -> None:
        """
        Fill missing required attributes with safe defaults in-place.
        Call this at the end of build_cascade() if some nodes may lack attrs.
        """
        for node, data in G.nodes(data=True):
            if "followers" not in data:
                data["followers"] = default_followers
                logger.debug("Node %s: missing 'followers', set to %d", node, default_followers)
            if "friends" not in data:
                data["friends"] = default_friends
            if "time" not in data:
                # Estimate from topological depth (BFS layers × 30 min)
                data["time"] = 0.0

    def summary(self, G: nx.DiGraph) -> str:
        """One-line human-readable summary of a cascade graph."""
        roots = [n for n, d in G.in_degree() if d == 0]
        root_id = roots[0] if roots else "?"
        root_followers = G.nodes[root_id].get("followers", "?") if roots else "?"
        return (
            f"Cascade: {G.number_of_nodes()} nodes, "
            f"{G.number_of_edges()} edges, "
            f"root={root_id} (followers={root_followers})"
        )

"""
instagram_ingestor.py — Instagram cascade ingestor for InfoShield.

Converts Instagram public post engagement data into propagation DiGraphs
compatible with simulate_cascade_following() and the SBM pipeline.

──────────────────────────────────────────────────────────────────────────────
EDGE DIRECTION (matches WICO convention after G.reverse()):

    A → B  means A's engagement was triggered by B's share.
            B shared the content → A saw it through B → A engaged.

Root node (original poster) has in_degree == 0.
This is consistent with find_root_user() in pipeline/sbm_fitter.py.

──────────────────────────────────────────────────────────────────────────────
API USAGE:

    Real mode (Instagram Basic Display API + Graph API):
        ingestor = InstagramIngestor(access_token="IGQV...")
        G = ingestor.get_post_cascade("17858893269000001")

    Mock mode (no token required — for CI / offline testing):
        ingestor = InstagramIngestor(mock=True, seed=42)
        G = ingestor.get_post_cascade("any_id")

──────────────────────────────────────────────────────────────────────────────
MOCK STATISTICAL TARGETS (derived from WICO dataset analysis):

    node count    : Uniform(20, 100)
    followers     : LogNormal(μ=6.215, σ=2.0)  → geometric mean ≈ 500
    avg degree    : ~2.82  (WICO baseline)
    edge density  : tree-like backbone + ~15% cross-edges
    timing        : Exponential inter-arrival, root at t=0
──────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import logging
import math
import time
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlencode

import networkx as nx
import numpy as np

from pipeline.ingestors.base_ingestor import BaseIngestor, ValidationReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Instagram Graph API constants
# ---------------------------------------------------------------------------

_GRAPH_API_BASE = "https://graph.instagram.com/v18.0"
_BASIC_API_BASE = "https://api.instagram.com/v1"

# Fields requested for each user/node
_USER_FIELDS = "id,username,followers_count,media_count,account_type,biography"

# Comment fields — used for fallback cascade construction
_COMMENT_FIELDS = "id,username,text,timestamp,replies{id,username,text,timestamp}"

# Rate-limit: Instagram Graph API allows ~200 calls/hour for basic tokens.
# We back off 0.5 s between calls to stay well within limits.
_API_CALL_DELAY_S = 0.5


# ---------------------------------------------------------------------------
# Tiny HTTP helper (avoids mandatory requests/httpx dependency)
# ---------------------------------------------------------------------------

def _http_get(url: str, params: dict | None = None, timeout: int = 15) -> dict:
    """
    Perform an HTTP GET and return the parsed JSON body.

    Uses urllib (stdlib) so no extra dependency is needed.
    Raises RuntimeError on non-200 status or JSON parse failure.
    """
    import json
    import urllib.request

    full_url = url if not params else f"{url}?{urlencode(params)}"
    logger.debug("GET %s", full_url)

    try:
        with urllib.request.urlopen(full_url, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
    except Exception as exc:
        raise RuntimeError(f"HTTP request failed: {exc}") from exc

    try:
        data = json.loads(body)
    except Exception as exc:
        raise RuntimeError(f"Failed to parse JSON response: {exc}") from exc

    if "error" in data:
        err = data["error"]
        raise RuntimeError(
            f"Instagram API error {err.get('code')}: {err.get('message')}"
        )

    return data


# ---------------------------------------------------------------------------
# InstagramIngestor
# ---------------------------------------------------------------------------

class InstagramIngestor(BaseIngestor):
    """
    Builds InfoShield-compatible propagation graphs from Instagram public posts.

    Parameters
    ----------
    access_token : str, optional
        Instagram Graph API / Basic Display API user access token.
        Required when mock=False.
    mock : bool
        If True, all API calls are bypassed and synthetic graphs are generated
        matching WICO statistical properties.  Ideal for CI and offline dev.
    seed : int
        RNG seed for reproducible mock graphs.
    timeout : int
        HTTP timeout in seconds for real API calls.
    """

    # WICO-derived mock parameters
    _MOCK_MIN_NODES = 20
    _MOCK_MAX_NODES = 100
    _MOCK_FOLLOWER_MU = math.log(500)   # lognormal in log-space: e^6.215 ≈ 500
    _MOCK_FOLLOWER_SIGMA = 2.0
    _MOCK_AVG_DEGREE = 2.82
    _MOCK_CROSS_EDGE_PROB = 0.15        # fraction of non-tree edges
    _MOCK_INTER_ARRIVAL_MEAN_S = 1800   # 30 min avg between engagements

    def __init__(
        self,
        access_token: str = "",
        mock: bool = False,
        seed: int = 42,
        timeout: int = 15,
    ) -> None:
        if not mock and not access_token:
            raise ValueError(
                "Provide access_token= or set mock=True for offline testing."
            )
        self.access_token = access_token
        self.mock = mock
        self.seed = seed
        self.timeout = timeout
        self._rng = np.random.default_rng(seed)

        # In-memory cache: node_id → metadata dict (avoids redundant API calls)
        self._meta_cache: dict[str, dict[str, Any]] = {}

    # -----------------------------------------------------------------------
    # BaseIngestor abstract implementations
    # -----------------------------------------------------------------------

    def build_cascade(self, content_id: str) -> nx.DiGraph:
        """
        Top-level entry point required by BaseIngestor.

        Delegates to get_post_cascade() for a natural API.
        """
        return self.get_post_cascade(content_id)

    def get_root_node(self, G: nx.DiGraph) -> str:
        """
        Return the original poster's node ID.

        Matches find_root_user() logic in pipeline/sbm_fitter.py:
        prefer the in_degree==0 node with the highest follower count.
        """
        roots = [n for n, d in G.in_degree() if d == 0]
        if not roots:
            raise ValueError("Graph has no root (no node with in_degree == 0).")
        if len(roots) == 1:
            return roots[0]
        # Tiebreak by followers
        return max(roots, key=lambda n: G.nodes[n].get("followers", 0))

    def get_node_metadata(self, node_id: str) -> dict[str, Any]:
        """
        Return WICO-compatible metadata for a node.

        Real mode  : fetches from Graph API (with caching).
        Mock mode  : returns synthetic data consistent with mock graph.

        Required keys returned: followers, time, friends.
        """
        if node_id in self._meta_cache:
            return self._meta_cache[node_id]

        if self.mock:
            meta = self._mock_node_meta(node_id)
        else:
            meta = self._fetch_user_meta(node_id)

        self._meta_cache[node_id] = meta
        return meta

    # -----------------------------------------------------------------------
    # Primary public methods
    # -----------------------------------------------------------------------

    def get_post_cascade(
        self,
        post_id: str,
        access_token: str | None = None,
    ) -> nx.DiGraph:
        """
        Build a propagation cascade for a single Instagram post.

        Real mode
        ---------
        Attempts Graph API reshare endpoints first.  Falls back to
        build_from_comments() if reshare data is unavailable (which is
        typical for non-Creator/Business accounts).

        Mock mode
        ---------
        Ignores post_id and generates a synthetic cascade that matches
        WICO graph statistics.

        Parameters
        ----------
        post_id : str
            Instagram media ID (numeric string, e.g. "17858893269000001").
        access_token : str, optional
            Override the instance-level token for this call only.

        Returns
        -------
        nx.DiGraph
            Propagation graph.  Validates before returning.
        """
        if self.mock:
            return self._generate_mock_cascade(post_id)

        token = access_token or self.access_token

        # Attempt 1: reshare chain (requires advanced permissions)
        try:
            G = self._build_from_reshares(post_id, token)
            logger.info("Built cascade from reshares for post %s", post_id)
        except RuntimeError as exc:
            logger.warning(
                "Reshare data unavailable (%s). Falling back to comments.", exc
            )
            G = self.build_from_comments(post_id, token)

        self._attach_platform_tag(G, "instagram")
        report = self.validate_graph(G)
        if not report.ok:
            logger.warning("Post %s cascade failed validation:\n%s", post_id, report)
        else:
            logger.info("%s", self.summary(G))

        return G

    def search_hashtag_cascade(
        self,
        hashtag: str,
        access_token: str | None = None,
        hours_back: int = 24,
    ) -> list[nx.DiGraph]:
        """
        Return one cascade per top post under a hashtag.

        Searches the most recent posts (up to hours_back hours) and builds
        a cascade for each.  Mock mode returns 3 synthetic cascades.

        Parameters
        ----------
        hashtag : str
            Hashtag without the '#' prefix (e.g. "misinformation").
        access_token : str, optional
            Override instance token.
        hours_back : int
            How far back to search (real mode only; mock always uses 24 h).

        Returns
        -------
        list[nx.DiGraph]
            One graph per post found.  Empty list if no posts are found.
        """
        if self.mock:
            n_posts = int(self._rng.integers(2, 6))
            logger.info(
                "Mock: generating %d cascades for #%s", n_posts, hashtag
            )
            return [
                self._generate_mock_cascade(f"mock_{hashtag}_{i}")
                for i in range(n_posts)
            ]

        token = access_token or self.access_token
        post_ids = self._search_hashtag_posts(hashtag, token, hours_back)

        if not post_ids:
            logger.info("No posts found for #%s in last %d h", hashtag, hours_back)
            return []

        cascades: list[nx.DiGraph] = []
        for pid in post_ids:
            try:
                G = self.get_post_cascade(pid, token)
                cascades.append(G)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Skipping post %s: %s", pid, exc)
            time.sleep(_API_CALL_DELAY_S)

        return cascades

    def build_from_comments(
        self,
        post_id: str,
        access_token: str | None = None,
    ) -> nx.DiGraph:
        """
        Approximate cascade from the comment reply thread.

        When reshare data is unavailable (most public accounts), the
        comment tree provides a usable proxy for the engagement cascade:
          - Root post → top-level comments  (B engaged after seeing root)
          - Top-level comment → replies     (C engaged after seeing B)

        Edges follow the same convention as get_post_cascade():
            A → B  means A engaged because of B's share.

        Parameters
        ----------
        post_id : str
            Instagram media ID.
        access_token : str, optional
            Override instance token.

        Returns
        -------
        nx.DiGraph
        """
        if self.mock:
            return self._generate_mock_cascade(post_id)

        token = access_token or self.access_token

        # Fetch post owner first (the root)
        post_data = self._api_get(f"/{post_id}", token, fields="id,username,owner")
        root_id = str(post_data.get("owner", {}).get("id", post_id + "_owner"))

        G = nx.DiGraph()
        root_meta = self._fetch_user_meta(root_id)
        root_meta["time"] = 0.0
        G.add_node(root_id, **root_meta)

        # Fetch comments
        comments = self._paginate(f"/{post_id}/comments", token, _COMMENT_FIELDS)

        for comment in comments:
            c_id = str(comment["id"])
            c_user = comment.get("username", c_id)
            c_time = self._iso_to_seconds_since(
                comment.get("timestamp", ""), root_meta.get("_created_at", "")
            )
            c_meta = self._fetch_user_meta(c_user)
            c_meta["time"] = c_time
            G.add_node(c_id, **c_meta)

            # Comment author engaged *because of* root post
            # Edge: c_id → root_id
            G.add_edge(c_id, root_id)

            # Replies: engaged because of the parent comment
            for reply in comment.get("replies", {}).get("data", []):
                r_id = str(reply["id"])
                r_user = reply.get("username", r_id)
                r_time = self._iso_to_seconds_since(
                    reply.get("timestamp", ""), root_meta.get("_created_at", "")
                )
                r_meta = self._fetch_user_meta(r_user)
                r_meta["time"] = r_time
                G.add_node(r_id, **r_meta)
                G.add_edge(r_id, c_id)
            
            time.sleep(_API_CALL_DELAY_S)

        self.ensure_required_attrs(G)
        return G

    # -----------------------------------------------------------------------
    # Real API internals
    # -----------------------------------------------------------------------

    def _api_get(
        self,
        endpoint: str,
        token: str,
        **params: str,
    ) -> dict:
        """Make a single authenticated Graph API call."""
        url = _GRAPH_API_BASE + endpoint
        params["access_token"] = token
        return _http_get(url, params, timeout=self.timeout)

    def _paginate(
        self,
        endpoint: str,
        token: str,
        fields: str,
        limit: int = 50,
    ) -> list[dict]:
        """
        Walk Graph API pagination cursors and collect all items.
        Stops after 500 items to prevent runaway calls on viral posts.
        """
        results: list[dict] = []
        params: dict[str, Any] = {
            "access_token": token,
            "fields": fields,
            "limit": limit,
        }
        url = _GRAPH_API_BASE + endpoint

        while url and len(results) < 500:
            data = _http_get(url, params if "access_token" in url else None)
            results.extend(data.get("data", []))
            # Follow next cursor if present
            paging = data.get("paging", {})
            url = paging.get("next", "")
            params = {}  # next URL is fully-qualified
            time.sleep(_API_CALL_DELAY_S)

        return results

    def _fetch_user_meta(self, user_id_or_name: str) -> dict[str, Any]:
        """
        Fetch public profile data and convert to WICO node attributes.

        Returns a dict with guaranteed keys: followers, time, friends.
        """
        try:
            data = self._api_get(
                f"/{user_id_or_name}",
                self.access_token,
                fields=_USER_FIELDS,
            )
        except RuntimeError as exc:
            logger.debug("Could not fetch user %s: %s", user_id_or_name, exc)
            data = {}

        joined_str: str = data.get("biography", "")  # no direct join date in Basic API
        account_age = self._estimate_account_age(data)

        return {
            "followers": int(data.get("followers_count", 0)),
            "friends": int(data.get("follows_count", 0)),
            "time": 0.0,  # filled by caller
            "account_age_days": account_age,
            "is_verified": bool(data.get("is_verified", False)),
            "username": data.get("username", user_id_or_name),
            "platform": "instagram",
        }

    def _build_from_reshares(self, post_id: str, token: str) -> nx.DiGraph:
        """
        Build cascade from the /media/{id}/reshared endpoint.

        This endpoint requires the instagram_manage_insights permission
        (available to Business/Creator accounts).  Raises RuntimeError
        if the endpoint returns no data or a permission error.
        """
        reshares = self._paginate(f"/{post_id}/reshared", token, "id,owner")
        if not reshares:
            raise RuntimeError("No reshare data (insufficient permissions or 0 reshares).")

        # Fetch root owner
        post_data = self._api_get(f"/{post_id}", token, fields="id,owner")
        root_id = str(post_data["owner"]["id"])

        G = nx.DiGraph()
        root_meta = self._fetch_user_meta(root_id)
        root_meta["time"] = 0.0
        G.add_node(root_id, **root_meta)

        for rs in reshares:
            rs_id = str(rs["id"])
            owner_id = str(rs.get("owner", {}).get("id", rs_id + "_u"))
            rs_meta = self._fetch_user_meta(owner_id)
            rs_meta["time"] = 0.0  # timestamps not available via reshare API
            G.add_node(rs_id, **rs_meta)
            # This resharer engaged because of root → rs_id → root_id
            G.add_edge(rs_id, root_id)

        self.ensure_required_attrs(G)
        return G

    def _search_hashtag_posts(
        self, hashtag: str, token: str, hours_back: int
    ) -> list[str]:
        """
        Use Hashtag Search API to get recent media IDs under a hashtag.

        Requires a Business/Creator account.
        Returns list of post IDs (up to 50).
        """
        # Step 1: resolve hashtag ID
        tag_data = self._api_get(
            "/ig_hashtag_search",
            token,
            q=hashtag,
        )
        tag_ids = tag_data.get("data", [])
        if not tag_ids:
            return []

        tag_id = tag_ids[0]["id"]

        # Step 2: fetch recent media
        cutoff = datetime.now(timezone.utc).timestamp() - hours_back * 3600
        posts = self._paginate(
            f"/{tag_id}/recent_media",
            token,
            fields="id,timestamp",
        )

        filtered = []
        for p in posts:
            ts_str = p.get("timestamp", "")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                    if ts >= cutoff:
                        filtered.append(str(p["id"]))
                except ValueError:
                    filtered.append(str(p["id"]))
            else:
                filtered.append(str(p["id"]))

        return filtered

    # -----------------------------------------------------------------------
    # Mock cascade generation
    # -----------------------------------------------------------------------

    def _generate_mock_cascade(self, post_id: str) -> nx.DiGraph:
        """
        Generate a synthetic propagation cascade matching WICO statistics.

        Construction strategy
        ---------------------
        1. Choose n = Uniform(20, 100) nodes.
        2. Build a random BFS-rooted tree: iterate nodes in shuffled order,
           each non-root picks a random parent from the already-visited set.
           → Every non-root has exactly in_degree == 1 from the tree.
           → Root (node 0) has in_degree == 0 by construction.
           → Tree is connected and has exactly ONE root. Guaranteed.
        3. Add cross-edges A→B where B ≠ root to raise avg_degree toward 2.82.
           Because B already has in_degree ≥ 1 from the tree, no new roots
           are created. Root's in_degree stays 0.
        4. Assign follower counts (lognormal), inter-arrival times (exponential).

        Edge convention: A→B means A's engagement was triggered by B's share.
        """
        rng = self._rng
        n = int(rng.integers(self._MOCK_MIN_NODES, self._MOCK_MAX_NODES + 1))

        node_ids = [f"{post_id}_u{i}" for i in range(n)]
        root_id = node_ids[0]

        G = nx.DiGraph()
        G.add_nodes_from(node_ids)

        # --- 1. Random BFS-rooted spanning tree ---
        # Visit nodes in a random order; each one selects a parent from the
        # already-committed set.  This produces a random recursive tree
        # (uniform over all labelled rooted trees when parent chosen uniformly).
        visit_order = list(range(1, n))
        rng.shuffle(visit_order)

        committed = [0]   # root is committed first
        for child_idx in visit_order:
            parent_idx = committed[int(rng.integers(0, len(committed)))]
            # parent → child: content/influence flows FROM parent TO child.
            G.add_edge(node_ids[parent_idx], node_ids[child_idx])
            committed.append(child_idx)


        # --- 2. Cross-edges (raise density toward WICO avg_degree ≈ 2.82) ---
        # Target total edges = n * avg_degree / 2  (treat as undirected for budget)
        tree_edge_count = n - 1
        target_total = int(n * self._MOCK_AVG_DEGREE / 2)
        cross_budget = max(
            int(n * self._MOCK_CROSS_EDGE_PROB),
            target_total - tree_edge_count,
        )

        # Non-root node indices (safe targets for cross-edges)
        non_root_indices = list(range(1, n))  # indices 1..n-1 are non-root

        added = 0
        attempts = 0
        while added < cross_budget and attempts < cross_budget * 15:
            a = int(rng.integers(0, n))
            # target must be non-root (to preserve unique root invariant)
            b = non_root_indices[int(rng.integers(0, len(non_root_indices)))]
            attempts += 1
            if a == b:
                continue
            if G.has_edge(node_ids[a], node_ids[b]):
                continue
            G.add_edge(node_ids[a], node_ids[b])
            added += 1

        # --- 3. Node attributes ---
        raw_followers = rng.lognormal(
            self._MOCK_FOLLOWER_MU, self._MOCK_FOLLOWER_SIGMA, size=n
        ).astype(int)

        # Root gets highest followers (original poster is typically influential)
        raw_followers[0] = int(np.max(raw_followers))

        # Timing: BFS depth in the REVERSED graph (influence flows from root)
        # G.reverse() shows "who influenced whom": root can reach everyone.
        depths = nx.single_source_shortest_path_length(G.reverse(), root_id)

        for i, node_id in enumerate(node_ids):
            depth = depths.get(node_id, 1)
            followers = int(raw_followers[i])
            friends = int(max(10, followers * float(rng.uniform(0.05, 0.8))))
            account_age = int(rng.integers(30, 3650))

            G.nodes[node_id].update(
                {
                    "followers": followers,
                    "friends": friends,
                    "time": 0.0 if node_id == root_id else float(
                        rng.exponential(self._MOCK_INTER_ARRIVAL_MEAN_S) * max(depth, 1)
                    ),
                    "account_age_days": account_age,
                    "is_verified": bool(followers > 100_000 and rng.random() < 0.3),
                    "username": node_id,
                    "platform": "instagram",
                    "mock": True,
                }
            )

        logger.debug("Mock cascade for %s: %s", post_id, self.summary(G))
        return G

    @staticmethod
    def _prufer_to_edges(nodes: list[int], sequence: list[int]) -> list[tuple[int, int]]:
        """
        Convert a Prüfer sequence to a list of tree edges.
        Standard O(n log n) algorithm.  Kept for unit-test coverage.
        (No longer used by _generate_mock_cascade — see BFS tree above.)
        """
        degree = [1] * len(nodes)
        for s in sequence:
            degree[s] += 1

        edges: list[tuple[int, int]] = []
        for s in sequence:
            for leaf in nodes:
                if degree[leaf] == 1:
                    edges.append((leaf, s))
                    degree[leaf] -= 1
                    degree[s] -= 1
                    break

        remaining = [v for v in nodes if degree[v] == 1]
        if len(remaining) >= 2:
            edges.append((remaining[0], remaining[1]))

        return edges

    # -----------------------------------------------------------------------
    # Utility helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _attach_platform_tag(G: nx.DiGraph, platform: str) -> None:
        """Add platform='instagram' to every node that is missing it."""
        for _, data in G.nodes(data=True):
            data.setdefault("platform", platform)

    @staticmethod
    def _iso_to_seconds_since(timestamp: str, reference: str) -> float:
        """
        Convert an ISO-8601 timestamp to seconds elapsed since *reference*.
        Returns 0.0 if either string is invalid.
        """
        def parse(s: str) -> float | None:
            try:
                return datetime.fromisoformat(
                    s.replace("Z", "+00:00")
                ).timestamp()
            except (ValueError, AttributeError):
                return None

        t = parse(timestamp)
        r = parse(reference)
        if t is None or r is None:
            return 0.0
        return max(0.0, t - r)

    @staticmethod
    def _estimate_account_age(data: dict) -> int:
        """
        Estimate account age in days from available API fields.
        Returns 365 as a conservative default when no signal is available.
        """
        # Instagram Basic Display API doesn't expose account creation date.
        # Heuristic: media_count as proxy for activity age.
        media_count = int(data.get("media_count", 0))
        if media_count > 1000:
            return 3000
        if media_count > 500:
            return 1500
        if media_count > 100:
            return 600
        return 365

    def _mock_node_meta(self, node_id: str) -> dict[str, Any]:
        """Generate synthetic metadata for a node ID in mock mode."""
        followers = int(
            self._rng.lognormal(self._MOCK_FOLLOWER_MU, self._MOCK_FOLLOWER_SIGMA)
        )
        return {
            "followers": followers,
            "friends": int(followers * self._rng.uniform(0.05, 0.8)),
            "time": 0.0,
            "account_age_days": int(self._rng.integers(30, 3650)),
            "is_verified": followers > 100_000,
            "username": node_id,
            "platform": "instagram",
            "mock": True,
        }

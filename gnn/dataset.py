"""
gnn/dataset.py
==============
Loaders for Twitter-15/16 and WICO propagation-tree datasets.

Tree file format (per BiGCN repo convention):
----------------------------------------------
Each line is either:
    ROOT -> tweet_id -> [t, uid, parent_id, nchild]
or an edge row:
    parent_id -> child_id -> [t, uid, parent_id, nchild]

Label file format:
    label\ttweet_id
e.g.:
    false\t123456789
    true\t987654321
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EDGE_RE = re.compile(
    r"'?(\w+)'?\s*->\s*'?(\w+)'?\s*->\s*\[([^\]]*)\]"
)


def _parse_edge_line(line: str) -> Optional[Tuple[str, str, List[float]]]:
    """Parse a single edge line from a BiGCN-format tree file.

    Returns (parent_id, child_id, [timestamp, user_id, parent_id, nchildren])
    or None if the line cannot be parsed.
    """
    m = _EDGE_RE.match(line.strip())
    if not m:
        return None
    parent, child, payload = m.groups()
    try:
        values = [float(x.strip()) for x in payload.split(",") if x.strip()]
    except ValueError:
        values = []
    return parent, child, values


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class CascadeTree:
    """Lightweight representation of one propagation tree."""

    def __init__(self, root_id: str, label: Optional[str] = None):
        self.root_id = root_id
        self.label = label                    # "true" / "false" / "unverified" etc.
        # adjacency: parent -> [children]
        self.children: Dict[str, List[str]] = {}
        # metadata per node: node_id -> [t, uid, parent_id, nchild]
        self.node_meta: Dict[str, List[float]] = {}
        self.nodes: set = {root_id}

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------
    def add_edge(self, parent: str, child: str, meta: List[float] = None):
        self.children.setdefault(parent, [])
        if child not in self.children[parent]:
            self.children[parent].append(child)
        self.nodes.add(parent)
        self.nodes.add(child)
        if meta:
            self.node_meta[child] = meta

    # ------------------------------------------------------------------
    # Graph metrics
    # ------------------------------------------------------------------
    def depth(self) -> int:
        """Longest path (in edges) from root to any leaf."""
        return self._max_depth(self.root_id, visited=set())

    def _max_depth(self, node: str, visited: set) -> int:
        if node in visited:
            return 0
        visited.add(node)
        kids = self.children.get(node, [])
        if not kids:
            return 0
        return 1 + max(self._max_depth(k, visited) for k in kids)

    def width(self) -> int:
        """Maximum number of nodes at any single BFS level."""
        if not self.children and len(self.nodes) == 1:
            return 1
        from collections import deque
        q: deque = deque([(self.root_id, 0)])
        level_count: Dict[int, int] = {}
        visited = set()
        while q:
            node, lvl = q.popleft()
            if node in visited:
                continue
            visited.add(node)
            level_count[lvl] = level_count.get(lvl, 0) + 1
            for child in self.children.get(node, []):
                q.append((child, lvl + 1))
        return max(level_count.values()) if level_count else 1

    def size(self) -> int:
        return len(self.nodes)

    def __repr__(self):
        return (f"CascadeTree(root={self.root_id!r}, label={self.label!r}, "
                f"nodes={self.size()}, depth={self.depth()}, width={self.width()})")


# ---------------------------------------------------------------------------
# File-level parsers
# ---------------------------------------------------------------------------

def _parse_tree_file(filepath: Path) -> CascadeTree:
    """Parse one Twitter-15/16-format tree file into a CascadeTree.

    This is the canonical parser; weibo_dataset.py delegates here (or to
    _parse_weibo_tree_file for format variations).
    """
    filepath = Path(filepath)
    root_id = filepath.stem          # filename without extension = tweet id

    tree = CascadeTree(root_id=root_id)

    with open(filepath, encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parsed = _parse_edge_line(line)
            if parsed is None:
                continue
            parent, child, meta = parsed
            # Normalise the sentinel "ROOT" -> actual root id
            if parent.upper() == "ROOT":
                parent = root_id
            if child.upper() == "ROOT":
                continue          # self-loops on ROOT are skip-worthy
            tree.add_edge(parent, child, meta)

    return tree


def _load_labels(label_file: Path) -> Dict[str, str]:
    """Load a BiGCN label.txt file.

    Expected formats (tab-separated OR colon-separated):
        false\\t123456
        label:tweet_id
    Returns dict mapping tweet_id -> label string.
    """
    labels: Dict[str, str] = {}
    with open(label_file, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                parts = line.split("\t", 1)
            elif ":" in line:
                parts = line.split(":", 1)
            else:
                continue
            if len(parts) == 2:
                label, tweet_id = parts[0].strip(), parts[1].strip()
                labels[tweet_id] = label.lower()
    return labels


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------

class Twitter15Dataset:
    """Loads the full Twitter-15 propagation-tree dataset."""

    LABEL_MAP = {
        "false rumour": 0, "false": 0,
        "true rumour": 1,  "true": 1,
        "unverified": 2,
        "non-rumour": 3,  "non-rumor": 3,
    }

    def __init__(self, tree_dir: Path, label_file: Path):
        self.tree_dir = Path(tree_dir)
        self.label_file = Path(label_file)
        self._labels: Dict[str, str] = _load_labels(self.label_file)
        self.trees: List[CascadeTree] = self._load_all()

    def _load_all(self) -> List[CascadeTree]:
        trees = []
        for fp in sorted(self.tree_dir.glob("*.txt")):
            tree = _parse_tree_file(fp)
            tree.label = self._labels.get(tree.root_id)
            trees.append(tree)
        return trees

    def __len__(self):
        return len(self.trees)

    def __repr__(self):
        return f"Twitter15Dataset(n={len(self)})"


class WICODataset:
    """Loader for the WICO COVID-conspiracy dataset.

    The tree format is identical to Twitter-15; only the label vocabulary
    differs: 'conspiracy' (false/harmful) and 'other' (benign).
    """

    def __init__(self, tree_dir: Path, label_file: Path):
        self.tree_dir = Path(tree_dir)
        self.label_file = Path(label_file)
        self._labels: Dict[str, str] = _load_labels(self.label_file)
        self.trees: List[CascadeTree] = self._load_all()

    def _load_all(self) -> List[CascadeTree]:
        trees = []
        for fp in sorted(self.tree_dir.glob("*.txt")):
            tree = _parse_tree_file(fp)
            tree.label = self._labels.get(tree.root_id)
            trees.append(tree)
        return trees

    def __len__(self):
        return len(self.trees)

    def __repr__(self):
        return f"WICODataset(n={len(self)})"
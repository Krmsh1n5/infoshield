"""
gnn/weibo_dataset.py
====================
Adapter for the Weibo (Sina Weibo) dataset.

Actual file format
------------------
weibotree.txt   — ALL events in a single tab-separated file:
    col 0: event_id       (string — the root post ID)
    col 1: parent_id      ("None" for root node, else local integer node index)
    col 2: node_id        (local integer node index within this event's tree)
    col 3: bow_features   (sparse bag-of-words "feat_id:count ..." — not used here)

weibo_id_label.txt — space-separated:
    event_id  label       (0 = rumour / false, 1 = non-rumour / true)

This format is entirely different from the BiGCN Twitter15 per-file format.
This module is self-contained — no imports from gnn/dataset.py.
"""

from __future__ import annotations

from collections import deque, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ── Label mapping ─────────────────────────────────────────────────────────────
WEIBO_LABEL_MAP: Dict[int, str] = {0: "rumour", 1: "non-rumour"}
WEIBO_INT_MAP:   Dict[str, int] = {"rumour": 0, "non-rumour": 1, "0": 0, "1": 1}


# ── Lightweight cascade tree ──────────────────────────────────────────────────

class CascadeTree:
    """Pure-Python directed tree for one propagation cascade.
    Nodes are local integer-string IDs (as they appear in weibotree.txt col 2/1).
    No torch / PyG dependencies.
    """

    __slots__ = ("event_id", "root_id", "label", "_children", "_nodes")

    def __init__(self, event_id: str, root_id: str, label: Optional[str] = None):
        self.event_id = event_id
        self.root_id  = root_id          # local node id of the root (col 2 where col 1 == "None")
        self.label    = label
        self._children: Dict[str, List[str]] = {}
        self._nodes: set = {root_id}

    def add_edge(self, parent: str, child: str) -> None:
        self._children.setdefault(parent, [])
        if child not in self._children[parent]:
            self._children[parent].append(child)
        self._nodes.add(parent)
        self._nodes.add(child)

    # ── structural metrics ────────────────────────────────────────────────────
    def depth(self) -> int:
        """Longest root-to-leaf path in hops."""
        visited: set = set()
        return self._dfs(self.root_id, visited)

    def _dfs(self, node: str, visited: set) -> int:
        if node in visited:
            return 0
        visited.add(node)
        kids = self._children.get(node, [])
        return 0 if not kids else 1 + max(self._dfs(k, visited) for k in kids)

    def width(self) -> int:
        """Max nodes at any single BFS level."""
        level_count: Dict[int, int] = {}
        seen: set = set()
        q: deque = deque([(self.root_id, 0)])
        while q:
            node, lvl = q.popleft()
            if node in seen:
                continue
            seen.add(node)
            level_count[lvl] = level_count.get(lvl, 0) + 1
            for c in self._children.get(node, []):
                q.append((c, lvl + 1))
        return max(level_count.values()) if level_count else 1

    def size(self) -> int:
        return len(self._nodes)

    def edges(self) -> List[Tuple[str, str]]:
        return [(p, c) for p, cs in self._children.items() for c in cs]

    def __repr__(self) -> str:
        return (f"CascadeTree(event={self.event_id!r}, label={self.label!r}, "
                f"nodes={self.size()}, depth={self.depth()}, width={self.width()})")


# ── File parsers ──────────────────────────────────────────────────────────────

def _load_weibo_labels(label_file: Path) -> Dict[str, str]:
    """Parse weibo_id_label.txt → {event_id: label_string}."""
    labels: Dict[str, str] = {}
    with open(label_file, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) >= 2:
                eid, raw_lbl = parts[0], parts[1]
                try:
                    labels[eid] = WEIBO_LABEL_MAP.get(int(raw_lbl), raw_lbl)
                except ValueError:
                    labels[eid] = raw_lbl.lower()
    return labels


def _parse_weibo_tree_file(tree_file: Path) -> Dict[str, CascadeTree]:
    """Parse weibotree.txt → {event_id: CascadeTree}.

    Format per line (tab-separated):
        event_id  parent_id  node_id  bow_features...
    parent_id == "None" marks the root node of that event's tree.
    """
    # First pass: collect all edges per event
    edges_by_event: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    root_by_event:  Dict[str, str] = {}

    with open(tree_file, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            event_id   = parts[0].strip()
            parent_str = parts[1].strip()
            node_str   = parts[2].strip()

            if parent_str == "None":
                root_by_event[event_id] = node_str
            else:
                edges_by_event[event_id].append((parent_str, node_str))

    # Second pass: build CascadeTree objects
    trees: Dict[str, CascadeTree] = {}
    all_events = set(root_by_event) | set(edges_by_event)
    for eid in all_events:
        root_id = root_by_event.get(eid, "1")   # fallback to "1" if no None row
        tree = CascadeTree(event_id=eid, root_id=root_id)
        for parent, child in edges_by_event.get(eid, []):
            tree.add_edge(parent, child)
        trees[eid] = tree

    return trees


# ── Public dataset class ──────────────────────────────────────────────────────

class WeiboDataset:
    """Loads the Weibo dataset from weibotree.txt + weibo_id_label.txt.

    Parameters
    ----------
    tree_file  : path to weibotree.txt
    label_file : path to weibo_id_label.txt

    Attributes
    ----------
    trees : List[CascadeTree]
    """

    def __init__(self, tree_file: Path, label_file: Path):
        self.tree_file  = Path(tree_file)
        self.label_file = Path(label_file)
        raw_labels = _load_weibo_labels(self.label_file)
        tree_dict  = _parse_weibo_tree_file(self.tree_file)

        self.trees: List[CascadeTree] = []
        for eid, tree in sorted(tree_dict.items()):
            tree.label = raw_labels.get(eid)
            self.trees.append(tree)

    def label_int(self, tree: CascadeTree) -> Optional[int]:
        if tree.label is None:
            return None
        return WEIBO_INT_MAP.get(tree.label.lower())

    def to_pyg_edges(self, idx: int) -> Tuple[str, List[Tuple[str, str]]]:
        t = self.trees[idx]
        return t.event_id, t.edges()

    def __len__(self) -> int:
        return len(self.trees)

    def __getitem__(self, idx: int) -> CascadeTree:
        return self.trees[idx]

    def __repr__(self) -> str:
        from collections import Counter
        return f"WeiboDataset(n={len(self)}, labels={dict(Counter(t.label for t in self.trees))})"
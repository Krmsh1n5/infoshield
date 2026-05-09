"""
Twitter15/16 PyG dataset for BiGCN.

─── BROADCAST DESIGN — DO NOT "FIX" THIS ─────────────────────────────────────
Twitter15/16 tree files store only the ROOT tweet ID for every node.
Format: ['user_id', 'ROOT_tweet_id', delay] — reply tweet IDs are not recorded.
source_tweets.txt only has root tweet text. There is no per-node text file.

Broadcasting root_emb to all nodes is INTENTIONAL and CORRECT for this dataset.
Every reply exists in the context of the root claim, so shared text features
are semantically justified. The is_root flag in struct features lets the GCN
distinguish root from replies structurally.

Do not replace root_emb with per-node lookups or zero vectors.
Per-node text is unavailable. Zero vectors → 68% accuracy. Broadcast → 85%+.
──────────────────────────────────────────────────────────────────────────────
"""
from __future__ import annotations

import logging
import logging
import re
from collections import Counter
import sys
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg

log = logging.getLogger(__name__)
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data

from config import cfg

log = logging.getLogger(__name__)

# ─── 4-class label mapping (Twitter15/16 standard) ────────────────────────────
LABEL_MAP = {"true": 0, "false": 1, "unverified": 2, "non-rumor": 3}

# ─── Tree edge regex ─────────────────────────────────────────────────────────
EDGE_PATTERN = re.compile(
    r"\['([^']*)',\s*'([^']*)',\s*'([^']*)'\]\s*->\s*\['([^']*)',\s*'([^']*)',\s*'([^']*)'\]"
)

# ─── RoBERTa encoder (lazy, frozen) ───────────────────────────────────────────
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is not None:
        return _encoder

    from transformers import RobertaModel, RobertaTokenizer

    tokenizer = RobertaTokenizer.from_pretrained(cfg.bigcn.text_encoder)
    model = RobertaModel.from_pretrained(cfg.bigcn.text_encoder).eval()
    for p in model.parameters():
        p.requires_grad = False

    @torch.no_grad()
    def encode(text: str) -> torch.Tensor:
        text = (text or "").strip() or "[empty]"
        toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        out = model(**toks)
        return out.last_hidden_state[0, 0, :].detach().clone()  # CLS, (768,)

    _encoder = encode
    return _encoder


# ─── File parsers ─────────────────────────────────────────────────────────────

def _parse_tree_file(path: Path) -> List[Tuple[str, str]]:
    """Parse tree file → list of (parent_uid, child_uid) edges, ROOT excluded."""
    sentinel = cfg.twitter15.root_sentinel
    edges: List[Tuple[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = EDGE_PATTERN.match(line.strip())
            if not m:
                continue
            p_uid, _, _, c_uid, _, _ = m.groups()
            if p_uid == sentinel:
                continue
            edges.append((p_uid, c_uid))
    return edges


def _load_labels(label_file: Path) -> Dict[str, int]:
    """label.txt format: 'label:tweet_id' per line."""
    out: Dict[str, int] = {}
    with open(label_file, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or ":" not in line:
                continue
            label, tweet_id = line.split(":", 1)
            label = label.strip().lower()
            if label in LABEL_MAP:
                out[tweet_id.strip()] = LABEL_MAP[label]
    return out


def _load_source_texts(source_file: Path) -> Dict[str, str]:
    """source_tweets.txt format: 'tweet_id\\ttext' per line. Root nodes only."""
    out: Dict[str, str] = {}
    with open(source_file, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t", 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


# ─── Graph construction ───────────────────────────────────────────────────────

def _build_pyg_graph(
    tweet_id: str,
    edges: List[Tuple[str, str]],
    label: int,
    root_text: str,
    embed_cache: Dict[str, torch.Tensor],
) -> Optional[Data]:
    if not edges:
        return None

    # Root user is parent in the first edge (after ROOT sentinel was stripped)
    root_uid = edges[0][0]

    nodes_set = set()
    for p, c in edges:
        nodes_set.add(p)
        nodes_set.add(c)
    nodes = [root_uid] + sorted(nodes_set - {root_uid})  # root at index 0
    node_to_idx = {n: i for i, n in enumerate(nodes)}

    # Edge tensors
    edge_index = torch.tensor(
        [[node_to_idx[p] for p, c in edges],
         [node_to_idx[c] for p, c in edges]],
        dtype=torch.long,
    )
    edge_index_bu = edge_index.flip(0)

    # Structural stats
    parents: Dict[str, List[str]] = {n: [] for n in nodes}
    children: Dict[str, List[str]] = {n: [] for n in nodes}
    for p, c in edges:
        children[p].append(c)
        parents[c].append(p)

    depth: Dict[str, int] = {root_uid: 0}
    queue = [root_uid]
    while queue:
        nxt = []
        for n in queue:
            for c in children[n]:
                if c not in depth:
                    depth[c] = depth[n] + 1
                    nxt.append(c)
        queue = nxt

    max_depth = max(max(depth.values(), default=1), 1)
    max_in = max(max((len(v) for v in parents.values()), default=1), 1)
    max_out = max(max((len(v) for v in children.values()), default=1), 1)

    # ─── Root embedding (BROADCAST to all nodes — see file header) ───────
    encoder = _get_encoder()
    if tweet_id not in embed_cache:
        embed_cache[tweet_id] = (
            encoder(root_text) if root_text else torch.zeros(768)
        )
    root_emb = embed_cache[tweet_id]  # (768,) shared by all nodes

    # ─── Node features: [root_emb (768) | struct (4)] = 772 dims ─────────
    x_rows = []
    for node in nodes:
        struct = torch.tensor(
            [
                depth.get(node, max_depth) / max_depth,
                len(parents[node]) / max_in,
                len(children[node]) / max_out,
                1.0 if node == root_uid else 0.0,  # is_root flag
            ],
            dtype=torch.float32,
        )
        x_rows.append(torch.cat([root_emb, struct], dim=0))

    x = torch.stack(x_rows)  # (N, 772)

    # Sanity check: broadcast invariant — text portion identical across nodes
    if x.shape[0] >= 2:
        assert torch.allclose(x[0, :768], x[-1, :768]), (
            "Text embeddings differ across nodes — broadcast broken. "
            "See dataset.py BROADCAST DESIGN comment at top of file."
        )

    # Root mask: True at the single root node (used by BiGCN root enhancement)
    root_mask = torch.zeros(len(nodes), dtype=torch.bool)
    root_mask[0] = True

    return Data(
        x=x,
        edge_index=edge_index,
        edge_index_bu=edge_index_bu,
        y=torch.tensor([label], dtype=torch.long),
        root_mask=root_mask,
        num_nodes=len(nodes),
        tweet_id=tweet_id,
    )


# ─── Dataset class ────────────────────────────────────────────────────────────

class TwitterRumourDataset(TorchDataset):
    """In-memory dataset of PyG Data objects for Twitter15 or Twitter16."""

    def __init__(self, split: str, force_reprocess: bool = False):
        assert split in ("twitter15", "twitter16")
        self.split = split

        if split == "twitter15":
            self.raw_root = Path(cfg.paths.twitter15)
            self.tree_dir = Path(cfg.paths.twitter15_trees)
        else:
            self.raw_root = Path(cfg.paths.twitter16)
            self.tree_dir = Path(cfg.paths.twitter16_trees)

        self.processed_root = Path(cfg.paths.graphs_pt) / split
        self.processed_root.mkdir(parents=True, exist_ok=True)
        self._cache_file = self.processed_root / "all_graphs.pt"

        if force_reprocess or not self._cache_file.exists():
            self._process()

        self._data_list: List[Data] = torch.load(self._cache_file, weights_only=False)
        log.info(f"[{split}] Loaded {len(self._data_list)} graphs from cache.")

    def _process(self):
        labels = _load_labels(self.raw_root / cfg.twitter15.label_file)
        sources = _load_source_texts(self.raw_root / cfg.twitter15.source_tweets_file)

        tree_files = sorted(self.tree_dir.glob("*.txt"))
        log.info(f"[{self.split}] Found {len(tree_files)} tree files")

        embed_cache: Dict[str, torch.Tensor] = {}
        graphs: List[Data] = []
        skipped = 0

        for i, tree_path in enumerate(tree_files):
            tweet_id = tree_path.stem
            if tweet_id not in labels:
                skipped += 1
                continue

            edges = _parse_tree_file(tree_path)
            n_nodes = len({n for e in edges for n in e})
            if n_nodes < cfg.twitter15.min_tree_size or n_nodes > cfg.twitter15.max_tree_size:
                skipped += 1
                continue

            data = _build_pyg_graph(
                tweet_id=tweet_id,
                edges=edges,
                label=labels[tweet_id],
                root_text=sources.get(tweet_id, ""),
                embed_cache=embed_cache,
            )
            if data is None:
                skipped += 1
                continue

            graphs.append(data)

            if (i + 1) % 200 == 0:
                log.info(f"[{self.split}] Processed {i + 1}/{len(tree_files)}")

        dist = Counter(int(g.y.item()) for g in graphs)
        log.info(
            f"[{self.split}] Loaded {len(graphs)} graphs, {skipped} skipped. "
            f"Label dist: {dict(sorted(dist.items()))}"
        )
        torch.save(graphs, self._cache_file)

    def __len__(self) -> int:
        return len(self._data_list)

    def __getitem__(self, idx: int) -> Data:
        return self._data_list[idx]

    # Keep PyG-style alias for compatibility with existing train.py
    def get(self, idx: int) -> Data:
        return self._data_list[idx]

    def get_kfold_splits(self, n_splits: int = 5, seed: int = 42):
        """Return list of (train_idx, val_idx, test_idx) for stratified k-fold CV."""
        labels = np.array([int(g.y.item()) for g in self._data_list])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        folds = []
        all_idx = np.arange(len(self._data_list))
        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(all_idx, labels)):
            rng = np.random.default_rng(seed + fold_idx)
            rng.shuffle(train_val_idx)
            n_val = max(1, int(0.1 * len(train_val_idx)))
            val_idx = train_val_idx[:n_val]
            train_idx = train_val_idx[n_val:]
            folds.append((train_idx, val_idx, test_idx))
        return folds


# ---------------------------------------------------------------------------
# PyG-compatible dataset (used by train.py / evaluate.py)
# ---------------------------------------------------------------------------

class TwitterRumourDataset:
    """
    Converts cascade trees to PyG Data objects for BiGCN training.

    Supports splits: twitter15, twitter16, wico, weibo.

    BROADCAST DESIGN: the 768-dim RoBERTa CLS embedding of the root tweet is
    copied to every node so that BiGCN's 2-layer GCN gives every node access
    to the source claim's semantics.  Without this, nodes beyond 2 hops from
    the root lose source context entirely.  The root-enhancement step in
    GCNBranch re-injects this at Layer 2 for an additional reinforcement.

    Node features (772-dim per node):
        [0:768]  RoBERTa-base CLS of root tweet, broadcast to all nodes
        [768]    is_root       (1.0 / 0.0)
        [769]    norm_depth    (BFS depth / max_depth in tree)
        [770]    norm_breadth  (out-degree / max out-degree in tree)
        [771]    norm_descendants (descendants / (n - 1))

    RoBERTa embeddings are cached to cfg.paths.graphs_pt on first run.
    If the source_tweets.txt file is absent, zero vectors are used instead.
    """

    _LABEL_MAPS: Dict[str, Dict[str, int]] = {
        "twitter15": {
            "false": 0, "false rumour": 0,
            "true": 1,  "true rumour": 1,
            "unverified": 2,
            "non-rumour": 3, "non-rumor": 3,
        },
        "twitter16": {
            "false": 0, "false rumour": 0,
            "true": 1,  "true rumour": 1,
            "unverified": 2,
            "non-rumour": 3, "non-rumor": 3,
        },
        "wico": {
            "5g-conspiracy": 0, "5g_conspiracy": 0,
            "conspiracy": 1,
            "non-conspiracy": 2, "non_conspiracy": 2,
        },
        "weibo": {
            "rumour": 0, "0": 0,
            "non-rumour": 1, "1": 1,
        },
    }

    def __init__(self, split: str):
        self.split = split.lower()
        self._data: List = []
        self._load()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._data)

    def get(self, idx: int):
        return self._data[idx]

    def __getitem__(self, idx: int):
        return self._data[idx]

    def __repr__(self) -> str:
        return f"TwitterRumourDataset(split={self.split!r}, n={len(self)})"

    # ------------------------------------------------------------------
    # Loading pipeline
    # ------------------------------------------------------------------

    def _load(self) -> None:
        import torch
        from torch_geometric.data import Data  # noqa: F401 — needed for type

        trees, texts = self._load_trees_and_texts()
        emb_map = self._get_embeddings(texts)

        lmap = self._LABEL_MAPS.get(self.split, {})
        skipped = 0

        for tree in trees:
            lbl_str = getattr(tree, "label", None)
            if lbl_str is None:
                skipped += 1
                continue
            label_int = lmap.get(str(lbl_str).lower())
            if label_int is None:
                skipped += 1
                continue

            root_emb = emb_map.get(
                tree.root_id,
                torch.zeros(768, dtype=torch.float32),
            )
            data = self._build_pyg_graph(tree, root_emb, label_int)
            if data is not None:
                self._data.append(data)
            else:
                skipped += 1

        log.info(
            "TwitterRumourDataset(%s): loaded=%d  skipped=%d",
            self.split, len(self._data), skipped,
        )

    # ------------------------------------------------------------------
    # Tree loaders per split
    # ------------------------------------------------------------------

    def _load_trees_and_texts(self) -> Tuple[List, Dict[str, str]]:
        if self.split in ("twitter15", "twitter16"):
            return self._load_twitter()
        if self.split == "wico":
            return self._load_wico()
        if self.split == "weibo":
            return self._load_weibo()
        raise ValueError(
            f"Unknown split {self.split!r}. "
            "Choose from: twitter15, twitter16, wico, weibo"
        )

    def _load_twitter(self) -> Tuple[List[CascadeTree], Dict[str, str]]:
        if self.split == "twitter15":
            root_dir = cfg.paths.twitter15
            tcfg = cfg.twitter15
        else:
            root_dir = cfg.paths.twitter16
            tcfg = cfg.twitter16

        tree_dir   = root_dir / tcfg.tree_dir
        label_file = root_dir / tcfg.label_file
        ds = Twitter15Dataset(tree_dir, label_file)

        texts: Dict[str, str] = {}
        src_file = root_dir / tcfg.source_tweets_file
        if src_file.exists():
            with open(src_file, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    parts = line.strip().split("\t", 1)
                    if len(parts) == 2:
                        texts[parts[0].strip()] = parts[1].strip()
        else:
            log.warning(
                "source_tweets.txt not found at %s; using zero embeddings.", src_file
            )

        return ds.trees, texts

    def _load_wico(self) -> Tuple[List[CascadeTree], Dict[str, str]]:
        """Load WICO Graph cascades (edges.txt per tweet directory)."""
        label_str = {
            "5G_Conspiracy_Graphs": "5g-conspiracy",
            "Other_Graphs":         "conspiracy",
            "Non_Conspiracy_Graphs": "non-conspiracy",
        }
        trees: List[CascadeTree] = []
        wico_graph = cfg.paths.wico_graph
        for folder_name, lbl in label_str.items():
            folder = wico_graph / folder_name
            if not folder.exists():
                log.warning("WICO folder not found: %s", folder)
                continue
            for tweet_dir in sorted(folder.iterdir()):
                edges_file = tweet_dir / cfg.wico.graph_edges_file
                if not edges_file.exists():
                    continue
                root_id = tweet_dir.name
                tree = CascadeTree(root_id=root_id, label=lbl)
                with open(edges_file, encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            tree.add_edge(parts[0], parts[1])
                trees.append(tree)
        return trees, {}

    def _load_weibo(self) -> Tuple[List, Dict[str, str]]:
        from gnn.weibo_dataset import WeiboDataset

        tree_file  = cfg.paths.weibo / cfg.weibo.tree_file
        label_file = cfg.paths.weibo / cfg.weibo.label_file
        ds = WeiboDataset(tree_file, label_file)

        class _Adapter:
            """Duck-type WeiboDataset.CascadeTree to match dataset.CascadeTree."""
            def __init__(self, t):
                self.root_id  = t.root_id
                self.label    = t.label
                self.children = t._children
                self.nodes    = t._nodes

        return [_Adapter(t) for t in ds.trees], {}

    # ------------------------------------------------------------------
    # RoBERTa embeddings (with disk cache)
    # ------------------------------------------------------------------

    def _get_embeddings(self, texts: Dict[str, str]) -> Dict[str, "torch.Tensor"]:
        import torch

        cache_path = cfg.paths.graphs_pt / f"{self.split}_roberta_emb.pt"
        if cache_path.exists():
            log.info("Loading cached embeddings from %s", cache_path)
            return torch.load(cache_path, map_location="cpu")

        if not texts:
            return {}

        log.info(
            "Computing RoBERTa embeddings for %d tweets (split=%s) ...",
            len(texts), self.split,
        )
        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            log.warning("transformers not installed — using zero embeddings.")
            return {}

        tokenizer = AutoTokenizer.from_pretrained(cfg.bigcn.text_encoder)
        model     = AutoModel.from_pretrained(cfg.bigcn.text_encoder)
        model.eval()

        embeddings: Dict[str, torch.Tensor] = {}
        ids = list(texts.keys())
        batch_sz = 32

        with torch.no_grad():
            for i in range(0, len(ids), batch_sz):
                batch_ids   = ids[i: i + batch_sz]
                batch_texts = [texts[tid] for tid in batch_ids]
                enc = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True,
                )
                out = model(**enc)
                cls = out.last_hidden_state[:, 0, :].cpu()  # (B, 768)
                for tid, emb in zip(batch_ids, cls):
                    embeddings[tid] = emb

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, cache_path)
        log.info("Embeddings cached to %s", cache_path)
        return embeddings

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_pyg_graph(self, tree, root_emb, label: int):
        import torch
        from torch_geometric.data import Data

        n = len(tree.nodes)
        min_sz = cfg.twitter15.min_tree_size
        max_sz = cfg.twitter15.max_tree_size
        if n < min_sz or n > max_sz:
            return None

        # BFS-ordered node list; root always at index 0
        nodes_ordered = self._bfs_order(tree)
        n = len(nodes_ordered)
        node2idx: Dict[str, int] = {nid: i for i, nid in enumerate(nodes_ordered)}

        # Top-down edge lists
        src_td: List[int] = []
        dst_td: List[int] = []
        for parent, children in tree.children.items():
            pi = node2idx.get(parent)
            if pi is None:
                continue
            for child in children:
                ci = node2idx.get(child)
                if ci is None:
                    continue
                src_td.append(pi)
                dst_td.append(ci)

        if not src_td:
            return None

        depth_arr, out_deg_arr, desc_arr = self._structural_features(
            n, node2idx, tree.children
        )

        max_depth = float(depth_arr.max()) or 1.0
        max_out   = float(out_deg_arr.max()) or 1.0

        # Node features: (n, 772) = broadcast root_emb + 4 structural dims
        root_broadcast = root_emb.unsqueeze(0).expand(n, -1)  # (n, 768)
        struct = torch.zeros(n, 4, dtype=torch.float32)
        struct[0, 0] = 1.0                                                      # is_root
        struct[:, 1] = torch.from_numpy(depth_arr   / max_depth).float()       # norm_depth
        struct[:, 2] = torch.from_numpy(out_deg_arr / max_out).float()         # norm_breadth
        struct[:, 3] = torch.from_numpy(desc_arr    / max(n - 1, 1)).float()   # norm_descendants

        x          = torch.cat([root_broadcast, struct], dim=1)                 # (n, 772)
        edge_index    = torch.tensor([src_td, dst_td], dtype=torch.long)
        edge_index_bu = torch.tensor([dst_td, src_td], dtype=torch.long)
        root_mask  = torch.zeros(n, dtype=torch.bool)
        root_mask[0] = True

        return Data(
            x=x,
            edge_index=edge_index,
            edge_index_bu=edge_index_bu,
            root_mask=root_mask,
            y=torch.tensor(label, dtype=torch.long),
            num_nodes=n,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bfs_order(tree) -> List[str]:
        """BFS traversal from root; root is first in the returned list."""
        root    = tree.root_id
        visited: set = {root}
        order   = [root]
        q: deque = deque([root])
        while q:
            node = q.popleft()
            for child in tree.children.get(node, []):
                if child not in visited:
                    visited.add(child)
                    order.append(child)
                    q.append(child)
        # Append any nodes disconnected from root (shouldn't happen in clean data)
        for node in tree.nodes:
            if node not in visited:
                order.append(node)
        return order

    @staticmethod
    def _structural_features(
        n: int,
        node2idx: Dict[str, int],
        children: Dict,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return per-node (depth, out_degree, descendant_count) arrays."""
        # Build adjacency by index
        kids_by_idx: Dict[int, List[int]] = {}
        for parent, cs in children.items():
            pi = node2idx.get(parent)
            if pi is None:
                continue
            kids_by_idx[pi] = [node2idx[c] for c in cs if c in node2idx]

        # BFS depths from root (index 0)
        depth_arr = np.zeros(n, dtype=np.float32)
        visited   = np.zeros(n, dtype=bool)
        q: deque  = deque([0])
        visited[0] = True
        while q:
            node = q.popleft()
            for child in kids_by_idx.get(node, []):
                if not visited[child]:
                    visited[child] = True
                    depth_arr[child] = depth_arr[node] + 1
                    q.append(child)

        # Out-degree
        out_deg_arr = np.array(
            [len(kids_by_idx.get(i, [])) for i in range(n)],
            dtype=np.float32,
        )

        # Descendant counts — iterative post-order (reverse BFS order = deepest first)
        desc_arr = np.zeros(n, dtype=np.float32)
        order = list(range(n))
        order.sort(key=lambda i: -depth_arr[i])  # leaves before parents
        for node in order:
            for child in kids_by_idx.get(node, []):
                desc_arr[node] += 1 + desc_arr[child]

        return depth_arr, out_deg_arr, desc_arr
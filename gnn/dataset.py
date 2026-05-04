"""
gnn/dataset.py
==============
PyG dataset loader for Twitter15 / Twitter16 rumour propagation trees.

Pipeline
--------
1. Parse every tree/<tweet_id>.txt into a directed graph (top-down + bottom-up).
2. Encode the root tweet text with frozen RoBERTa (once, offline).
3. Broadcast root embedding as the feature vector for every node.
4. Persist each tree as a PyG Data object at cfg.paths.graphs_pt.
5. Expose a 5-fold cross-validation splitter matching published benchmarks.

Usage
-----
    from gnn.dataset import TwitterRumourDataset, get_cv_splits

    ds15 = TwitterRumourDataset("twitter15")
    ds16 = TwitterRumourDataset("twitter16")
    combined = TwitterRumourDataset("twitter15") + TwitterRumourDataset("twitter16")
    folds   = get_cv_splits(combined, n_splits=5, seed=42)
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from sklearn.model_selection import StratifiedKFold
from transformers import RobertaTokenizer, RobertaModel

# Import project-level config
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_MAP_15: Dict[str, int] = {
    "true": 0,
    "false": 1,
    "unverified": 2,
    "non-rumor": 3,
}

LABEL_MAP_16: Dict[str, int] = LABEL_MAP_15  # identical schema


# ---------------------------------------------------------------------------
# Tree-file parser
# ---------------------------------------------------------------------------

_EDGE_RE = re.compile(
    r"\['(?P<pu>[^']+)',\s*'(?P<pt>[^']+)',\s*'?(?P<pd>[^']+)'?\]"
    r"\s*->\s*"
    r"\['(?P<cu>[^']+)',\s*'(?P<ct>[^']+)',\s*'?(?P<cd>[^']+)'?\]"
)


def _parse_tree_file(path: Path) -> Tuple[List[Tuple[str, str]], str]:
    """
    Parse a Twitter15/16 tree file.

    Returns
    -------
    edges : list of (parent_tweet_id, child_tweet_id)
        ROOT sentinel is stripped; the source tweet ID is inferred from the
        sentinel line.
    root_tweet_id : str
    """
    edges: List[Tuple[str, str]] = []
    root_tweet_id: Optional[str] = None

    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = _EDGE_RE.match(line)
            if m is None:
                log.warning("Unparseable line in %s: %r", path.name, line)
                continue

            parent_uid, parent_tid = m.group("pu"), m.group("pt")
            child_uid,  child_tid  = m.group("cu"), m.group("ct")

            if parent_uid == cfg.twitter15.root_sentinel:
                root_tweet_id = child_tid
            else:
                edges.append((parent_tid, child_tid))

    if root_tweet_id is None:
        raise ValueError(f"No ROOT sentinel found in {path}")

    return edges, root_tweet_id


# ---------------------------------------------------------------------------
# RoBERTa encoder (offline, frozen)
# ---------------------------------------------------------------------------

class _RobertaEncoder:
    """Singleton that lazily loads RoBERTa and encodes text to 768-d vectors."""

    _instance: Optional[_RobertaEncoder] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _load(self):
        if self._loaded:
            return
        model_name: str = cfg.bigcn.text_encoder          # "roberta-base"
        log.info("Loading tokenizer/model: %s", model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model     = RobertaModel.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self._loaded = True

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """Return a (768,) float32 tensor — CLS embedding of *text*."""
        self._load()
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=False,
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        out = self.model(**inputs)
        # CLS token embedding, moved to CPU
        return out.last_hidden_state[:, 0, :].squeeze(0).cpu()


_encoder = _RobertaEncoder()


# ---------------------------------------------------------------------------
# Core graph builder
# ---------------------------------------------------------------------------

def _build_pyg_graph(
    tweet_id: str,
    edges: List[Tuple[str, str]],
    label: int,
    root_text: str,
    embed_cache: Dict[str, torch.Tensor],
) -> Data:
    """
    Construct a PyG Data object for one cascade tree.

    Node features
    -------------
    Every node gets the RoBERTa CLS embedding of the root tweet text, plus structural node features:
    depth, in_degree, out_degree, is_root

    Feature dimension = 768 + 4 = 772.
    Edge indices
    ------------
    edge_index    : top-down  (parent → child)
    edge_index_bu : bottom-up (child  → parent)
    """
    # --- collect unique nodes -------------------------------------------
    all_nodes: List[str] = [tweet_id]  # root is node 0
    for p, c in edges:
        if p not in all_nodes:
            all_nodes.append(p)
        if c not in all_nodes:
            all_nodes.append(c)

    node2idx = {n: i for i, n in enumerate(all_nodes)}
    n = len(all_nodes)

    # --- node features: broadcast root embedding ------------------------
    # --- node features: root embedding + structural features -------------
    if tweet_id not in embed_cache:
        embed_cache[tweet_id] = _encoder.encode(root_text)
    root_emb = embed_cache[tweet_id]                         # (768,)

    children: Dict[str, List[str]] = {node: [] for node in all_nodes}
    parents: Dict[str, List[str]] = {node: [] for node in all_nodes}

    for p, c in edges:
        children[p].append(c)
        parents[c].append(p)

    # Compute BFS depth from root tweet node.
    depth: Dict[str, int] = {tweet_id: 0}
    queue: List[str] = [tweet_id]

    while queue:
        current = queue.pop(0)
        for child in children.get(current, []):
            if child not in depth:
                depth[child] = depth[current] + 1
                queue.append(child)

    max_depth = max(depth.values()) if depth else 1
    max_in_degree = max((len(parents[node]) for node in all_nodes), default=1)
    max_out_degree = max((len(children[node]) for node in all_nodes), default=1)

    max_depth = max(max_depth, 1)
    max_in_degree = max(max_in_degree, 1)
    max_out_degree = max(max_out_degree, 1)

    x_rows: List[torch.Tensor] = []

    for node in all_nodes:
        struct = torch.tensor(
            [
                depth.get(node, max_depth) / max_depth,
                len(parents[node]) / max_in_degree,
                len(children[node]) / max_out_degree,
                1.0 if node == tweet_id else 0.0,
            ],
            dtype=torch.float32,
        )
        x_rows.append(torch.cat([root_emb, struct], dim=0))

    x = torch.stack(x_rows, dim=0)                           # (n, 772)

    # --- edge index (top-down) ------------------------------------------
    if edges:
        src = torch.tensor([node2idx[p] for p, _ in edges], dtype=torch.long)
        dst = torch.tensor([node2idx[c] for _, c in edges], dtype=torch.long)
        edge_index    = torch.stack([src, dst], dim=0)       # top-down
        edge_index_bu = torch.stack([dst, src], dim=0)       # bottom-up
    else:
        # Single-node tree (shouldn't happen after size filtering)
        edge_index    = torch.zeros((2, 0), dtype=torch.long)
        edge_index_bu = torch.zeros((2, 0), dtype=torch.long)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_index_bu=edge_index_bu,
        y=torch.tensor(label, dtype=torch.long),
        num_nodes=n,
        tweet_id=tweet_id,
    )


# ---------------------------------------------------------------------------
# Source-tweet text loader
# ---------------------------------------------------------------------------

def _load_source_texts(source_file: Path) -> Dict[str, str]:
    """Return {tweet_id: text} from source_tweets.txt."""
    texts: Dict[str, str] = {}
    if not source_file.exists():
        log.warning("source_tweets.txt not found at %s", source_file)
        return texts
    with open(source_file, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                texts[parts[0]] = parts[1]
    return texts


def _load_labels(label_file: Path, label_map: Dict[str, int]) -> Dict[str, int]:
    """Return {tweet_id: int_label} from label.txt."""
    labels: Dict[str, int] = {}
    if not label_file.exists():
        log.warning("label.txt not found at %s", label_file)
        return labels
    with open(label_file, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if ":" not in line:
                continue
            lbl_str, tid = line.split(":", 1)
            lbl_str = lbl_str.strip().lower()
            tid = tid.strip()
            if lbl_str in label_map:
                labels[tid] = label_map[lbl_str]
            else:
                log.warning("Unknown label %r in %s", lbl_str, label_file)
    return labels


# ---------------------------------------------------------------------------
# TwitterRumourDataset
# ---------------------------------------------------------------------------

class TwitterRumourDataset(Dataset):
    """
    In-memory PyG dataset for Twitter15 / Twitter16.

    Parameters
    ----------
    split : "twitter15" | "twitter16"
    force_reprocess : bool
        If True, ignore cached .pt files and re-parse from raw.

    Attributes
    ----------
    data_list : list[Data]
    labels    : np.ndarray  (int, shape (N,))  — for stratified splits
    """

    def __init__(
        self,
        split: str = "twitter15",
        force_reprocess: bool = False,
    ):
        super().__init__()
        assert split in ("twitter15", "twitter16"), \
            f"split must be 'twitter15' or 'twitter16', got {split!r}"

        self.split = split
        self._ds_cfg = cfg.twitter15 if split == "twitter15" else cfg.twitter16
        self._label_map = LABEL_MAP_15 if split == "twitter15" else LABEL_MAP_16

        # Paths
        self._raw_root    = Path(cfg.paths.twitter15 if split == "twitter15"
                                 else cfg.paths.twitter16)
        self._tree_dir    = Path(cfg.paths.twitter15_trees if split == "twitter15"
                                 else cfg.paths.twitter16_trees)
        self._cache_dir   = Path(cfg.paths.graphs_pt) / split
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self.data_list: List[Data] = []
        self.labels: np.ndarray    = np.array([], dtype=int)

        self._load(force_reprocess)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_path(self, tweet_id: str) -> Path:
        return self._cache_dir / f"{tweet_id}.pt"

    def _load(self, force_reprocess: bool) -> None:
        label_file  = self._raw_root / self._ds_cfg.label_file
        source_file = self._raw_root / self._ds_cfg.source_tweets_file

        labels_map  = _load_labels(label_file, self._label_map)
        source_map  = _load_source_texts(source_file)

        tree_files = sorted(self._tree_dir.glob("*.txt"))
        if not tree_files:
            raise FileNotFoundError(
                f"No tree files found in {self._tree_dir}. "
                "Check cfg.paths.twitter15_trees / twitter16_trees."
            )

        log.info("[%s] Found %d tree files", self.split, len(tree_files))

        embed_cache: Dict[str, torch.Tensor] = {}
        skipped = 0
        data_list: List[Data]  = []
        label_list: List[int]  = []

        min_size: int = self._ds_cfg.min_tree_size
        max_size: int = self._ds_cfg.max_tree_size

        for tf in tree_files:
            tweet_id = tf.stem  # filename without .txt

            # --- label check ---
            if tweet_id not in labels_map:
                skipped += 1
                continue
            label = labels_map[tweet_id]

            # --- load or rebuild .pt cache ---
            cache_p = self._cache_path(tweet_id)
            if cache_p.exists() and not force_reprocess:
                try:
                    data = torch.load(cache_p, weights_only=False)
                    data_list.append(data)
                    label_list.append(int(data.y.item()))
                    continue
                except Exception as e:
                    log.warning("Corrupt cache %s (%s) — rebuilding", cache_p, e)

            # --- parse tree ---
            try:
                edges, root_id = _parse_tree_file(tf)
            except ValueError as e:
                log.warning("Skipping %s: %s", tf.name, e)
                skipped += 1
                continue

            # --- size filter ---
            node_set = {tweet_id}
            for p, c in edges:
                node_set.add(p)
                node_set.add(c)
            n_nodes = len(node_set)
            # n_nodes = len({n for pair in edges for n in pair}) + 1
            if not (min_size <= n_nodes <= max_size):
                skipped += 1
                continue

            # --- root text ---
            root_text = source_map.get(tweet_id, "")
            if not root_text:
                log.debug("No source text for %s — using empty string", tweet_id)

            # --- build PyG Data ---
            data = _build_pyg_graph(
                tweet_id=tweet_id,
                edges=edges,
                label=label,
                root_text=root_text,
                embed_cache=embed_cache,
            )

            torch.save(data, cache_p)
            data_list.append(data)
            label_list.append(label)

        self.data_list = data_list
        self.labels    = np.array(label_list, dtype=int)
        log.info(
            "[%s] Loaded %d graphs (%d skipped). Label dist: %s",
            self.split, len(self.data_list), skipped,
            {str(k): int((self.labels == k).sum()) for k in np.unique(self.labels)},
        )

    # ------------------------------------------------------------------
    # PyG Dataset interface
    # ------------------------------------------------------------------

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]

    # ------------------------------------------------------------------
    # Concatenation
    # ------------------------------------------------------------------

    def __add__(self, other: "TwitterRumourDataset") -> "TwitterRumourDataset":
        combined = TwitterRumourDataset.__new__(TwitterRumourDataset)
        combined.split     = f"{self.split}+{other.split}"
        combined.data_list = self.data_list + other.data_list
        combined.labels    = np.concatenate([self.labels, other.labels])
        return combined


# ---------------------------------------------------------------------------
# 5-fold cross-validation splitter
# ---------------------------------------------------------------------------

def get_cv_splits(
    dataset: TwitterRumourDataset,
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[List[int], List[int], List[int]]]:
    """
    Stratified 5-fold CV following the standard Twitter15/16 benchmark protocol.

    Each fold = (train_indices, val_indices, test_indices).
    Val = 10% of the training portion of the current fold (held out before CV).

    Returns
    -------
    folds : list of (train_idx, val_idx, test_idx)  — length == n_splits
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(dataset))
    folds: List[Tuple[List[int], List[int], List[int]]] = []

    for train_val_idx, test_idx in skf.split(indices, dataset.labels):
        # Further split train_val → 90% train / 10% val (stratified)
        inner_skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        inner_labels = dataset.labels[train_val_idx]
        train_rel, val_rel = next(iter(inner_skf.split(train_val_idx, inner_labels)))
        train_idx = train_val_idx[train_rel].tolist()
        val_idx   = train_val_idx[val_rel].tolist()
        folds.append((train_idx, val_idx, test_idx.tolist()))

    return folds
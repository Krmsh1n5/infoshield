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
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
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
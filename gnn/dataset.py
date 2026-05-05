"""
gnn/dataset.py
==============
PyG dataset loader for Twitter15 / Twitter16 rumour propagation trees.

Phase 2.2 — paper-faithful BiGCN feature construction.

Per-node features (cfg.bigcn.text_embed_dim = 768 + 4 = 772)
------------------------------------------------------------
  • RoBERTa CLS embedding of root tweet text         (768) — broadcast to all nodes
    (or per-node text if a tweets file/dir is found, see _load_per_node_text)
  • depth      / 20.0                                (1)
  • log1p(in_degree)                                 (1)
  • log1p(out_degree)                                (1)
  • is_root  ∈ {0, 1}                                (1)

Graph-level features (data.graph_features, shape (1, 5))
--------------------------------------------------------
  • log1p(num_nodes)        / 8.0
  • max_depth               / 20.0
  • log1p(max_width)        / 8.0
  • branching_ratio         / 5.0   (geometric mean of per-step ratios)
  • leaf_ratio              ∈ [0, 1]

Identity invariants
-------------------
  tweet_id = tf.stem (filename) — used for label, source-text, cache, and graph identity.
  The parsed root tweet ID from the ROOT sentinel is NOT used as the node identifier.

Cache invalidation
------------------
On load, the first cached graph is inspected. If its feature dimension does not match
cfg.bigcn.text_embed_dim, or it lacks graph_features, force_reprocess is auto-enabled.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import Data, Dataset
from transformers import RobertaModel, RobertaTokenizer

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

NUM_STRUCTURAL_FEATS = 4   # depth, in_deg, out_deg, is_root
NUM_GRAPH_FEATS      = 5   # n_nodes, max_depth, max_width, branching_ratio, leaf_ratio


# ---------------------------------------------------------------------------
# Tree-file parser  (lenient — accepts quoted or unquoted IDs/delays)
# ---------------------------------------------------------------------------

_EDGE_RE = re.compile(
    r"\[\s*'?(?P<pu>[^,'\]]+?)'?\s*,\s*"
    r"'?(?P<pt>[^,'\]]+?)'?\s*,\s*"
    r"'?(?P<pd>[^,'\]]+?)'?\s*\]"
    r"\s*->\s*"
    r"\[\s*'?(?P<cu>[^,'\]]+?)'?\s*,\s*"
    r"'?(?P<ct>[^,'\]]+?)'?\s*,\s*"
    r"'?(?P<cd>[^,'\]]+?)'?\s*\]"
)


def _parse_tree_file(path: Path) -> Tuple[List[Tuple[str, str]], str]:
    """
    Parse a Twitter15/16 tree file.

    Returns
    -------
    edges          : list of (parent_tweet_id, child_tweet_id) — ROOT line stripped
    root_tweet_id  : str (taken from the ROOT sentinel line)
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
                log.debug("Unparseable line in %s: %r", path.name, line)
                continue

            parent_uid = m.group("pu").strip()
            parent_tid = m.group("pt").strip()
            child_tid  = m.group("ct").strip()

            if parent_uid == cfg.twitter15.root_sentinel:
                root_tweet_id = child_tid
            else:
                edges.append((parent_tid, child_tid))

    if root_tweet_id is None:
        raise ValueError(f"No ROOT sentinel found in {path}")

    return edges, root_tweet_id


# ---------------------------------------------------------------------------
# RoBERTa encoder (offline, frozen, batched)
# ---------------------------------------------------------------------------

class _RobertaEncoder:
    """Singleton — lazily loads RoBERTa and encodes text to 768-d CLS vectors."""

    _instance: Optional["_RobertaEncoder"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def _load(self):
        if self._loaded:
            return
        model_name: str = cfg.bigcn.text_encoder
        log.info("Loading RoBERTa: %s", model_name)
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model     = RobertaModel.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        self._loaded = True

    @torch.no_grad()
    def encode(self, text: str) -> torch.Tensor:
        """Single text → (768,) float32 tensor."""
        self._load()
        inputs = self.tokenizer(text or "", return_tensors="pt",
                                truncation=True, max_length=128, padding=False)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        out = self.model(**inputs)
        return out.last_hidden_state[:, 0, :].squeeze(0).cpu()

    @torch.no_grad()
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """List[text] → (N, 768) tensor of CLS embeddings."""
        self._load()
        chunks: List[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = [t or "" for t in texts[i:i + batch_size]]
            inputs = self.tokenizer(batch, return_tensors="pt",
                                    truncation=True, max_length=128, padding=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            out = self.model(**inputs)
            chunks.append(out.last_hidden_state[:, 0, :].cpu())
        return torch.cat(chunks, dim=0) if chunks else torch.zeros(0, 768)


_encoder = _RobertaEncoder()


# ---------------------------------------------------------------------------
# Per-node text loader (optional)
# ---------------------------------------------------------------------------

def _load_per_node_text(dataset_root: Path) -> Optional[Dict[str, str]]:
    """
    Look for per-node tweet text. Recognised layouts:

      • <root>/tweets.txt          — "tweet_id\\ttext" per line
      • <root>/all_tweets.txt      — "tweet_id\\ttext" per line
      • <root>/tweets/<tid>.txt    — one file per tweet, content = tweet text

    Returns a {tweet_id: text} dict, or None if no source is available.
    """
    flat_candidates = [dataset_root / "tweets.txt", dataset_root / "all_tweets.txt"]
    for fp in flat_candidates:
        if fp.exists():
            log.info("Found per-node text file: %s", fp)
            tweets: Dict[str, str] = {}
            with open(fp, "r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.strip().split("\t", 1)
                    if len(parts) == 2:
                        tweets[parts[0]] = parts[1]
            return tweets

    tweets_dir = dataset_root / "tweets"
    if tweets_dir.exists() and tweets_dir.is_dir():
        log.info("Found per-node text directory: %s", tweets_dir)
        tweets = {}
        for fp in tweets_dir.glob("*.txt"):
            try:
                tweets[fp.stem] = fp.read_text(encoding="utf-8").strip()
            except Exception:
                continue
        return tweets

    return None


# ---------------------------------------------------------------------------
# Per-node structural features
# ---------------------------------------------------------------------------

def _compute_node_structural(
    edges:     List[Tuple[str, str]],
    all_nodes: List[str],
    node2idx:  Dict[str, int],
    root_id:   str,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Compute per-node structural features (depth, in_deg, out_deg, is_root).

    Returns
    -------
    features : Tensor (n, 4) float32
    depth_of : dict {tweet_id: int} — used by the graph-level feature builder
    """
    n = len(all_nodes)

    in_deg  = torch.zeros(n, dtype=torch.float32)
    out_deg = torch.zeros(n, dtype=torch.float32)
    for p, c in edges:
        out_deg[node2idx[p]] += 1
        in_deg[node2idx[c]]  += 1

    is_root = torch.zeros(n, dtype=torch.float32)
    is_root[node2idx[root_id]] = 1.0

    # BFS for depth from root
    children: Dict[str, List[str]] = defaultdict(list)
    for p, c in edges:
        children[p].append(c)

    depth_arr  = torch.zeros(n, dtype=torch.float32)
    depth_of: Dict[str, int] = {}
    visited    = {root_id}
    q: deque   = deque([(root_id, 0)])
    while q:
        node, d = q.popleft()
        depth_arr[node2idx[node]] = float(d)
        depth_of[node] = d
        for ch in children[node]:
            if ch not in visited:
                visited.add(ch)
                q.append((ch, d + 1))

    feats = torch.stack([
        depth_arr / 20.0,
        torch.log1p(in_deg),
        torch.log1p(out_deg),
        is_root,
    ], dim=-1)  # (n, 4)
    return feats, depth_of


# ---------------------------------------------------------------------------
# Graph-level features (5 dims, normalised)
# ---------------------------------------------------------------------------

def _compute_graph_features(
    n_nodes:  int,
    depth_of: Dict[str, int],
    edges:    List[Tuple[str, str]],
) -> torch.Tensor:
    """
    Returns (1, 5) float32:
        log1p(n_nodes)/8, max_depth/20, log1p(max_width)/8, branching_ratio/5, leaf_ratio
    """
    if depth_of:
        depths = list(depth_of.values())
        max_depth = max(depths)
        depth_counts: Dict[int, int] = defaultdict(int)
        for d in depths:
            depth_counts[d] += 1
        max_width = max(depth_counts.values())

        sorted_depths = sorted(depth_counts.keys())
        ratios = [
            depth_counts[d + 1] / depth_counts[d]
            for d in sorted_depths
            if (d + 1) in depth_counts and depth_counts[d] > 0
        ]
        log_ratios = [np.log(r) for r in ratios if r > 0]
        branching_ratio = float(np.exp(np.mean(log_ratios))) if log_ratios else 0.0
    else:
        max_depth = 0
        max_width = 1
        branching_ratio = 0.0

    children_set = {p for p, _ in edges}
    n_leaves   = max(0, n_nodes - len(children_set))
    leaf_ratio = n_leaves / max(1, n_nodes)

    return torch.tensor([[
        float(np.log1p(n_nodes)) / 8.0,
        max_depth / 20.0,
        float(np.log1p(max_width)) / 8.0,
        min(branching_ratio, 5.0) / 5.0,
        leaf_ratio,
    ]], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Core graph builder
# ---------------------------------------------------------------------------

def _build_pyg_graph(
    tweet_id:      str,
    edges:         List[Tuple[str, str]],
    label:         int,
    root_text:     str,
    embed_cache:   Dict[str, torch.Tensor],
    per_node_text: Optional[Dict[str, str]] = None,
) -> Data:
    """
    Build a PyG Data object for one cascade tree.

    The root node is always index 0 (so batch.ptr[:-1] gives the root's
    position in a batched node tensor).
    """
    # --- collect unique nodes (root first) ---
    all_nodes: List[str] = [tweet_id]
    for p, c in edges:
        if p not in all_nodes:
            all_nodes.append(p)
        if c not in all_nodes:
            all_nodes.append(c)

    node2idx = {n: i for i, n in enumerate(all_nodes)}
    n = len(all_nodes)

    # --- text embedding(s) ---
    if tweet_id not in embed_cache:
        embed_cache[tweet_id] = _encoder.encode(root_text)
    root_emb = embed_cache[tweet_id]                                  # (768,)

    if per_node_text:
        node_texts = [per_node_text.get(node, "") for node in all_nodes]
        text_emb = torch.zeros(n, root_emb.size(0), dtype=root_emb.dtype)
        to_encode_idx, to_encode_text = [], []
        for i, t in enumerate(node_texts):
            if not t:
                text_emb[i] = root_emb                                # fallback
            elif t in embed_cache:
                text_emb[i] = embed_cache[t]
            else:
                to_encode_idx.append(i)
                to_encode_text.append(t)
        if to_encode_text:
            embs = _encoder.encode_batch(to_encode_text, batch_size=32)
            for k, idx in enumerate(to_encode_idx):
                text_emb[idx] = embs[k]
                embed_cache[to_encode_text[k]] = embs[k]
        text_feats = text_emb                                          # (n, 768)
    else:
        text_feats = root_emb.unsqueeze(0).expand(n, -1).clone()       # broadcast

    # --- structural features ---
    struct_feats, depth_of = _compute_node_structural(
        edges=edges, all_nodes=all_nodes, node2idx=node2idx, root_id=tweet_id,
    )                                                                  # (n, 4)
    x = torch.cat([text_feats, struct_feats], dim=-1)                  # (n, 772)

    # --- edges ---
    if edges:
        src = torch.tensor([node2idx[p] for p, _ in edges], dtype=torch.long)
        dst = torch.tensor([node2idx[c] for _, c in edges], dtype=torch.long)
        edge_index    = torch.stack([src, dst], dim=0)                 # parent → child  (TD)
        edge_index_bu = torch.stack([dst, src], dim=0)                 # child  → parent (BU)
    else:
        edge_index    = torch.zeros((2, 0), dtype=torch.long)
        edge_index_bu = torch.zeros((2, 0), dtype=torch.long)

    graph_feats = _compute_graph_features(n_nodes=n, depth_of=depth_of, edges=edges)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_index_bu=edge_index_bu,
        y=torch.tensor(label, dtype=torch.long),
        num_nodes=n,
        tweet_id=tweet_id,
        graph_features=graph_feats,                                    # (1, 5)
    )


# ---------------------------------------------------------------------------
# Source-tweet text + label loaders
# ---------------------------------------------------------------------------

def _load_source_texts(source_file: Path) -> Dict[str, str]:
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
            tid     = tid.strip()
            if lbl_str in label_map:
                labels[tid] = label_map[lbl_str]
            else:
                log.warning("Unknown label %r in %s", lbl_str, label_file)
    return labels


# ---------------------------------------------------------------------------
# TwitterRumourDataset
# ---------------------------------------------------------------------------

class TwitterRumourDataset(Dataset):
    """In-memory PyG dataset for Twitter15/16 with paper-faithful BiGCN features."""

    def __init__(self, split: str = "twitter15", force_reprocess: bool = False):
        super().__init__()
        assert split in ("twitter15", "twitter16"), \
            f"split must be 'twitter15' or 'twitter16', got {split!r}"

        self.split      = split
        self._ds_cfg    = cfg.twitter15 if split == "twitter15" else cfg.twitter16
        self._label_map = LABEL_MAP_15 if split == "twitter15" else LABEL_MAP_16

        self._raw_root  = Path(cfg.paths.twitter15 if split == "twitter15"
                               else cfg.paths.twitter16)
        self._tree_dir  = Path(cfg.paths.twitter15_trees if split == "twitter15"
                               else cfg.paths.twitter16_trees)
        self._cache_dir = Path(cfg.paths.graphs_pt) / split
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        self.data_list: List[Data] = []
        self.labels: np.ndarray    = np.array([], dtype=int)

        self._load(force_reprocess)

    # ------------------------------------------------------------------
    def _cache_path(self, tweet_id: str) -> Path:
        return self._cache_dir / f"{tweet_id}.pt"

    def _cache_is_stale(self) -> bool:
        """First-cache schema check: feature dim + graph_features presence."""
        sample_p = next(self._cache_dir.glob("*.pt"), None)
        if sample_p is None:
            return False
        try:
            s = torch.load(sample_p, weights_only=False)
        except Exception:
            return True
        expected_dim = int(cfg.bigcn.text_embed_dim)
        if s.x.size(-1) != expected_dim:
            log.warning("Cache feature dim %d != cfg.bigcn.text_embed_dim %d — reprocessing",
                        s.x.size(-1), expected_dim)
            return True
        if not hasattr(s, "graph_features"):
            log.warning("Cache missing graph_features — reprocessing")
            return True
        return False

    # ------------------------------------------------------------------
    def _load(self, force_reprocess: bool) -> None:
        if not force_reprocess and self._cache_is_stale():
            force_reprocess = True

        label_file  = self._raw_root / self._ds_cfg.label_file
        source_file = self._raw_root / self._ds_cfg.source_tweets_file

        labels_map  = _load_labels(label_file, self._label_map)
        source_map    = _load_source_texts(source_file)
        per_node_text = _load_per_node_text(self._raw_root) or (
        source_map if len(source_map) > len(tree_files) else None)
        if per_node_text:
            log.info("[%s] Per-node text available — %d entries",
                     self.split, len(per_node_text))
        else:
            log.info("[%s] No per-node text found — broadcasting root embedding "
                     "(structural features compensate). To enable, place a "
                     "tweets.txt or tweets/ directory in %s",
                     self.split, self._raw_root)

        tree_files = sorted(self._tree_dir.glob("*.txt"))
        if not tree_files:
            raise FileNotFoundError(
                f"No tree files found in {self._tree_dir}. "
                "Check cfg.paths.twitter15_trees / twitter16_trees."
            )
        log.info("[%s] Found %d tree files", self.split, len(tree_files))

        embed_cache: Dict[str, torch.Tensor] = {}
        skipped = 0
        data_list:  List[Data] = []
        label_list: List[int]  = []

        min_size = int(self._ds_cfg.min_tree_size)
        max_size = int(self._ds_cfg.max_tree_size)

        for tf in tree_files:
            tweet_id = tf.stem  # SINGLE source of truth for identity

            if tweet_id not in labels_map:
                skipped += 1
                continue
            label = labels_map[tweet_id]

            cache_p = self._cache_path(tweet_id)
            if cache_p.exists() and not force_reprocess:
                try:
                    data = torch.load(cache_p, weights_only=False)
                    data_list.append(data)
                    label_list.append(int(data.y.item()))
                    continue
                except Exception as e:
                    log.warning("Corrupt cache %s (%s) — rebuilding", cache_p, e)

            try:
                edges, _root_id_from_file = _parse_tree_file(tf)
                # _root_id_from_file is intentionally discarded.
                # tweet_id (from filename) is the single identity.
            except ValueError as e:
                log.warning("Skipping %s: %s", tf.name, e)
                skipped += 1
                continue

            # Original size filter (per spec)
            n_nodes = len({n for pair in edges for n in pair}) + 1
            if not (min_size <= n_nodes <= max_size):
                skipped += 1
                continue

            root_text = source_map.get(tweet_id, "")
            data = _build_pyg_graph(
                tweet_id=tweet_id,
                edges=edges,
                label=label,
                root_text=root_text,
                embed_cache=embed_cache,
                per_node_text=per_node_text,
            )
            torch.save(data, cache_p)
            data_list.append(data)
            label_list.append(label)

        self.data_list = data_list
        self.labels    = np.array(label_list, dtype=int)

        log.info(
            "[%s] Loaded %d graphs, %d skipped. Label dist: %s",
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

    def __add__(self, other: "TwitterRumourDataset") -> "TwitterRumourDataset":
        combined = TwitterRumourDataset.__new__(TwitterRumourDataset)
        combined.split     = f"{self.split}+{other.split}"
        combined.data_list = self.data_list + other.data_list
        combined.labels    = np.concatenate([self.labels, other.labels])
        return combined


# ---------------------------------------------------------------------------
# 5-fold stratified cross-validation
# ---------------------------------------------------------------------------

def get_cv_splits(
    dataset: TwitterRumourDataset,
    n_splits: int = 5,
    seed: int = 42,
) -> List[Tuple[List[int], List[int], List[int]]]:
    """
    Stratified k-fold CV. Each fold = (train_idx, val_idx, test_idx).
    Val = held-out 10% of the train portion of the current fold.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    indices = np.arange(len(dataset))
    folds: List[Tuple[List[int], List[int], List[int]]] = []

    for train_val_idx, test_idx in skf.split(indices, dataset.labels):
        inner_skf    = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
        inner_labels = dataset.labels[train_val_idx]
        train_rel, val_rel = next(iter(inner_skf.split(train_val_idx, inner_labels)))
        train_idx = train_val_idx[train_rel].tolist()
        val_idx   = train_val_idx[val_rel].tolist()
        folds.append((train_idx, val_idx, test_idx.tolist()))

    return folds
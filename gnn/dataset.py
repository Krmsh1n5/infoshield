"""
Twitter15/16/WICO/Weibo PyG dataset for BiGCN.

Important Twitter15/16 design note
----------------------------------
The Twitter tree files contain propagation structure, but only the source
tweet text is available locally through source_tweets.txt. Reply tweet text is
not available. Therefore the RoBERTa CLS embedding of the source tweet is
broadcast to every node and concatenated with structural features.

Each node feature is:
    [root_tweet_roberta_cls_768 | is_root | norm_depth | norm_out_degree | norm_descendants]

So each node has 772 features when using roberta-base.
"""

from __future__ import annotations

import logging
import re
import sys
from collections import Counter, deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset as TorchDataset
from torch_geometric.data import Data

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg


log = logging.getLogger(__name__)

TEXT_EMBED_DIM = 768
STRUCT_FEATURE_DIM = 4

EDGE_PATTERN = re.compile(
    r"\['([^']*)',\s*'([^']*)',\s*'([^']*)'\]\s*->\s*"
    r"\['([^']*)',\s*'([^']*)',\s*'([^']*)'\]"
)


class CascadeTree:
    """
    Propagation tree for one source tweet.

    root_id:
        Source tweet id. Used for label lookup, text embedding lookup, and
        evaluation CSV output.

    root_uid:
        Source-user node id inside the propagation tree. This is the actual
        graph root. It is parsed from the ROOT -> source_user line.

    children:
        Directed top-down propagation edges: parent_uid -> child_uid.
    """

    def __init__(self, root_id: str, label: str) -> None:
        self.root_id = str(root_id)
        self.label = str(label).lower()
        self.root_uid: Optional[str] = None
        self.children: Dict[str, List[str]] = {}
        self.nodes: set[str] = set()

    @property
    def graph_root(self) -> Optional[str]:
        return self.root_uid

    def set_root_uid(self, uid: str) -> None:
        uid = str(uid)
        self.root_uid = uid
        self.nodes.add(uid)

    def add_edge(self, parent: str, child: str) -> None:
        parent = str(parent)
        child = str(child)

        if self.root_uid is None:
            self.root_uid = parent

        self.nodes.add(parent)
        self.nodes.add(child)
        self.children.setdefault(parent, []).append(child)


class Twitter15Dataset:
    """
    Lightweight parser for Twitter15/16 tree files.

    The source tweet id is the tree filename stem. The graph root is the user id
    found on the child side of the ROOT sentinel edge.
    """

    def __init__(self, tree_dir: Path, label_file: Path, root_sentinel: str) -> None:
        self.tree_dir = Path(tree_dir)
        self.label_file = Path(label_file)
        self.root_sentinel = root_sentinel
        self.trees: List[CascadeTree] = []
        self._parse()

    def _parse(self) -> None:
        raw_labels = self._load_raw_labels(self.label_file)

        for tree_path in sorted(self.tree_dir.glob("*.txt")):
            tweet_id = tree_path.stem
            label = raw_labels.get(tweet_id)
            if label is None:
                continue

            tree = CascadeTree(root_id=tweet_id, label=label)

            try:
                with open(tree_path, encoding="utf-8", errors="replace") as fh:
                    for line in fh:
                        m = EDGE_PATTERN.match(line.strip())
                        if not m:
                            continue

                        parent_uid, _parent_tid, _parent_delay, child_uid, _child_tid, _child_delay = m.groups()

                        if parent_uid == self.root_sentinel:
                            tree.set_root_uid(child_uid)
                            continue

                        tree.add_edge(parent_uid, child_uid)
            except OSError as exc:
                log.warning("Could not read tree file %s: %s", tree_path, exc)
                continue

            self.trees.append(tree)

    @staticmethod
    def _load_raw_labels(label_file: Path) -> Dict[str, str]:
        labels: Dict[str, str] = {}

        with open(label_file, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or ":" not in line:
                    continue

                label, tweet_id = line.split(":", 1)
                labels[tweet_id.strip()] = label.strip().lower()

        return labels


class TwitterRumourDataset(TorchDataset):
    """
    PyTorch/PyG-compatible dataset used by train.py and evaluate.py.

    Supports:
        twitter15, twitter16, wico, weibo
    """

    _LABEL_MAPS: Dict[str, Dict[str, int]] = {
        "twitter15": {
            "true": 0,
            "true rumour": 0,
            "true rumor": 0,
            "false": 1,
            "false rumour": 1,
            "false rumor": 1,
            "unverified": 2,
            "non-rumour": 3,
            "non-rumor": 3,
        },
        "twitter16": {
            "true": 0,
            "true rumour": 0,
            "true rumor": 0,
            "false": 1,
            "false rumour": 1,
            "false rumor": 1,
            "unverified": 2,
            "non-rumour": 3,
            "non-rumor": 3,
        },
        "wico": {
            "5g-conspiracy": 0,
            "5g_conspiracy": 0,
            "conspiracy": 1,
            "other-conspiracy": 1,
            "other_conspiracy": 1,
            "non-conspiracy": 2,
            "non_conspiracy": 2,
        },
        "weibo": {
            "rumour": 0,
            "rumor": 0,
            "rumours": 0,
            "rumors": 0,
            "0": 0,
            "non-rumour": 1,
            "non-rumor": 1,
            "non-rumours": 1,
            "non-rumors": 1,
            "1": 1,
        },
    }

    def __init__(self, split: str) -> None:
        self.split = split.lower()
        self._data: List[Data] = []
        self._load()

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Data:
        return self._data[idx]

    def get(self, idx: int) -> Data:
        return self._data[idx]

    def __repr__(self) -> str:
        return f"TwitterRumourDataset(split={self.split!r}, n={len(self)})"

    def get_kfold_splits(self, n_splits: int = 5, seed: int = 42):
        labels = np.array([int(graph.y.item()) for graph in self._data])
        all_idx = np.arange(len(self._data))

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = []

        for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(all_idx, labels)):
            rng = np.random.default_rng(seed + fold_idx)
            train_val_idx = np.array(train_val_idx, copy=True)
            rng.shuffle(train_val_idx)

            n_val = max(1, int(0.1 * len(train_val_idx)))
            val_idx = train_val_idx[:n_val]
            train_idx = train_val_idx[n_val:]

            folds.append((train_idx, val_idx, test_idx))

        return folds

    def _load(self) -> None:
        trees, texts = self._load_trees_and_texts()
        embeddings = self._get_embeddings(texts)

        label_map = self._LABEL_MAPS.get(self.split)
        if label_map is None:
            raise ValueError(
                f"Unknown split {self.split!r}. Choose from: "
                f"{', '.join(sorted(self._LABEL_MAPS))}"
            )

        skipped = 0
        for tree in trees:
            label = label_map.get(tree.label)
            if label is None:
                skipped += 1
                continue

            root_emb = embeddings.get(tree.root_id)
            if root_emb is None:
                root_emb = torch.zeros(TEXT_EMBED_DIM, dtype=torch.float32)

            data = self._build_pyg_graph(tree, root_emb, label)
            if data is None:
                skipped += 1
                continue

            self._data.append(data)

        dist = Counter(int(graph.y.item()) for graph in self._data)
        log.info(
            "TwitterRumourDataset(%s): loaded=%d  skipped=%d  label_dist=%s",
            self.split,
            len(self._data),
            skipped,
            dict(sorted(dist.items())),
        )

    def _load_trees_and_texts(self) -> Tuple[List[CascadeTree], Dict[str, str]]:
        if self.split in ("twitter15", "twitter16"):
            return self._load_twitter()
        if self.split == "wico":
            return self._load_wico()
        if self.split == "weibo":
            return self._load_weibo()

        raise ValueError(
            f"Unknown split {self.split!r}. Choose from: twitter15, twitter16, wico, weibo"
        )

    def _load_twitter(self) -> Tuple[List[CascadeTree], Dict[str, str]]:
        if self.split == "twitter15":
            root_dir = Path(cfg.paths.twitter15)
            tcfg = cfg.twitter15
        else:
            root_dir = Path(cfg.paths.twitter16)
            tcfg = cfg.twitter16

        tree_dir = root_dir / tcfg.tree_dir
        label_file = root_dir / tcfg.label_file

        parser = Twitter15Dataset(
            tree_dir=tree_dir,
            label_file=label_file,
            root_sentinel=tcfg.root_sentinel,
        )

        texts: Dict[str, str] = {}
        source_file = root_dir / tcfg.source_tweets_file

        if source_file.exists():
            with open(source_file, encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    parts = line.rstrip("\n").split("\t", 1)
                    if len(parts) == 2:
                        texts[parts[0].strip()] = parts[1].strip()
        else:
            log.warning("source_tweets.txt not found at %s; using zero embeddings.", source_file)

        return parser.trees, texts

    def _load_wico(self) -> Tuple[List[CascadeTree], Dict[str, str]]:
        label_by_folder = {
            "5G_Conspiracy_Graphs": "5g-conspiracy",
            "Other_Graphs": "conspiracy",
            "Non_Conspiracy_Graphs": "non-conspiracy",
        }

        trees: List[CascadeTree] = []
        wico_graph_root = Path(cfg.paths.wico_graph)

        for folder_name, label in label_by_folder.items():
            folder = wico_graph_root / folder_name
            if not folder.exists():
                log.warning("WICO folder not found: %s", folder)
                continue

            for tweet_dir in sorted(folder.iterdir()):
                edges_file = tweet_dir / cfg.wico.graph_edges_file
                if not edges_file.exists():
                    continue

                tree = CascadeTree(root_id=tweet_dir.name, label=label)

                try:
                    with open(edges_file, encoding="utf-8", errors="replace") as fh:
                        for line in fh:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                tree.add_edge(parts[0], parts[1])
                except OSError as exc:
                    log.warning("Could not read WICO edges file %s: %s", edges_file, exc)
                    continue

                trees.append(tree)

        return trees, {}

    def _load_weibo(self) -> Tuple[List[CascadeTree], Dict[str, str]]:
        if not hasattr(cfg.paths, "weibo"):
            raise ValueError("cfg.paths.weibo is not defined, so split='weibo' cannot be loaded.")

        from gnn.weibo_dataset import WeiboDataset

        weibo_root = Path(cfg.paths.weibo)
        tree_name = getattr(cfg.weibo, "tree_file", getattr(cfg.weibo, "tree_dir", "tree"))
        label_name = getattr(cfg.weibo, "label_file", "label.txt")

        ds = WeiboDataset(weibo_root / tree_name, weibo_root / label_name)

        trees: List[CascadeTree] = []
        for original in ds.trees:
            tree = CascadeTree(root_id=original.root_id, label=original.label)
            original_children = getattr(original, "_children", getattr(original, "children", {}))
            for parent, children in original_children.items():
                for child in children:
                    tree.add_edge(parent, child)
            trees.append(tree)

        return trees, {}

    def _get_embeddings(self, texts: Dict[str, str]) -> Dict[str, torch.Tensor]:
        cache_path = Path(cfg.paths.graphs_pt) / f"{self.split}_roberta_emb.pt"

        if cache_path.exists():
            log.info("Loading cached embeddings from %s", cache_path)
            embeddings = torch.load(cache_path, map_location="cpu")

            # Keep a light validation so stale/corrupt caches fail early.
            if embeddings:
                first = next(iter(embeddings.values()))
                if int(first.numel()) != TEXT_EMBED_DIM:
                    raise ValueError(
                        f"Embedding cache at {cache_path} has dim {first.numel()}, "
                        f"expected {TEXT_EMBED_DIM}. Delete the cache and rerun."
                    )

            return embeddings

        if not texts:
            return {}

        log.info(
            "Computing RoBERTa embeddings for %d source tweets (split=%s)",
            len(texts),
            self.split,
        )

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            log.warning("transformers is not installed; using zero embeddings.")
            return {}

        tokenizer = AutoTokenizer.from_pretrained(cfg.bigcn.text_encoder)
        model = AutoModel.from_pretrained(cfg.bigcn.text_encoder)
        model.eval()

        embeddings: Dict[str, torch.Tensor] = {}
        tweet_ids = list(texts.keys())
        batch_size = 32

        with torch.no_grad():
            for start in range(0, len(tweet_ids), batch_size):
                batch_ids = tweet_ids[start:start + batch_size]
                batch_texts = [texts[tweet_id] or "[empty]" for tweet_id in batch_ids]

                encoded = tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128,
                    padding=True,
                )
                output = model(**encoded)
                cls = output.last_hidden_state[:, 0, :].cpu().float()

                for tweet_id, emb in zip(batch_ids, cls):
                    embeddings[tweet_id] = emb

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, cache_path)
        log.info("Saved embeddings to %s", cache_path)

        return embeddings

    def _build_pyg_graph(
        self,
        tree: CascadeTree,
        root_emb: torch.Tensor,
        label: int,
    ) -> Optional[Data]:
        graph_root = tree.graph_root
        if graph_root is None:
            return None

        n_raw = len(tree.nodes)
        min_size, max_size = self._tree_size_limits()
        if n_raw < min_size or n_raw > max_size:
            return None

        nodes_ordered = self._bfs_order(tree)
        if not nodes_ordered:
            return None

        num_nodes = len(nodes_ordered)
        node_to_idx = {node_id: i for i, node_id in enumerate(nodes_ordered)}

        src_td: List[int] = []
        dst_td: List[int] = []

        for parent, children in tree.children.items():
            parent_idx = node_to_idx.get(parent)
            if parent_idx is None:
                continue

            for child in children:
                child_idx = node_to_idx.get(child)
                if child_idx is None:
                    continue
                src_td.append(parent_idx)
                dst_td.append(child_idx)

        if not src_td:
            return None

        depth_arr, out_degree_arr, descendant_arr = self._structural_features(
            num_nodes,
            node_to_idx,
            tree.children,
        )

        max_depth = float(depth_arr.max()) or 1.0
        max_out_degree = float(out_degree_arr.max()) or 1.0

        root_emb = root_emb.detach().float().view(-1)
        if root_emb.numel() != TEXT_EMBED_DIM:
            raise ValueError(
                f"Expected root embedding dim {TEXT_EMBED_DIM}, got {root_emb.numel()} "
                f"for tweet_id={tree.root_id}"
            )

        root_broadcast = root_emb.unsqueeze(0).expand(num_nodes, -1)

        struct = torch.zeros(num_nodes, STRUCT_FEATURE_DIM, dtype=torch.float32)
        struct[0, 0] = 1.0
        struct[:, 1] = torch.from_numpy(depth_arr / max_depth).float()
        struct[:, 2] = torch.from_numpy(out_degree_arr / max_out_degree).float()
        struct[:, 3] = torch.from_numpy(descendant_arr / max(num_nodes - 1, 1)).float()

        x = torch.cat([root_broadcast, struct], dim=1)

        edge_index = torch.tensor([src_td, dst_td], dtype=torch.long)
        edge_index_bu = torch.tensor([dst_td, src_td], dtype=torch.long)

        root_mask = torch.zeros(num_nodes, dtype=torch.bool)
        root_mask[0] = True

        return Data(
            x=x,
            edge_index=edge_index,
            edge_index_bu=edge_index_bu,
            root_mask=root_mask,
            y=torch.tensor(label, dtype=torch.long),
            num_nodes=num_nodes,
            tweet_id=tree.root_id,
        )

    def _tree_size_limits(self) -> Tuple[int, int]:
        if self.split == "twitter16":
            return int(cfg.twitter16.min_tree_size), int(cfg.twitter16.max_tree_size)
        if self.split == "weibo":
            return int(cfg.weibo.min_tree_size), int(cfg.weibo.max_tree_size)

        # Twitter15 defaults are also a good guard for WICO in this project.
        return int(cfg.twitter15.min_tree_size), int(cfg.twitter15.max_tree_size)

    @staticmethod
    def _bfs_order(tree: CascadeTree) -> List[str]:
        root = tree.graph_root
        if root is None:
            return []

        visited = {root}
        order = [root]
        queue: deque[str] = deque([root])

        while queue:
            node = queue.popleft()
            for child in tree.children.get(node, []):
                if child in visited:
                    continue
                visited.add(child)
                order.append(child)
                queue.append(child)

        # Deterministic fallback for disconnected nodes.
        for node in sorted(tree.nodes, key=str):
            if node not in visited:
                visited.add(node)
                order.append(node)

        return order

    @staticmethod
    def _structural_features(
        num_nodes: int,
        node_to_idx: Dict[str, int],
        children: Dict[str, List[str]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        kids_by_idx: Dict[int, List[int]] = {}

        for parent, child_nodes in children.items():
            parent_idx = node_to_idx.get(parent)
            if parent_idx is None:
                continue
            kids_by_idx[parent_idx] = [
                node_to_idx[child]
                for child in child_nodes
                if child in node_to_idx
            ]

        depth_arr = np.zeros(num_nodes, dtype=np.float32)
        visited = np.zeros(num_nodes, dtype=bool)
        queue: deque[int] = deque([0])
        visited[0] = True

        while queue:
            node = queue.popleft()
            for child in kids_by_idx.get(node, []):
                if visited[child]:
                    continue
                visited[child] = True
                depth_arr[child] = depth_arr[node] + 1
                queue.append(child)

        out_degree_arr = np.array(
            [len(kids_by_idx.get(i, [])) for i in range(num_nodes)],
            dtype=np.float32,
        )

        descendant_arr = np.zeros(num_nodes, dtype=np.float32)
        post_order = list(range(num_nodes))
        post_order.sort(key=lambda idx: -depth_arr[idx])

        for node in post_order:
            for child in kids_by_idx.get(node, []):
                descendant_arr[node] += 1.0 + descendant_arr[child]

        return depth_arr, out_degree_arr, descendant_arr

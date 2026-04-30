"""
InfoGuard — Central Configuration
===================================
Single source of truth for all paths, dataset settings, model hyperparameters,
and algorithm constants. Import this in every module instead of hard-coding values.

Usage:
    from config import cfg
    print(cfg.DATA_RAW / "wico")
"""

from pathlib import Path
from dataclasses import dataclass, field


# ── Project root (the folder this file lives in) ─────────────────────────────
ROOT = Path(__file__).parent.resolve()


@dataclass
class Paths:
    """All filesystem paths used across the project."""
    # Raw downloaded datasets — never modified
    data_raw:           Path = ROOT / "data" / "raw"
    wico_text:          Path = ROOT / "data" / "raw" / "wico-text"
    wico_graph:         Path = ROOT / "data" / "raw" / "wico-graph"
    twitter15:          Path = ROOT / "data" / "raw" / "twitter15"
    twitter16:          Path = ROOT / "data" / "raw" / "twitter16"
    # Convenience sub-path shortcuts for tree files
    twitter15_trees:    Path = ROOT / "data" / "raw" / "twitter15" / "tree"
    twitter16_trees:    Path = ROOT / "data" / "raw" / "twitter16" / "tree"

    # Processed — PyG Data objects, fitted SBM matrices, etc.
    data_processed:     Path = ROOT / "data" / "processed"
    graphs_pt:          Path = ROOT / "data" / "processed" / "graphs"
    sbm_matrices:       Path = ROOT / "data" / "processed" / "sbm_matrices"

    # Model checkpoints
    checkpoints:        Path = ROOT / "checkpoints"
    bigcn_checkpoint:   Path = ROOT / "checkpoints" / "bigcn_best.pt"

    # Notebooks, results
    evaluation:         Path = ROOT / "evaluation"

    def make_all(self):
        """Create all directories if they don't exist."""
        for f in self.__dataclass_fields__:
            path = getattr(self, f)
            path.mkdir(parents=True, exist_ok=True)
        print("✓ All project directories created.")


@dataclass
class WICOConfig:
    """
    WICO dataset settings.

    WICO has two variants — use BOTH in the project:
      - WICO Text : tweet text + labels (5G-conspiracy / other-conspiracy / non-conspiracy)
      - WICO Graph: Twitter subgraphs showing follower connections around each tweet

    Download URLs (confirmed from Simula Research Laboratory):
      Graph: https://datasets.simula.no/wico-graph
      Text:  https://datasets.simula.no/wico-text
             (also available via HuggingFace Hub — see below)

    HuggingFace Hub (easiest for text variant):
      from datasets import load_dataset
      ds = load_dataset("Schroeder2021/wico-text")

    WICO Text structure:
        wico-text/
            5g_corona_conspiracy.txt     ← one tweet ID (int) per line, label=0
            other_conspiracy.txt         ← one tweet ID (int) per line, label=1
            non_conspiracy.txt           ← one tweet ID (int) per line, label=2

        Example line:  1123359858106032130

    WICO Graph structure:
        wico-graph/
            5G_Conspiracy_Graphs/
                <tweet_id>/
                    edges.txt   ← one edge per line: "source_id target_id"  (space-separated ints)
                    nodes.csv   ← CSV with header: id,time,friends,followers
                    plot.png    ← visualisation (not used in training)
            Other_Graphs/
                <tweet_id>/  ...
            Non_Conspiracy_Graphs/
                <tweet_id>/  ...

        Example edges.txt line:  59596605 155604404
        Example nodes.csv rows:
            id,time,friends,followers
            59596605,106295,10,8
            155604404,34633,10,5
    """
    # Label mapping used by the paper and this project.
    # Keys match the WICO dataset category names exactly
    # (5G_Conspiracy_Graphs / Other_Graphs / Non_Conspiracy_Graphs).
    label_map: dict = field(default_factory=lambda: {
        "5g-conspiracy":  0,   # False content  → feeds G- model
        "conspiracy":     1,   # False content  → feeds G- model
        "non-conspiracy": 2,   # True content   → feeds G+ model
    })
    # Re-mapped binary labels (used for SBM fitting)
    binary_label_map: dict = field(default_factory=lambda: {
        0: "false",  # 5g-conspiracy
        1: "false",  # other conspiracy
        2: "true",   # non-conspiracy
    })

    # ── WICO Text: one .txt file per class ───────────────────────────────────
    # Each file contains one tweet ID (bare integer) per line — no header.
    text_files: dict = field(default_factory=lambda: {
        0: "5g_corona_conspiracy.txt",
        1: "other_conspiracy.txt",
        2: "non_conspiracy.txt",
    })

    # ── WICO Graph: one sub-directory per class ───────────────────────────────
    # Each sub-directory holds per-tweet folders named by their tweet ID.
    graph_dirs: dict = field(default_factory=lambda: {
        0: "5G_Conspiracy_Graphs",
        1: "Other_Graphs",
        2: "Non_Conspiracy_Graphs",
    })

    # edges.txt — two space-separated integer node IDs per line, no header
    #   format:  <source_id> <target_id>
    graph_edges_file: str = "edges.txt"

    # nodes.csv — CSV with header row, four columns:
    #   id        : int  — node / user ID
    #   time      : int  — account age proxy (minutes or days, dataset-native)
    #   friends   : int  — number of accounts this user follows
    #   followers : int  — number of accounts following this user
    graph_nodes_file:    str  = "nodes.csv"
    graph_nodes_columns: tuple = ("id", "time", "friends", "followers")


@dataclass
class Twitter15Config:
    """
    Twitter15 dataset settings.
    Source: https://github.com/TianBian95/BiGCN (includes the data/)

    Directory structure:
        twitter15/
            tree/
                <tweet_id>.txt      ← propagation tree, one edge per line
            label.txt               ← "label:source_tweet_id" per line (4-class)
            source_tweets.txt       ← "source_tweet_id \\t tweet_text" per line

    ── tree/<tweet_id>.txt ──────────────────────────────────────────────────
    Each line encodes a directed edge as two Python-list-style tuples:
        ['parent_uid', 'parent_tweet_id', 'delay_min'] -> ['child_uid', 'child_tweet_id', 'delay_min']

    Special root sentinel — the source tweet has no real parent:
        ['ROOT', 'ROOT', '0.0'] -> ['uid', 'source_tweet_id', '0.0']

    All delay values are floats (minutes since source tweet).

    Example (twitter15/tree/1.txt):
        ['ROOT', 'ROOT', '0.0']->['972651', '80080680482123777', '0.0']
        ['972651', '80080680482123777', '0.0']->['189397006', '80080680482123777', '1.8']
        ['972651', '80080680482123777', '0.0']->['10678072', '80080680482123777', '1.8']

    ── label.txt ────────────────────────────────────────────────────────────
    One entry per source tweet, colon-separated, no header:
        unverified:731166399389962242

    ── source_tweets.txt ────────────────────────────────────────────────────
    One entry per source tweet, tab-separated (tweet_id \\t text), no header:
        731166399389962242\\t🔥ca kkk grand wizard 🔥 endorses @hillaryclinton #neverhillary URL

    Note: only source tweet text is distributed (Twitter ToS); all other tweet
    content must be hydrated via the Twitter API using the supplied tweet IDs.

    Labels: true / false / unverified / non-rumor
    """
    label_map: dict = field(default_factory=lambda: {
        "true":       0,
        "false":      1,
        "unverified": 2,
        "non-rumor":  3,
    })
    binary_label_map: dict = field(default_factory=lambda: {
        0: "true",
        1: "false",
        2: "uncertain",
        3: "true",   # non-rumor treated as verified true
    })

    # Sub-directory and file names inside the dataset root
    tree_dir:           str = "tree"              # folder of per-tweet .txt files
    label_file:         str = "label.txt"         # "label:tweet_id" per line
    source_tweets_file: str = "source_tweets.txt" # "tweet_id \t text" per line

    # Sentinel string used in tree files to mark the root node
    root_sentinel: str = "ROOT"

    min_tree_size:  int = 3    # skip trees with fewer than 3 nodes (too sparse)
    max_tree_size:  int = 2000 # skip abnormally large trees


@dataclass
class Twitter16Config:
    """
    Twitter16 dataset settings.
    Identical structure to Twitter15 — see Twitter15Config for full format details.
    Source: https://github.com/TianBian95/BiGCN (includes the data/)

    Directory structure:
        twitter16/
            tree/
                <tweet_id>.txt      ← same edge format as Twitter15
            label.txt               ← "label:source_tweet_id" per line
            source_tweets.txt       ← "source_tweet_id \\t tweet_text" per line

    Labels: true / false / unverified / non-rumor
    """
    label_map: dict = field(default_factory=lambda: {
        "true":       0,
        "false":      1,
        "unverified": 2,
        "non-rumor":  3,
    })
    binary_label_map: dict = field(default_factory=lambda: {
        0: "true",
        1: "false",
        2: "uncertain",
        3: "true",   # non-rumor treated as verified true
    })

    # Sub-directory and file names inside the dataset root
    tree_dir:           str = "tree"
    label_file:         str = "label.txt"
    source_tweets_file: str = "source_tweets.txt"

    # Sentinel string used in tree files to mark the root node
    root_sentinel: str = "ROOT"

    min_tree_size:  int = 3
    max_tree_size:  int = 2000


@dataclass
class BiGCNConfig:
    """
    Hyperparameters for the BiGCN model (Phase 2).
    Based on the original BiGCN paper defaults with minor adjustments.
    """
    # Text encoder
    text_encoder:       str   = "roberta-base"  # HuggingFace model name
    text_embed_dim:     int   = 768             # RoBERTa hidden size
    freeze_encoder:     bool  = True            # Freeze RoBERTa weights during Phase 2

    # GCN layers (applied to both top-down and bottom-up trees)
    gcn_hidden_dim:     int   = 256
    gcn_output_dim:     int   = 128
    gcn_num_layers:     int   = 2
    dropout:            float = 0.3

    # Training
    num_epochs:         int   = 200
    batch_size:         int   = 32
    learning_rate:      float = 5e-4
    weight_decay:       float = 1e-4
    patience:           int   = 20             # Early stopping

    # Output
    num_classes:        int   = 4              # Twitter15 has 4 labels
    binary_mode:        bool  = False          # If True → only true/false output


@dataclass
class SBMConfig:
    """
    Stochastic Block Model settings (Phase 3 — graph engine).
    These match the paper's notation: b+_uv and b-_uv matrices.
    """
    # How many polarization classes to fit from the data
    # 13 partitions for WICO (from modularity clustering)
    num_partitions:         int   = 13

    # Modularity clustering resolution parameter (higher = more, smaller partitions)
    clustering_resolution:  float = 2.0

    # Minimum fraction of total users for a partition to be kept
    # Partitions below this threshold are merged into the nearest class
    # The paper merged all partitions with < 1% of users
    min_partition_fraction: float = 0.01

    # Confidence threshold from GNN output before a label feeds the SBM fitter
    # Labels below this are discarded (treated as "uncertain")
    label_confidence_threshold: float = 0.65


@dataclass
class LPOptimizerConfig:
    """
    Linear Program optimizer settings
    """
    # Safety parameter α: minimum branching ratio preserved for true content
    # α ∈ {1.5, 2.0, 3.0} — start with 1.5
    alpha: float = 1.5

    # Weight λ: importance of preserving true content vs suppressing false content
    # Used in the softened LP when the hard LP is infeasible
    lambda_weight: float = 1.0

    # Dropout bounds: d_uv ∈ [dropout_min, 1.0]
    # Setting dropout_min > 0 prevents total suppression of any connection
    dropout_min:  float = 0.0
    dropout_max:  float = 1.0


@dataclass
class Config:
    """Master config — import this single object everywhere."""
    paths:      Paths             = field(default_factory=Paths)
    wico:       WICOConfig        = field(default_factory=WICOConfig)
    twitter15:  Twitter15Config   = field(default_factory=Twitter15Config)
    twitter16:  Twitter16Config   = field(default_factory=Twitter16Config)
    bigcn:      BiGCNConfig       = field(default_factory=BiGCNConfig)
    sbm:        SBMConfig         = field(default_factory=SBMConfig)
    lp:         LPOptimizerConfig = field(default_factory=LPOptimizerConfig)

    # Global seed for reproducibility
    seed: int = 42

    def __post_init__(self):
        import torch, numpy as np, random
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)


# Singleton — import this in all modules
cfg = Config()


# ── Quick setup helper ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("InfoGuard Configuration")
    print("=" * 40)
    print(f"Project root : {ROOT}")
    print(f"WICO path    : {cfg.paths.wico_text} (text), {cfg.paths.wico_graph} (graph)")
    print(f"BiGCN model  : {cfg.bigcn.text_encoder}")
    print(f"LP alpha     : {cfg.lp.alpha}")
    print(f"LP lambda    : {cfg.lp.lambda_weight}")
    print(f"SBM parts    : {cfg.sbm.num_partitions}")
    print()
    cfg.paths.make_all()
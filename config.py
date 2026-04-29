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
    """
    # Label mapping used by the paper and this project
    label_map: dict = field(default_factory=lambda: {
        "5g-corona":    0,   # False content  → feeds G- model
        "conspiracy":   1,   # False content  → feeds G- model
        "non-conspiracy": 2, # True content   → feeds G+ model
    })
    # Re-mapped binary labels (used for SBM fitting)
    binary_label_map: dict = field(default_factory=lambda: {
        0: "false",  # 5g-corona conspiracy
        1: "false",  # other conspiracy
        2: "true",   # non-conspiracy
    })
    # Graph format: adjacency list .csv with columns: source_id, target_id, tweet_id
    graph_file:     str = "wico_graph_edges.csv"
    label_file:     str = "wico_graph_labels.csv"
    text_file:      str = "wico_text.csv"


@dataclass
class Twitter15Config:
    """
    Twitter15 dataset settings.
    Source: https://github.com/TianBian95/BiGCN (includes the data/)

    Structure per event:
        <tweet_id>.txt  →  each line: "parent_id  user_id  timestamp  text"
        label.txt       →  "<tweet_id>  <label>"   (4-class)

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
        3: "true",
    })
    min_tree_size:  int = 3    # Skip trees with fewer than 3 nodes (too sparse)
    max_tree_size:  int = 2000 # Skip abnormally large trees


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
    paths:      Paths           = field(default_factory=Paths)
    wico:       WICOConfig      = field(default_factory=WICOConfig)
    twitter15:  Twitter15Config = field(default_factory=Twitter15Config)
    bigcn:      BiGCNConfig     = field(default_factory=BiGCNConfig)
    sbm:        SBMConfig       = field(default_factory=SBMConfig)
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
    print(f"WICO path    : {cfg.paths.wico}")
    print(f"BiGCN model  : {cfg.bigcn.text_encoder}")
    print(f"LP alpha     : {cfg.lp.alpha}")
    print(f"LP lambda    : {cfg.lp.lambda_weight}")
    print(f"SBM parts    : {cfg.sbm.num_partitions}")
    print()
    cfg.paths.make_all()
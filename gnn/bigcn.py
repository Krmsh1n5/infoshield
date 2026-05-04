"""
gnn/bigcn.py
============
Bi-directional Graph Convolutional Network (BiGCN) for rumour detection.

Reference
---------
Bian et al. (2020). "Rumor Detection on Social Media with Bi-Directional
Graph Convolutional Networks." AAAI-20.

Architecture
------------
                      ┌──────────────┐
    TD edge_index ──▶ │  TD-GCN (↓)  │──▶ z_td  (gcn_output_dim)
                      └──────────────┘         \
    node features x                             cat ──▶ FC ──▶ logits
                      ┌──────────────┐         /
    BU edge_index ──▶ │  BU-GCN (↑)  │──▶ z_bu  (gcn_output_dim)
                      └──────────────┘

Both GCN branches share the same hyperparameters but have independent weights.
Node features x: RoBERTa CLS embedding of root tweet, broadcast to all nodes.
Readout: mean pooling over all nodes in each branch.

Note on RoBERTa
---------------
RoBERTa is NOT inside the forward pass. Embeddings are pre-computed once
during dataset preprocessing (cfg.bigcn.freeze_encoder = True) and stored
as node features in the PyG Data objects. BiGCN only handles the GCN layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg


class _GCNBranch(nn.Module):
    """
    A stack of GCNConv layers with ReLU + Dropout.

    Input  : (x: Tensor[N, in_dim], edge_index: LongTensor[2, E])
    Output : (z: Tensor[N, out_dim])
    """

    def __init__(
        self,
        in_dim:     int,
        hidden_dim: int,
        out_dim:    int,
        num_layers: int,
        dropout:    float,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be ≥ 1"

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.convs    = nn.ModuleList(
            [GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)]
        )
        self.dropout  = dropout
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.num_layers - 1:          # no activation after last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x                                  # (N, out_dim)


class BiGCN(nn.Module):
    """
    Bi-directional GCN rumour classifier.

    Parameters (all drawn from cfg.bigcn)
    --------------------------------------
    text_embed_dim  : input feature dimension (768 for roberta-base)
    gcn_hidden_dim  : hidden dimension in each GCN branch  (256)
    gcn_output_dim  : output dimension of each GCN branch  (128)
    gcn_num_layers  : number of GCN layers per branch       (2)
    dropout         : dropout probability                   (0.3)
    num_classes     : number of output classes              (4)

    Forward input
    -------------
    data : PyG Data or Batch
        data.x           — node features  (N_total, 768)
        data.edge_index  — top-down edges (2, E_td)
        data.edge_index_bu — bottom-up edges (2, E_bu)
        data.batch       — batch assignment vector (N_total,) — None for single graph

    Forward output
    --------------
    logits : Tensor (B, num_classes) where B = batch size
    """

    def __init__(self):
        super().__init__()

        # +4 for structural node features (depth, in_degree, out_degree, is_root)
        # appended in dataset._build_pyg_graph — actual input dim is 772
        in_dim     = cfg.bigcn.text_embed_dim + 4    # 768 + 4 = 772
        hidden_dim = cfg.bigcn.gcn_hidden_dim    # 256
        out_dim    = cfg.bigcn.gcn_output_dim    # 128
        num_layers = cfg.bigcn.gcn_num_layers    # 2
        dropout    = cfg.bigcn.dropout           # 0.3
        num_classes= cfg.bigcn.num_classes       # 4

        self.td_branch = _GCNBranch(in_dim, hidden_dim, out_dim, num_layers, dropout)
        self.bu_branch = _GCNBranch(in_dim, hidden_dim, out_dim, num_layers, dropout)

        # Classifier head: concat(z_td, z_bu) → num_classes
        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 6, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    # ------------------------------------------------------------------
    # Weight initialisation
    # ------------------------------------------------------------------

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, data: Data) -> torch.Tensor:
        x            = data.x                   # (N, 768)
        edge_td      = data.edge_index           # (2, E_td)
        edge_bu      = data.edge_index_bu        # (2, E_bu)
        batch        = getattr(data, "batch", None)

        # Handle single-graph inference (no batch vector)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # --- GCN branches ---
        z_td = self.td_branch(x, edge_td)       # (N, 128)
        z_bu = self.bu_branch(x, edge_bu)       # (N, 128)

        # --- Mean pooling per graph ---
                # --- Root-aware graph pooling ---
        z_td_mean = self._mean_pool(z_td, batch)        # (B, 128)
        z_bu_mean = self._mean_pool(z_bu, batch)        # (B, 128)

        z_td_max = self._max_pool(z_td, batch)          # (B, 128)
        z_bu_max = self._max_pool(z_bu, batch)          # (B, 128)

        root_indices = self._root_indices(batch)        # (B,)
        z_td_root = z_td[root_indices]                  # (B, 128)
        z_bu_root = z_bu[root_indices]                  # (B, 128)

        # --- Concatenate and classify ---
        z = torch.cat(
            [z_td_mean, z_td_max, z_td_root, z_bu_mean, z_bu_max, z_bu_root],
            dim=-1,
        )                                                # (B, 768)

        logits = self.classifier(z)                     # (B, num_classes)
        # print("DEBUG z shape before classifier:", z.shape)
        return logits

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Segment mean over nodes within each graph in the batch."""
        num_graphs = int(batch.max().item()) + 1
        out = torch.zeros(num_graphs, x.size(-1), device=x.device, dtype=x.dtype)
        count = torch.zeros(num_graphs, 1, device=x.device, dtype=x.dtype)
        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
        count.scatter_add_(0, batch.unsqueeze(-1),
                           torch.ones(batch.size(0), 1, device=x.device))
        count = count.clamp(min=1)
        return out / count
    
    @staticmethod
    def _max_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """Segment max over nodes within each graph in the batch."""
        num_graphs = int(batch.max().item()) + 1
        out = torch.full(
            (num_graphs, x.size(-1)),
            fill_value=-float("inf"),
            device=x.device,
            dtype=x.dtype,
        )

        for graph_id in range(num_graphs):
            mask = batch == graph_id
            if mask.any():
                out[graph_id] = x[mask].max(dim=0).values
            else:
                out[graph_id] = 0.0

        return out


    @staticmethod
    def _root_indices(batch: torch.Tensor) -> torch.Tensor:
        """
        Return the first node index for each graph in the batch.

        This works because each Data object is built with the root tweet as node 0,
        and PyG preserves per-graph node order inside a batch.
        """
        num_graphs = int(batch.max().item()) + 1
        roots = []

        for graph_id in range(num_graphs):
            roots.append(torch.where(batch == graph_id)[0][0])

        return torch.stack(roots)


# ---------------------------------------------------------------------------
# Convenience: count trainable parameters
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick smoke test (single graph, no labels needed)
    import torch
    from torch_geometric.data import Data

    N, E = 50, 60
    x         = torch.randn(N, cfg.bigcn.text_embed_dim + 4)  # 772
    src       = torch.randint(0, N, (E,))
    dst       = torch.randint(0, N, (E,))
    ei_td     = torch.stack([src, dst])
    ei_bu     = torch.stack([dst, src])

    dummy = Data(x=x, edge_index=ei_td, edge_index_bu=ei_bu,
                 y=torch.tensor(0))

    model  = BiGCN()
    logits = model(dummy)
    print(f"BiGCN logits shape : {logits.shape}")          # (1, 4)
    print(f"Trainable params   : {count_parameters(model):,}")
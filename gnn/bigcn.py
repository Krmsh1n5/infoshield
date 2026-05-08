"""
BiGCN — Bi-directional Graph Convolutional Network for rumour classification.

Implements Bian et al. (2020) "Rumor Detection on Social Media with
Bi-Directional Graph Convolutional Networks", AAAI 2020.

Two parallel GCN branches (top-down and bottom-up) with root-feature
enhancement after each layer. The root enhancement is critical for
benchmark-level accuracy — it lets the model retain the source claim's
context as the message propagates through the graph.

Input feature dim: 772 = 768 (RoBERTa CLS, broadcast from root) + 4 (structural).
See dataset.py BROADCAST DESIGN comment for why broadcast is correct.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

from config import cfg


class GCNBranch(nn.Module):
    """One direction of BiGCN (TD or BU) with root-feature enhancement."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        # After concat with broadcast root features, input becomes hidden_dim + in_dim
        self.conv2 = GCNConv(hidden_dim + in_dim, out_dim)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        root_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Layer 1
        h1 = F.relu(self.conv1(x, edge_index))
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # Root enhancement: broadcast each graph's root features to all its nodes
        root_x = x[root_mask]              # (B, in_dim) — one root per graph
        root_expanded = root_x[batch]      # (N, in_dim) — broadcast back to nodes
        h1_aug = torch.cat([h1, root_expanded], dim=1)

        # Layer 2
        h2 = F.relu(self.conv2(h1_aug, edge_index))
        h2 = F.dropout(h2, p=self.dropout, training=self.training)

        # Mean pool to graph-level representation
        return global_mean_pool(h2, batch)  # (B, out_dim)


class BiGCN(nn.Module):
    """Top-down + bottom-up GCN branches → concat → classifier head."""

    def __init__(self):
        super().__init__()
        in_dim = cfg.bigcn.text_embed_dim   # 772
        hidden = cfg.bigcn.gcn_hidden_dim   # 256
        out_dim = cfg.bigcn.gcn_output_dim  # 128
        dropout = cfg.bigcn.dropout         # 0.3
        n_classes = cfg.bigcn.num_classes   # 4

        self.td_branch = GCNBranch(in_dim, hidden, out_dim, dropout)
        self.bu_branch = GCNBranch(in_dim, hidden, out_dim, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, n_classes),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        batch = data.batch
        root_mask = data.root_mask

        td_out = self.td_branch(x, data.edge_index, batch, root_mask)
        bu_out = self.bu_branch(x, data.edge_index_bu, batch, root_mask)

        combined = torch.cat([td_out, bu_out], dim=1)  # (B, 2*out_dim)
        return self.classifier(combined)               # (B, n_classes)
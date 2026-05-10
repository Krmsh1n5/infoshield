"""
BiGCN: Bi-directional Graph Convolutional Network for rumour classification.

This module expects each PyG Data object to contain:
    x             : FloatTensor [num_nodes, node_feature_dim]
    edge_index    : LongTensor  [2, num_td_edges]
    edge_index_bu : LongTensor  [2, num_bu_edges]
    root_mask     : BoolTensor  [num_nodes], exactly one True per graph
    y             : LongTensor  scalar label, optional for inference

The dataset builds node features as:
    [RoBERTa root CLS embedding broadcast to all nodes | 4 structural features]

For the current project config, node_feature_dim is 772.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

from config import cfg


STRUCT_FEATURE_DIM = 4


def _node_feature_dim_from_cfg() -> int:
    """
    Resolve the model input feature dimension.

    Some project versions used cfg.bigcn.text_embed_dim to mean the raw RoBERTa
    dimension, 768. The current config uses it as the final node feature size,
    772. This helper supports both conventions.
    """
    dim = int(cfg.bigcn.text_embed_dim)

    # Backward-compatible case: config stores only RoBERTa-base hidden size.
    if dim == 768:
        return dim + STRUCT_FEATURE_DIM

    # Current project case: config already stores full node feature size.
    return dim


class GCNBranch(nn.Module):
    """One directional GCN branch with root-feature enhancement."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim + in_dim, out_dim)
        self.dropout = float(dropout)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        root_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return one graph-level vector per graph in the batch.

        Root enhancement:
            after the first GCN layer, the original root feature vector for
            each graph is broadcast back to every node in that graph and
            concatenated before the second GCN layer.
        """
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)

        root_x = x[root_mask]
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        if root_x.size(0) != num_graphs:
            raise ValueError(
                "root_mask must contain exactly one root node per graph. "
                f"Got {root_x.size(0)} roots for {num_graphs} graphs."
            )

        root_expanded = root_x[batch]
        h = torch.cat([h, root_expanded], dim=-1)

        h = F.relu(self.conv2(h, edge_index))
        h = F.dropout(h, p=self.dropout, training=self.training)

        return global_mean_pool(h, batch)


class BiGCN(nn.Module):
    """Top-down and bottom-up GCN branches followed by a classifier head."""

    def __init__(self) -> None:
        super().__init__()

        in_dim = _node_feature_dim_from_cfg()
        hidden_dim = int(cfg.bigcn.gcn_hidden_dim)
        out_dim = int(cfg.bigcn.gcn_output_dim)
        dropout = float(cfg.bigcn.dropout)
        num_classes = int(cfg.bigcn.num_classes)

        if int(cfg.bigcn.gcn_num_layers) != 2:
            raise ValueError(
                "This BiGCN implementation uses exactly 2 GCN layers per branch. "
                f"cfg.bigcn.gcn_num_layers={cfg.bigcn.gcn_num_layers} is not supported."
            )

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.td_branch = GCNBranch(in_dim, hidden_dim, out_dim, dropout)
        self.bu_branch = GCNBranch(in_dim, hidden_dim, out_dim, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, num_classes),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        if x.dim() != 2:
            raise ValueError(f"data.x must be rank-2 [N, F], got shape {tuple(x.shape)}")

        if x.size(-1) != self.in_dim:
            raise ValueError(
                f"BiGCN expected node feature dim {self.in_dim}, "
                f"but data.x has dim {x.size(-1)}. "
                "Check cfg.bigcn.text_embed_dim and dataset node features."
            )

        edge_td = data.edge_index
        edge_bu = getattr(data, "edge_index_bu", None)
        if edge_bu is None:
            edge_bu = edge_td.flip(0)

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        else:
            batch = batch.to(x.device)

        root_mask = getattr(data, "root_mask", None)
        if root_mask is None:
            root_mask = self._first_node_root_mask(batch, x.size(0))
        else:
            root_mask = root_mask.to(device=x.device, dtype=torch.bool)

        z_td = self.td_branch(x, edge_td, batch, root_mask)
        z_bu = self.bu_branch(x, edge_bu, batch, root_mask)

        z = torch.cat([z_td, z_bu], dim=-1)
        return self.classifier(z)

    @staticmethod
    def _first_node_root_mask(batch: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Fallback root mask: first node of each graph in a PyG batch.

        The dataset explicitly supplies root_mask. This fallback keeps single-graph
        smoke tests and older cached Data objects usable.
        """
        root_mask = torch.zeros(num_nodes, dtype=torch.bool, device=batch.device)
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1

        for graph_id in range(num_graphs):
            idx = torch.where(batch == graph_id)[0]
            if idx.numel() > 0:
                root_mask[idx[0]] = True

        return root_mask


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from torch_geometric.data import Data

    num_nodes, num_edges = 50, 60
    x = torch.randn(num_nodes, _node_feature_dim_from_cfg())

    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    edge_td = torch.stack([src, dst], dim=0)
    edge_bu = edge_td.flip(0)

    root_mask = torch.zeros(num_nodes, dtype=torch.bool)
    root_mask[0] = True

    dummy = Data(
        x=x,
        edge_index=edge_td,
        edge_index_bu=edge_bu,
        root_mask=root_mask,
        y=torch.tensor(0, dtype=torch.long),
        tweet_id="dummy",
    )

    model = BiGCN()
    logits = model(dummy)

    print(f"BiGCN logits shape : {tuple(logits.shape)}")
    print(f"Trainable params   : {count_parameters(model):,}")

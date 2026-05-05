r"""
gnn/bigcn.py
============
BiGCN for rumor detection.

This version uses:
1. TD-GCN branch
2. BU-GCN branch
3. Root feature enhancement before graph convolution
4. Root-aware pooling:
     TD mean, TD max, TD root, BU mean, BU max, BU root
5. Graph-level structural features
"""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GCNConv

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import cfg


class _GCNBranch(nn.Module):
    """Stack of GCNConv layers."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()

        assert num_layers >= 1, "num_layers must be >= 1"

        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]

        self.convs = nn.ModuleList([
            GCNConv(dims[i], dims[i + 1])
            for i in range(num_layers)
        ])

        self.dropout = dropout
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if i < self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class BiGCN(nn.Module):
    """Root-enhanced bi-directional GCN classifier."""

    def __init__(self):
        super().__init__()

        in_dim = int(cfg.bigcn.text_embed_dim)
        hidden_dim = int(cfg.bigcn.gcn_hidden_dim)
        out_dim = int(cfg.bigcn.gcn_output_dim)
        num_layers = int(cfg.bigcn.gcn_num_layers)
        dropout = float(cfg.bigcn.dropout)
        num_classes = int(cfg.bigcn.num_classes)

        self.graph_feature_dim = int(getattr(cfg.bigcn, "graph_feature_dim", 5))

        # Root enhancement doubles the input dimension:
        # x_aug = concat(node_x, root_x_for_same_graph)
        branch_in_dim = in_dim * 2

        self.td_branch = _GCNBranch(
            branch_in_dim,
            hidden_dim,
            out_dim,
            num_layers,
            dropout,
        )

        self.bu_branch = _GCNBranch(
            branch_in_dim,
            hidden_dim,
            out_dim,
            num_layers,
            dropout,
        )

        classifier_in_dim = out_dim * 6 + self.graph_feature_dim

        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, data: Data) -> torch.Tensor:
        x = data.x
        edge_td = data.edge_index
        edge_bu = data.edge_index_bu

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        root_indices = self._root_indices(batch)
        root_x = x[root_indices]
        root_x_per_node = root_x[batch]

        # Root feature enhancement before GCN.
        x_aug = torch.cat([x, root_x_per_node], dim=-1)

        z_td = self.td_branch(x_aug, edge_td)
        z_bu = self.bu_branch(x_aug, edge_bu)

        z_td_mean = self._mean_pool(z_td, batch)
        z_bu_mean = self._mean_pool(z_bu, batch)

        z_td_max = self._max_pool(z_td, batch)
        z_bu_max = self._max_pool(z_bu, batch)

        z_td_root = z_td[root_indices]
        z_bu_root = z_bu[root_indices]

        graph_features = self._get_graph_features(data, batch, x.device, x.dtype)

        z = torch.cat(
            [
                z_td_mean,
                z_td_max,
                z_td_root,
                z_bu_mean,
                z_bu_max,
                z_bu_root,
                graph_features,
            ],
            dim=-1,
        )

        return self.classifier(z)

    @staticmethod
    def _mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1

        out = torch.zeros(
            num_graphs,
            x.size(-1),
            device=x.device,
            dtype=x.dtype,
        )

        count = torch.zeros(
            num_graphs,
            1,
            device=x.device,
            dtype=x.dtype,
        )

        out.scatter_add_(0, batch.unsqueeze(-1).expand_as(x), x)
        count.scatter_add_(
            0,
            batch.unsqueeze(-1),
            torch.ones(batch.size(0), 1, device=x.device, dtype=x.dtype),
        )

        return out / count.clamp(min=1)

    @staticmethod
    def _max_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
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
        num_graphs = int(batch.max().item()) + 1
        roots = []

        for graph_id in range(num_graphs):
            roots.append(torch.where(batch == graph_id)[0][0])

        return torch.stack(roots)

    def _get_graph_features(
        self,
        data: Data,
        batch: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        num_graphs = int(batch.max().item()) + 1

        graph_features = getattr(data, "graph_features", None)

        if graph_features is None:
            return torch.zeros(
                num_graphs,
                self.graph_feature_dim,
                device=device,
                dtype=dtype,
            )

        graph_features = graph_features.to(device=device, dtype=dtype)

        if graph_features.dim() == 1:
            graph_features = graph_features.view(1, -1)

        if graph_features.size(0) != num_graphs:
            graph_features = graph_features.view(num_graphs, -1)

        # Pad or truncate if config differs.
        if graph_features.size(1) < self.graph_feature_dim:
            pad = torch.zeros(
                num_graphs,
                self.graph_feature_dim - graph_features.size(1),
                device=device,
                dtype=dtype,
            )
            graph_features = torch.cat([graph_features, pad], dim=-1)

        if graph_features.size(1) > self.graph_feature_dim:
            graph_features = graph_features[:, :self.graph_feature_dim]

        return graph_features


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from torch_geometric.data import Batch, Data

    num_graphs = 3
    graphs = []

    for _ in range(num_graphs):
        n = 20
        e = 30

        x = torch.randn(n, int(cfg.bigcn.text_embed_dim))

        src = torch.randint(0, n, (e,))
        dst = torch.randint(0, n, (e,))

        edge_index = torch.stack([src, dst])
        edge_index_bu = torch.stack([dst, src])

        graph_features = torch.randn(1, int(getattr(cfg.bigcn, "graph_feature_dim", 5)))

        graphs.append(Data(
            x=x,
            edge_index=edge_index,
            edge_index_bu=edge_index_bu,
            graph_features=graph_features,
            y=torch.tensor(0),
        ))

    batch = Batch.from_data_list(graphs)

    model = BiGCN()
    logits = model(batch)

    print(f"BiGCN logits shape : {tuple(logits.shape)}")
    print(f"Trainable params   : {count_parameters(model):,}")
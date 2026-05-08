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

        # +4 for structural node features (depth, in_degree, out_degree, is_root)
        # appended in dataset._build_pyg_graph — actual input dim is 772
        in_dim     = cfg.bigcn.text_embed_dim + 4    # 768 + 4 = 772
        hidden_dim = cfg.bigcn.gcn_hidden_dim    # 256
        out_dim    = cfg.bigcn.gcn_output_dim    # 128
        num_layers = cfg.bigcn.gcn_num_layers    # 2
        dropout    = cfg.bigcn.dropout           # 0.3
        num_classes= cfg.bigcn.num_classes       # 4

        self.td_branch = GCNBranch(in_dim, hidden, out_dim, dropout)
        self.bu_branch = GCNBranch(in_dim, hidden, out_dim, dropout)

        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, n_classes),
        )

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
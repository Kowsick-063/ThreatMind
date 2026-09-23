from __future__ import annotations

import torch
from torch import nn

from graph_experiment.temporal_gnn import EdgeAwareGraphEncoder


class DeltaTemporalGNNWorldModel(nn.Module):
    """Temporal graph model whose primary target is the next graph-state delta."""

    def __init__(
        self,
        node_dim: int = 17,
        edge_dim: int = 11,
        graph_dim: int = 10,
        hidden_dim: int = 64,
        graph_embedding_dim: int = 64,
        temporal_hidden_dim: int = 64,
    ):
        super().__init__()
        self.graph_encoder = EdgeAwareGraphEncoder(
            node_dim,
            edge_dim,
            hidden_dim,
            graph_embedding_dim,
        )
        self.temporal = nn.GRU(
            input_size=graph_embedding_dim,
            hidden_size=temporal_hidden_dim,
            batch_first=True,
        )
        self.delta_head = nn.Linear(temporal_hidden_dim, graph_dim)

    def forward_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        _, hidden = self.temporal(embeddings)
        return self.delta_head(hidden[-1])

    def forward(self, snapshots) -> torch.Tensor:
        embeddings = torch.stack(
            [
                self.graph_encoder(
                    snapshot["node_features"],
                    snapshot["edge_index"],
                    snapshot["edge_features"],
                )
                for snapshot in snapshots
            ],
            dim=0,
        ).unsqueeze(0)
        return self.forward_embeddings(embeddings)
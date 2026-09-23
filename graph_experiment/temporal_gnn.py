from __future__ import annotations

import torch
from torch import nn


class EdgeAwareGraphEncoder(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.node_projection = nn.Linear(node_dim, hidden_dim)
        self.edge_projection = nn.Linear(edge_dim, hidden_dim)
        self.update = nn.Linear(hidden_dim * 2, hidden_dim)
        self.embedding = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, node_features, edge_index, edge_features):
        node_state = self.node_projection(node_features)
        if edge_index.numel():
            source = edge_index[:, 0].long()
            destination = edge_index[:, 1].long()
            messages = node_state[source] + self.edge_projection(edge_features)
            aggregated = torch.zeros_like(node_state)
            aggregated.index_add_(0, destination, messages)
            counts = torch.zeros(
                node_state.shape[0], 1, device=node_state.device, dtype=node_state.dtype
            )
            counts.index_add_(
                0,
                destination,
                torch.ones(destination.shape[0], 1, device=node_state.device),
            )
            aggregated = aggregated / counts.clamp_min(1.0)
            edge_summary = self.edge_projection(edge_features).mean(dim=0)
        else:
            aggregated = torch.zeros_like(node_state)
            edge_summary = torch.zeros(
                node_state.shape[1], device=node_state.device, dtype=node_state.dtype
            )
        updated = torch.relu(self.update(torch.cat([node_state, aggregated], dim=1)))
        node_summary = updated.mean(dim=0)
        return torch.relu(self.embedding(torch.cat([node_summary, edge_summary], dim=0)))


class TemporalGNNWorldModel(nn.Module):
    def __init__(
        self,
        node_dim: int = 17,
        edge_dim: int = 11,
        graph_dim: int = 10,
        hidden_dim: int = 64,
        graph_embedding_dim: int = 64,
        temporal_hidden_dim: int = 64,
        category_count: int = 4,
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
        self.graph_head = nn.Linear(temporal_hidden_dim, graph_dim)
        self.attack_head = nn.Linear(temporal_hidden_dim, 1)
        self.category_head = nn.Linear(temporal_hidden_dim, category_count)

    def encode_sequence(self, snapshots):
        embeddings = [
            self.graph_encoder(
                snapshot["node_features"],
                snapshot["edge_index"],
                snapshot["edge_features"],
            )
            for snapshot in snapshots
        ]
        temporal_input = torch.stack(embeddings, dim=0).unsqueeze(0)
        _, hidden = self.temporal(temporal_input)
        return hidden[-1, 0]

    def forward_embeddings(self, embeddings):
        _, hidden = self.temporal(embeddings)
        current = hidden[-1]
        return {
            "graph_features": self.graph_head(current),
            "attack_logit": self.attack_head(current).squeeze(-1),
            "category_logits": self.category_head(current),
        }

    def forward(self, snapshots):
        hidden = self.encode_sequence(snapshots)
        return {
            "graph_features": self.graph_head(hidden),
            "attack_logit": self.attack_head(hidden).squeeze(-1),
            "category_logits": self.category_head(hidden),
        }


def multitask_loss(
    outputs,
    graph_target,
    attack_target,
    category_target,
    graph_loss_weight: float = 1.0,
    attack_loss_weight: float = 1.0,
    category_loss_weight: float = 1.0,
):
    graph_loss = nn.functional.mse_loss(outputs["graph_features"], graph_target)
    attack_loss = nn.functional.binary_cross_entropy_with_logits(
        outputs["attack_logit"], attack_target
    )
    category_loss = nn.functional.cross_entropy(
        outputs["category_logits"], category_target.reshape(-1)
    )
    total = (
        graph_loss_weight * graph_loss
        + attack_loss_weight * attack_loss
        + category_loss_weight * category_loss
    )
    return total, {
        "graph_loss": float(graph_loss.detach()),
        "attack_loss": float(attack_loss.detach()),
        "category_loss": float(category_loss.detach()),
        "total_loss": float(total.detach()),
    }
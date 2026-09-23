from pathlib import Path

import joblib
import numpy as np
import torch

from graph_experiment.dataset import EDGE_FEATURE_NAMES, GRAPH_FEATURE_NAMES, NODE_FEATURE_NAMES
from graph_experiment.delta_temporal_gnn import DeltaTemporalGNNWorldModel


CHECKPOINT_FILE = Path("graph_experiment/artifacts/delta_temporal_gnn_world_model.pt")
STATE_SCALER_FILE = Path("data/processed/graphs/temporal_gnn_scaler.joblib")
DELTA_SCALER_FILE = Path("data/processed/graphs/delta_temporal_gnn_scaler.joblib")


def load_model():
    checkpoint = torch.load(CHECKPOINT_FILE, map_location="cpu", weights_only=True)
    model = DeltaTemporalGNNWorldModel(
        node_dim=checkpoint["node_dim"],
        edge_dim=checkpoint["edge_dim"],
        graph_dim=checkpoint["graph_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        graph_embedding_dim=checkpoint["graph_embedding_dim"],
        temporal_hidden_dim=checkpoint["temporal_hidden_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def predict_next_delta(sequence, model=None, state_scalers=None, delta_scaler=None):
    if model is None:
        model = load_model()
    if state_scalers is None:
        state_scalers = joblib.load(STATE_SCALER_FILE)
    if delta_scaler is None:
        delta_scaler = joblib.load(DELTA_SCALER_FILE)
    prepared = [
        {
            **snapshot,
            "node_features": torch.from_numpy(
                state_scalers["node"].transform(snapshot["node_features"]).astype(np.float32)
            ),
            "edge_features": torch.from_numpy(
                state_scalers["edge"].transform(snapshot["edge_features"]).astype(np.float32)
            ),
            "edge_index": torch.from_numpy(snapshot["edge_index"]).long(),
        }
        for snapshot in sequence
    ]
    with torch.no_grad():
        embeddings = torch.stack(
            [
                model.graph_encoder(
                    snapshot["node_features"],
                    snapshot["edge_index"],
                    snapshot["edge_features"],
                )
                for snapshot in prepared
            ]
        ).unsqueeze(0)
        scaled_delta = model.forward_embeddings(embeddings).numpy()[0]
    return delta_scaler.inverse_transform(scaled_delta.reshape(1, -1))[0]


def predict_next_graph(sequence, model=None, state_scalers=None, delta_scaler=None):
    delta = predict_next_delta(sequence, model, state_scalers, delta_scaler)
    current = sequence[-1]["graph_features"]
    return current + delta
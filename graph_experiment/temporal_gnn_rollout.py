from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch

from graph_experiment.dataset import (
    EDGE_FEATURE_NAMES,
    GRAPH_FEATURE_NAMES,
    LABEL_NAMES,
    NODE_FEATURE_NAMES,
)
from graph_experiment.temporal_gnn import TemporalGNNWorldModel


CHECKPOINT_FILE = Path("graph_experiment/artifacts/temporal_gnn_world_model.pt")
SCALER_FILE = Path("data/processed/graphs/temporal_gnn_scaler.joblib")


def load_model():
    checkpoint = torch.load(CHECKPOINT_FILE, map_location="cpu", weights_only=True)
    model = TemporalGNNWorldModel(
        node_dim=checkpoint["node_dim"],
        edge_dim=checkpoint["edge_dim"],
        graph_dim=checkpoint["graph_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        graph_embedding_dim=checkpoint["graph_embedding_dim"],
        temporal_hidden_dim=checkpoint["temporal_hidden_dim"],
        category_count=checkpoint["category_count"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _prepare_sequence(sequence, scalers):
    return [
        {
            **snapshot,
            "node_features": torch.from_numpy(
                scalers["node"].transform(snapshot["node_features"]).astype(np.float32)
            ),
            "edge_features": torch.from_numpy(
                scalers["edge"].transform(snapshot["edge_features"]).astype(np.float32)
            ),
            "edge_index": torch.from_numpy(snapshot["edge_index"]).long(),
        }
        for snapshot in sequence
    ]


def predict_next_graph(sequence, model=None, scalers=None):
    if model is None:
        model = load_model()
    if scalers is None:
        scalers = joblib.load(SCALER_FILE)
    with torch.no_grad():
        outputs = model(_prepare_sequence(sequence, scalers))
    graph_features = scalers["graph"].inverse_transform(
        outputs["graph_features"].numpy().reshape(1, -1)
    )[0]
    attack_probability = float(torch.sigmoid(outputs["attack_logit"]).item())
    category_probabilities = torch.softmax(outputs["category_logits"], dim=0).numpy()
    category_index = int(category_probabilities.argmax())
    return {
        "graph_features": {
            name: float(graph_features[index])
            for index, name in enumerate(GRAPH_FEATURE_NAMES)
        },
        "attack_probability": attack_probability,
        "category_probabilities": {
            label: float(category_probabilities[index])
            for index, label in enumerate(LABEL_NAMES)
        },
        "predicted_category": LABEL_NAMES[category_index],
        "topology_prediction": "not produced",
    }


def forecast(sequence, horizons=12, model=None, scalers=None):
    if horizons < 1:
        raise ValueError("horizons must be positive.")
    point = predict_next_graph(sequence, model=model, scalers=scalers)
    return [
        {"horizon": horizon, **point}
        for horizon in range(1, horizons + 1)
    ]
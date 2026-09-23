from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch

from graph_experiment.dataset import DATASET_FILE, GRAPH_FEATURE_NAMES, GraphDataset
from graph_experiment.delta_temporal_gnn import DeltaTemporalGNNWorldModel


ARTIFACT_DIRECTORY = Path("graph_experiment/artifacts")
CHECKPOINT_FILE = ARTIFACT_DIRECTORY / "delta_temporal_gnn_world_model.pt"
HISTORY_FILE = ARTIFACT_DIRECTORY / "delta_temporal_gnn_training_history.json"
RESULTS_FILE = ARTIFACT_DIRECTORY / "delta_temporal_gnn_evaluation_results.json"
PREDICTIONS_FILE = ARTIFACT_DIRECTORY / "delta_temporal_gnn_predictions.npz"
STATE_SCALER_FILE = Path("data/processed/graphs/temporal_gnn_scaler.joblib")
DELTA_SCALER_FILE = Path("data/processed/graphs/delta_temporal_gnn_scaler.joblib")


def _load_model():
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


def _prepare_snapshots(dataset, state_scalers):
    prepared = []
    for index in range(len(dataset.arrays["snapshot__timestamps"])):
        snapshot = dataset.snapshot(index)
        prepared.append(
            {
                "node_features": torch.from_numpy(
                    state_scalers["node"].transform(snapshot["node_features"]).astype(np.float32)
                ),
                "edge_features": torch.from_numpy(
                    state_scalers["edge"].transform(snapshot["edge_features"]).astype(np.float32)
                ),
                "edge_index": torch.from_numpy(snapshot["edge_index"]).long(),
            }
        )
    return prepared


def evaluate():
    dataset = GraphDataset(DATASET_FILE)
    model = _load_model()
    state_scalers = joblib.load(STATE_SCALER_FILE)
    delta_scaler = joblib.load(DELTA_SCALER_FILE)
    prepared = _prepare_snapshots(dataset, state_scalers)
    test_sequences = dataset.arrays["test__sequence_indices"]
    predicted_deltas = []
    actual_deltas = []
    predicted_states = []
    actual_states = []
    persistence_states = []
    timestamps = []

    with torch.no_grad():
        for sequence in test_sequences:
            unique_indices = np.unique(sequence)
            embeddings_by_snapshot = {}
            for snapshot_index in unique_indices:
                snapshot = prepared[int(snapshot_index)]
                embeddings_by_snapshot[int(snapshot_index)] = model.graph_encoder(
                    snapshot["node_features"],
                    snapshot["edge_index"],
                    snapshot["edge_features"],
                )
            embeddings = torch.stack(
                [embeddings_by_snapshot[int(snapshot_index)] for snapshot_index in sequence]
            ).unsqueeze(0)
            scaled_delta = model.forward_embeddings(embeddings).numpy()[0]
            delta = delta_scaler.inverse_transform(scaled_delta.reshape(1, -1))[0]
            current = dataset.snapshot(int(sequence[-1]))["graph_features"]
            actual = dataset.snapshot(int(sequence[-1]) + 1)["graph_features"]
            predicted_deltas.append(delta)
            actual_deltas.append(actual - current)
            predicted_states.append(current + delta)
            actual_states.append(actual)
            persistence_states.append(current)
            timestamps.append(dataset.snapshot(int(sequence[-1]) + 1)["timestamp"])

    predicted_deltas = np.asarray(predicted_deltas, dtype=np.float32)
    actual_deltas = np.asarray(actual_deltas, dtype=np.float32)
    predicted_states = np.asarray(predicted_states, dtype=np.float32)
    actual_states = np.asarray(actual_states, dtype=np.float32)
    persistence_states = np.asarray(persistence_states, dtype=np.float32)

    absolute_predictions = np.load(
        ARTIFACT_DIRECTORY / "temporal_gnn_predictions.npz"
    )
    absolute_states = absolute_predictions["predicted_graph_features"]

    delta_rows = []
    absolute_rows = []
    for index, feature in enumerate(GRAPH_FEATURE_NAMES):
        delta_error = predicted_deltas[:, index] - actual_deltas[:, index]
        actual_delta = actual_deltas[:, index]
        if np.std(actual_delta) > 1e-12 and np.std(predicted_deltas[:, index]) > 1e-12:
            correlation = float(np.corrcoef(predicted_deltas[:, index], actual_delta)[0, 1])
        else:
            correlation = None
        delta_rows.append({
            "feature": feature,
            "delta_mae": float(np.mean(np.abs(delta_error))),
            "delta_rmse": float(np.sqrt(np.mean(delta_error ** 2))),
            "delta_correlation": correlation,
        })
        rows = {
            "feature": feature,
            "persistence_mae": float(np.mean(np.abs(persistence_states[:, index] - actual_states[:, index]))),
            "absolute_gnn_mae": float(np.mean(np.abs(absolute_states[:, index] - actual_states[:, index]))),
            "delta_gnn_mae": float(np.mean(np.abs(predicted_states[:, index] - actual_states[:, index]))),
            "persistence_rmse": float(np.sqrt(np.mean((persistence_states[:, index] - actual_states[:, index]) ** 2))),
            "absolute_gnn_rmse": float(np.sqrt(np.mean((absolute_states[:, index] - actual_states[:, index]) ** 2))),
            "delta_gnn_rmse": float(np.sqrt(np.mean((predicted_states[:, index] - actual_states[:, index]) ** 2))),
        }
        absolute_rows.append(rows)

    def aggregate(values):
        error = values - actual_states
        feature_mae = np.mean(np.abs(error), axis=0)
        feature_rmse = np.sqrt(np.mean(error ** 2, axis=0))
        return {
            "aggregate_mae": float(np.mean(np.abs(error))),
            "macro_mae": float(feature_mae.mean()),
            "aggregate_rmse": float(np.sqrt(np.mean(error ** 2))),
            "macro_rmse": float(feature_rmse.mean()),
        }

    results = {
        "training": {
            "train_sequences": int(len(dataset.arrays["train__sequence_indices"])),
            "validation_sequences": int(len(dataset.arrays["validation__sequence_indices"])),
            "test_sequences": int(len(test_sequences)),
            "history": json.loads(HISTORY_FILE.read_text(encoding="utf-8")),
        },
        "delta_prediction": delta_rows,
        "absolute_reconstructed_prediction": absolute_rows,
        "aggregate_metrics": {
            "persistence": aggregate(persistence_states),
            "absolute_gnn": aggregate(absolute_states),
            "delta_gnn": aggregate(predicted_states),
        },
        "delta_scaler": str(DELTA_SCALER_FILE),
        "horizon": 1,
        "limitation": "H1 only; future graph topology cannot be reconstructed for recursive H2-H12 graph inputs.",
    }
    np.savez_compressed(
        PREDICTIONS_FILE,
        predicted_deltas=predicted_deltas,
        actual_deltas=actual_deltas,
        predicted_states=predicted_states,
        actual_states=actual_states,
        persistence_states=persistence_states,
        absolute_gnn_states=absolute_states,
        target_timestamps=np.asarray(timestamps),
    )
    RESULTS_FILE.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    results = evaluate()
    print(json.dumps(results["aggregate_metrics"], indent=2))
    print(f"Saved: {RESULTS_FILE}")
    print(f"Saved: {PREDICTIONS_FILE}")
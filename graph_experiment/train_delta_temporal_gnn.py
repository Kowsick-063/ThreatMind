from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from graph_experiment.dataset import (
    DATASET_FILE,
    EDGE_FEATURE_NAMES,
    GRAPH_FEATURE_NAMES,
    GraphDataset,
    NODE_FEATURE_NAMES,
)
from graph_experiment.delta_temporal_gnn import DeltaTemporalGNNWorldModel


ARTIFACT_DIRECTORY = Path("graph_experiment/artifacts")
CHECKPOINT_FILE = ARTIFACT_DIRECTORY / "delta_temporal_gnn_world_model.pt"
HISTORY_FILE = ARTIFACT_DIRECTORY / "delta_temporal_gnn_training_history.json"
STATE_SCALER_FILE = Path("data/processed/graphs/temporal_gnn_scaler.joblib")
DELTA_SCALER_FILE = Path("data/processed/graphs/delta_temporal_gnn_scaler.joblib")
BATCH_SIZE = 16


def prepare_snapshots(dataset, scalers):
    prepared = []
    for index in range(len(dataset.arrays["snapshot__timestamps"])):
        snapshot = dataset.snapshot(index)
        prepared.append(
            {
                "node_features": torch.from_numpy(
                    scalers["node"].transform(snapshot["node_features"]).astype(np.float32)
                ),
                "edge_features": torch.from_numpy(
                    scalers["edge"].transform(snapshot["edge_features"]).astype(np.float32)
                ),
                "edge_index": torch.from_numpy(snapshot["edge_index"]).long(),
            }
        )
    return prepared


def raw_delta(dataset, sequence):
    current = dataset.snapshot(int(sequence[-1]))["graph_features"]
    target = dataset.snapshot(int(sequence[-1]) + 1)["graph_features"]
    return target - current


def fit_delta_scaler(dataset):
    train_sequences = dataset.arrays["train__sequence_indices"]
    deltas = np.asarray([raw_delta(dataset, sequence) for sequence in train_sequences])
    return StandardScaler().fit(deltas)


def run_epoch(model, dataset, split, state_scalers, delta_scaler, prepared_snapshots, optimizer):
    training = optimizer is not None
    model.train(training)
    sequences = dataset.arrays[f"{split}__sequence_indices"]
    total_loss = 0.0
    count = 0

    for batch_start in range(0, len(sequences), BATCH_SIZE):
        batch = sequences[batch_start:batch_start + BATCH_SIZE]
        unique_indices = np.unique(batch.reshape(-1))
        embeddings_by_snapshot = {}
        for snapshot_index in unique_indices:
            snapshot = prepared_snapshots[int(snapshot_index)]
            embeddings_by_snapshot[int(snapshot_index)] = model.graph_encoder(
                snapshot["node_features"],
                snapshot["edge_index"],
                snapshot["edge_features"],
            )
        embeddings = torch.stack(
            [
                torch.stack(
                    [embeddings_by_snapshot[int(snapshot_index)] for snapshot_index in sequence]
                )
                for sequence in batch
            ]
        )
        targets = np.asarray([raw_delta(dataset, sequence) for sequence in batch])
        scaled_targets = torch.from_numpy(
            delta_scaler.transform(targets).astype(np.float32)
        )
        if training:
            optimizer.zero_grad()
        predictions = model.forward_embeddings(embeddings)
        loss = torch.nn.functional.mse_loss(predictions, scaled_targets)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach()) * len(batch)
        count += len(batch)
    return total_loss / max(count, 1)


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)
    dataset = GraphDataset(DATASET_FILE)
    state_scalers = joblib.load(STATE_SCALER_FILE)
    delta_scaler = fit_delta_scaler(dataset)
    DELTA_SCALER_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(delta_scaler, DELTA_SCALER_FILE)
    prepared_snapshots = prepare_snapshots(dataset, state_scalers)

    model = DeltaTemporalGNNWorldModel(
        node_dim=len(NODE_FEATURE_NAMES),
        edge_dim=len(EDGE_FEATURE_NAMES),
        graph_dim=len(GRAPH_FEATURE_NAMES),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    best_validation = float("inf")
    best_epoch = 0
    stale = 0

    for epoch in range(1, 31):
        train_loss = run_epoch(
            model,
            dataset,
            "train",
            state_scalers,
            delta_scaler,
            prepared_snapshots,
            optimizer,
        )
        with torch.no_grad():
            validation_loss = run_epoch(
                model,
                dataset,
                "validation",
                state_scalers,
                delta_scaler,
                prepared_snapshots,
                None,
            )
        row = {
            "epoch": epoch,
            "train_delta_loss": train_loss,
            "validation_delta_loss": validation_loss,
        }
        history.append(row)
        print(row)
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_epoch = epoch
            stale = 0
            ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "node_dim": len(NODE_FEATURE_NAMES),
                    "edge_dim": len(EDGE_FEATURE_NAMES),
                    "graph_dim": len(GRAPH_FEATURE_NAMES),
                    "hidden_dim": 64,
                    "graph_embedding_dim": 64,
                    "temporal_hidden_dim": 64,
                    "sequence_length": 60,
                    "target": "raw_graph_state_delta",
                    "graph_feature_names": GRAPH_FEATURE_NAMES,
                },
                CHECKPOINT_FILE,
            )
        else:
            stale += 1
            if stale >= 7:
                break

    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"checkpoint={CHECKPOINT_FILE}")
    print(f"delta_scaler={DELTA_SCALER_FILE}")
    print(f"best_epoch={best_epoch}")


if __name__ == "__main__":
    main()
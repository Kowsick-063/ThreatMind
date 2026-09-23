from __future__ import annotations

import argparse
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
from graph_experiment.temporal_gnn import TemporalGNNWorldModel, multitask_loss


ARTIFACT_DIRECTORY = Path("graph_experiment/artifacts")
CHECKPOINT_FILE = ARTIFACT_DIRECTORY / "temporal_gnn_world_model.pt"
HISTORY_FILE = ARTIFACT_DIRECTORY / "temporal_gnn_training_history.json"
SCALER_FILE = Path("data/processed/graphs/temporal_gnn_scaler.joblib")
BATCH_SIZE = 16


def fit_scalers(dataset: GraphDataset, train_indices: np.ndarray) -> dict[str, StandardScaler]:
    snapshot_indices = np.unique(train_indices.reshape(-1))
    node_values = []
    edge_values = []
    graph_values = []
    for index in snapshot_indices:
        snapshot = dataset.snapshot(int(index))
        node_values.append(snapshot["node_features"])
        edge_values.append(snapshot["edge_features"])
        graph_values.append(snapshot["graph_features"])
    scalers = {
        "node": StandardScaler().fit(np.concatenate(node_values, axis=0)),
        "edge": StandardScaler().fit(np.concatenate(edge_values, axis=0)),
        "graph": StandardScaler().fit(np.asarray(graph_values)),
    }
    return scalers


def scaled_sequence(dataset, split: str, index: int, scalers):
    sequence = []
    for snapshot in dataset.sequence(split, index):
        sequence.append(
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
        )
    return sequence


def scaled_target(dataset, split: str, index: int, scalers):
    target = dataset.target(split, index)
    return (
        torch.from_numpy(
            scalers["graph"].transform(target["graph_features"].reshape(1, -1))[0].astype(np.float32)
        ),
        torch.tensor(float(target["is_attack"]), dtype=torch.float32),
        torch.tensor(int(target["label"]), dtype=torch.long),
    )


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


def run_epoch(model, dataset, split, scalers, optimizer, prepared_snapshots, max_sequences=None):
    training = optimizer is not None
    model.train(training)
    indices = dataset.arrays[f"{split}__sequence_indices"]
    if max_sequences is not None:
        indices = indices[:max_sequences]
    totals = {"total_loss": 0.0, "graph_loss": 0.0, "attack_loss": 0.0, "category_loss": 0.0}
    for batch_start in range(0, len(indices), BATCH_SIZE):
        batch_indices = indices[batch_start:batch_start + BATCH_SIZE]
        unique_indices = np.unique(batch_indices.reshape(-1))
        embeddings_by_snapshot = {}
        for snapshot_index in unique_indices:
            snapshot = prepared_snapshots[int(snapshot_index)]
            embeddings_by_snapshot[int(snapshot_index)] = model.graph_encoder(
                snapshot["node_features"],
                snapshot["edge_index"],
                snapshot["edge_features"],
            )
        sequence_embeddings = torch.stack(
            [
                torch.stack(
                    [embeddings_by_snapshot[int(snapshot_index)] for snapshot_index in sequence]
                )
                for sequence in batch_indices
            ]
        )
        graph_targets = []
        attack_targets = []
        category_targets = []
        for sequence in batch_indices:
            target = dataset.snapshot(int(sequence[-1]) + 1)
            graph_targets.append(
                scalers["graph"].transform(target["graph_features"].reshape(1, -1))[0]
            )
            attack_targets.append(float(target["is_attack"]))
            category_targets.append(int(target["label"]))
        graph_target = torch.from_numpy(np.asarray(graph_targets, dtype=np.float32))
        attack_target = torch.tensor(attack_targets, dtype=torch.float32)
        category_target = torch.tensor(category_targets, dtype=torch.long)
        if training:
            optimizer.zero_grad()
        outputs = model.forward_embeddings(sequence_embeddings)
        loss, values = multitask_loss(outputs, graph_target, attack_target, category_target)
        if training:
            loss.backward()
            optimizer.step()
        for key in totals:
            totals[key] += values[key] * len(batch_indices)
    count = max(len(indices), 1)
    return {key: value / count for key, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--max-sequences", type=int, default=None)
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    dataset = GraphDataset(DATASET_FILE)
    train_indices = dataset.arrays["train__sequence_indices"]
    scalers = fit_scalers(dataset, train_indices)
    prepared_snapshots = prepare_snapshots(dataset, scalers)
    SCALER_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scalers, SCALER_FILE)

    model = TemporalGNNWorldModel(
        node_dim=len(NODE_FEATURE_NAMES),
        edge_dim=len(EDGE_FEATURE_NAMES),
        graph_dim=len(GRAPH_FEATURE_NAMES),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    history = []
    best_validation = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, args.epochs + 1):
        train_values = run_epoch(
            model,
            dataset,
            "train",
            scalers,
            optimizer,
            prepared_snapshots,
            args.max_sequences,
        )
        with torch.no_grad():
            validation_values = run_epoch(
                model,
                dataset,
                "validation",
                scalers,
                None,
                prepared_snapshots,
                args.max_sequences,
            )
        row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_values.items()}, **{f"validation_{k}": v for k, v in validation_values.items()}}
        history.append(row)
        print(row)
        if validation_values["total_loss"] < best_validation:
            best_validation = validation_values["total_loss"]
            best_epoch = epoch
            stale_epochs = 0
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
                    "category_count": 4,
                    "label_names": ["BENIGN", "Bot", "PortScan", "DDoS"],
                },
                CHECKPOINT_FILE,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                break

    ARTIFACT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"checkpoint={CHECKPOINT_FILE}")
    print(f"scaler={SCALER_FILE}")
    print(f"best_epoch={best_epoch}")


if __name__ == "__main__":
    main()
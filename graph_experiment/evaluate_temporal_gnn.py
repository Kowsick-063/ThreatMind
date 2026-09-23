import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
import torch

from graph_experiment.dataset import (
    DATASET_FILE,
    EDGE_FEATURE_NAMES,
    GRAPH_FEATURE_NAMES,
    LABEL_NAMES,
    GraphDataset,
    NODE_FEATURE_NAMES,
)
from graph_experiment.train_temporal_gnn import scaled_sequence
from graph_experiment.temporal_gnn import TemporalGNNWorldModel


ARTIFACT_DIRECTORY = Path("graph_experiment/artifacts")
CHECKPOINT_FILE = ARTIFACT_DIRECTORY / "temporal_gnn_world_model.pt"
SCALER_FILE = Path("data/processed/graphs/temporal_gnn_scaler.joblib")
RESULTS_FILE = ARTIFACT_DIRECTORY / "temporal_gnn_evaluation_results.json"
PREDICTIONS_FILE = ARTIFACT_DIRECTORY / "temporal_gnn_predictions.npz"


def load_experiment():
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
    return model, joblib.load(SCALER_FILE), checkpoint


def _feature_metrics(predicted, actual, persistence):
    rows = []
    for index, name in enumerate(GRAPH_FEATURE_NAMES):
        error = predicted[:, index] - actual[:, index]
        baseline_error = persistence[:, index] - actual[:, index]
        rows.append({
            "feature": name,
            "gnn_mae": float(np.mean(np.abs(error))),
            "persistence_mae": float(np.mean(np.abs(baseline_error))),
            "gnn_rmse": float(np.sqrt(np.mean(error ** 2))),
            "persistence_rmse": float(np.sqrt(np.mean(baseline_error ** 2))),
        })
    return rows


def _aggregate_metrics(predicted, actual, persistence):
    error = predicted - actual
    baseline_error = persistence - actual
    feature_mae = np.mean(np.abs(error), axis=0)
    feature_baseline_mae = np.mean(np.abs(baseline_error), axis=0)
    feature_rmse = np.sqrt(np.mean(error ** 2, axis=0))
    feature_baseline_rmse = np.sqrt(np.mean(baseline_error ** 2, axis=0))
    return {
        "gnn_aggregate_mae": float(np.mean(np.abs(error))),
        "persistence_aggregate_mae": float(np.mean(np.abs(baseline_error))),
        "gnn_macro_mae": float(feature_mae.mean()),
        "persistence_macro_mae": float(feature_baseline_mae.mean()),
        "gnn_aggregate_rmse": float(np.sqrt(np.mean(error ** 2))),
        "persistence_aggregate_rmse": float(np.sqrt(np.mean(baseline_error ** 2))),
        "gnn_macro_rmse": float(feature_rmse.mean()),
        "persistence_macro_rmse": float(feature_baseline_rmse.mean()),
        "features_gnn_better_mae": int(np.sum(feature_mae < feature_baseline_mae)),
        "features_gnn_better_rmse": int(np.sum(feature_rmse < feature_baseline_rmse)),
    }


def _attack_metrics(probability, actual, baseline_probability):
    return {
        "population": {
            "BENIGN": int(np.sum(actual == 0)),
            "DDoS_or_attack": int(np.sum(actual == 1)),
        },
        "gnn_roc_auc": float(roc_auc_score(actual, probability)),
        "gnn_pr_auc": float(average_precision_score(actual, probability)),
        "gnn_brier": float(brier_score_loss(actual, probability)),
        "gnn_mean_probability": float(probability.mean()),
        "persistence_roc_auc": float(roc_auc_score(actual, baseline_probability)),
        "persistence_pr_auc": float(average_precision_score(actual, baseline_probability)),
        "persistence_brier": float(brier_score_loss(actual, baseline_probability)),
        "persistence_mean_probability": float(baseline_probability.mean()),
        "attack_rate": float(actual.mean()),
    }


def _category_metrics(logits, actual, train_labels):
    predicted = logits.argmax(axis=1)
    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(LABEL_NAMES))),
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    return {
        "training_classes": [LABEL_NAMES[index] for index in sorted(set(train_labels.tolist()))],
        "classes_absent_from_training": [
            label for index, label in enumerate(LABEL_NAMES)
            if index not in set(train_labels.tolist())
        ],
        "test_support": {
            label: int(np.sum(actual == index))
            for index, label in enumerate(LABEL_NAMES)
        },
        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=list(range(len(LABEL_NAMES))),
        ).tolist(),
        "classification_report": report,
        "classification_claim": "Temporal generalization limitation; unseen test classes are not evidence of learned classification.",
    }


def evaluate():
    dataset = GraphDataset(DATASET_FILE)
    model, scalers, checkpoint = load_experiment()
    test_indices = dataset.arrays["test__sequence_indices"]
    train_labels = dataset.arrays["snapshot__labels"][
        dataset.arrays["train__sequence_indices"][:, -1]
    ]
    predictions = []
    actual = []
    attack_probability = []
    category_logits = []
    current_graph = []
    timestamps = []

    with torch.no_grad():
        for index in range(len(test_indices)):
            sequence = scaled_sequence(dataset, "test", index, scalers)
            target = dataset.target("test", index)
            outputs = model(sequence)
            predictions.append(
                scalers["graph"].inverse_transform(
                    outputs["graph_features"].numpy().reshape(1, -1)
                )[0]
            )
            actual.append(target["graph_features"])
            current_graph.append(dataset.snapshot(int(test_indices[index, -1]))["graph_features"])
            attack_probability.append(float(torch.sigmoid(outputs["attack_logit"]).item()))
            category_logits.append(outputs["category_logits"].numpy())
            timestamps.append(float(target["timestamp"]))

    predicted = np.asarray(predictions, dtype=np.float32)
    actual = np.asarray(actual, dtype=np.float32)
    persistence = np.asarray(current_graph, dtype=np.float32)
    attack_probability = np.asarray(attack_probability, dtype=np.float32)
    category_logits = np.asarray(category_logits, dtype=np.float32)
    actual_attack = dataset.arrays["snapshot__is_attack"][test_indices[:, -1] + 1]
    actual_categories = dataset.arrays["snapshot__labels"][test_indices[:, -1] + 1]
    persistence_attack = dataset.arrays["snapshot__is_attack"][test_indices[:, -1]]

    feature_rows = _feature_metrics(predicted, actual, persistence)
    aggregate = _aggregate_metrics(predicted, actual, persistence)
    attack = _attack_metrics(attack_probability, actual_attack, persistence_attack.astype(np.float32))
    category = _category_metrics(category_logits, actual_categories, train_labels)
    metrics = {
        "horizon": 1,
        "seconds_ahead": 5,
        "feature_metrics": feature_rows,
        "aggregate_metrics": aggregate,
        "attack_metrics": attack,
        "category_metrics": category,
    }

    np.savez_compressed(
        PREDICTIONS_FILE,
        predicted_graph_features=predicted,
        actual_graph_features=actual,
        persistence_graph_features=persistence,
        attack_probability=attack_probability,
        actual_attack=actual_attack,
        persistence_attack=persistence_attack,
        category_logits=category_logits,
        actual_category=actual_categories,
        target_timestamps=np.asarray(timestamps),
    )
    history = json.loads((ARTIFACT_DIRECTORY / "temporal_gnn_training_history.json").read_text())
    results = {
        "training_configuration": {
            "sequence_length": 60,
            "hidden_dim": checkpoint["hidden_dim"],
            "graph_embedding_dim": checkpoint["graph_embedding_dim"],
            "temporal_hidden_dim": checkpoint["temporal_hidden_dim"],
            "learning_rate": 0.001,
            "batch_size": 1,
            "max_epochs": 30,
            "graph_loss_weight": 1.0,
            "attack_loss_weight": 1.0,
            "category_loss_weight": 1.0,
        },
        "split_sizes": {
            "train": int(len(dataset.arrays["train__sequence_indices"])),
            "validation": int(len(dataset.arrays["validation__sequence_indices"])),
            "test": int(len(test_indices)),
        },
        "feature_names": {
            "node": NODE_FEATURE_NAMES,
            "edge": EDGE_FEATURE_NAMES,
            "graph": GRAPH_FEATURE_NAMES,
        },
        "training_history": history,
        "best_epoch": int(min(history, key=lambda row: row["validation_total_loss"])["epoch"]),
        "forecast_metrics": metrics,
        "horizon_metrics": {
            "supported_horizons": [1],
            "unsupported_horizons": list(range(2, 13)),
            "limitation": "Recursive H2-H12 graph forecasts are not evaluated because future node/edge topology cannot be reconstructed from graph-level predictions.",
        },
        "persistence_metrics": aggregate,
    }
    RESULTS_FILE.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


def main() -> None:
    results = evaluate()
    print(json.dumps(results["forecast_metrics"], indent=2))
    print(f"Saved: {RESULTS_FILE}")
    print(f"Saved: {PREDICTIONS_FILE}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
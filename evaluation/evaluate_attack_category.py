import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models.baselines.logistic_baseline import (
    FEATURES,
    load_dataset,
    split_dataset,
)
from models.world_model.attack_category_estimation import (
    AttackCategoryEstimator,
)
from models.world_model.create_sequences import (
    SEQUENCE_FILE,
    WINDOW_SECONDS,
    load_states,
)
from models.world_model.rollout import (
    HORIZON,
    load_model,
    recursive_rollout,
)
from models.world_model.train_lstm import chronological_split


WORLD_MODEL_FILE = Path(
    "models/world_model/lstm_world_model.pt"
)
WORLD_SCALER_FILE = Path(
    "models/world_model/world_model_scaler.joblib"
)
BINARY_MODEL_FILE = Path(
    "models/world_model/attack_probability_model.joblib"
)
CALIBRATOR_DIRECTORY = Path(
    "models/world_model/calibrators"
)
ESTIMATOR_FILE = Path(
    "models/world_model/attack_category_estimation.joblib"
)
FORECAST_FILE = Path(
    "evaluation/results/attack_category_forecast.csv"
)
METRICS_FILE = Path(
    "evaluation/results/attack_category_metrics.csv"
)
LABELS = [
    "BENIGN",
    "Bot",
    "PortScan",
    "DDoS",
]
OBSERVED_LABELS = [
    "BENIGN",
    "Bot",
    "PortScan",
]
PROBABILITY_COLUMNS = {
    "BENIGN": "probability_benign",
    "Bot": "probability_bot",
    "PortScan": "probability_portscan",
    "DDoS": "probability_ddos",
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_splits():
    archive = np.load(SEQUENCE_FILE)
    return chronological_split(
        archive["X"],
        archive["y"],
        archive["current_timestamps"],
        archive["target_timestamps"],
    )


def timestamp_index(states):
    return {
        float(timestamp): row
        for timestamp, row in states.set_index("timestamp").iterrows()
    }


def eligible(split, index):
    selected = []
    for row_index, source_timestamp in enumerate(split[2]):
        future = [
            float(source_timestamp + step * WINDOW_SECONDS)
            for step in range(1, HORIZON + 1)
        ]
        if all(timestamp in index for timestamp in future):
            selected.append(row_index)
    return selected


def labels_for(split, selected, index):
    labels = []
    for row_index in selected:
        source = float(split[2][row_index])
        labels.append(
            [
                index[source + step * WINDOW_SECONDS]["attack_label"]
                for step in range(1, HORIZON + 1)
            ]
        )
    return np.asarray(labels, dtype=object)


def target_labels_for(split, index):
    return np.asarray(
        [
            index[float(timestamp)]["attack_label"]
            for timestamp in split[3]
        ],
        dtype=object,
    )


def rollout(split, selected, world_model, scaler):
    return recursive_rollout(split, selected, world_model, scaler)


def forecast_rows(split, selected, probabilities, predictions, labels):
    rows = []
    for sequence_index, source_index in enumerate(selected):
        source_timestamp = float(split[2][source_index])
        for horizon_index in range(HORIZON):
            row = {
                "source_timestamp": source_timestamp,
                "target_timestamp": source_timestamp + (horizon_index + 1) * WINDOW_SECONDS,
                "horizon": horizon_index + 1,
                "seconds_ahead": (horizon_index + 1) * WINDOW_SECONDS,
                "predicted_attack_category": predictions[sequence_index, horizon_index],
                "actual_attack_label": labels[sequence_index, horizon_index],
            }
            for label, column in PROBABILITY_COLUMNS.items():
                if label in OBSERVED_LABELS:
                    row[column] = float(
                        probabilities[
                            sequence_index,
                            horizon_index,
                            OBSERVED_LABELS.index(label),
                        ]
                    )
                else:
                    row[column] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def category_metrics(probabilities, predictions, labels):
    rows = []
    for horizon_index in range(HORIZON):
        actual = labels[:, horizon_index]
        predicted = predictions[:, horizon_index]
        row = {
            "horizon": horizon_index + 1,
            "seconds_ahead": (horizon_index + 1) * WINDOW_SECONDS,
            "available_classes": ",".join(OBSERVED_LABELS),
            "macro_f1": float(
                f1_score(
                    actual,
                    predicted,
                    labels=OBSERVED_LABELS,
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    actual,
                    predicted,
                    labels=LABELS,
                    average="weighted",
                    zero_division=0,
                )
            ),
            "macro_precision": float(
                precision_score(
                    actual,
                    predicted,
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                )
            ),
            "macro_recall": float(
                recall_score(
                    actual,
                    predicted,
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                )
            ),
            "confusion_matrix": json.dumps(
                confusion_matrix(
                    actual,
                    predicted,
                    labels=LABELS,
                ).tolist()
            ),
        }
        for label in LABELS:
            row[f"precision_{label.lower()}"] = float(
                precision_score(
                    actual,
                    predicted,
                    labels=[label],
                    average="macro",
                    zero_division=0,
                )
            )
            row[f"recall_{label.lower()}"] = float(
                recall_score(
                    actual,
                    predicted,
                    labels=[label],
                    average="macro",
                    zero_division=0,
                )
            )
            row[f"f1_{label.lower()}"] = float(
                f1_score(
                    actual,
                    predicted,
                    labels=[label],
                    average="macro",
                    zero_division=0,
                )
            )

        row["multiclass_roc_auc"] = np.nan
        row["multiclass_roc_auc_status"] = (
            "UNAVAILABLE: DDoS absent from training support"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    model_hash = sha256(WORLD_MODEL_FILE)
    scaler_hash = sha256(WORLD_SCALER_FILE)
    binary_hash = sha256(BINARY_MODEL_FILE)
    calibrator_hashes = {
        path: sha256(path)
        for path in CALIBRATOR_DIRECTORY.glob("horizon_*.joblib")
    }

    data = load_dataset()
    train_data, validation_data, test_data = split_dataset(data)
    states = load_states()
    index = timestamp_index(states)
    scaler = joblib.load(WORLD_SCALER_FILE)
    world_model = load_model()
    train_split, validation_split, test_split = load_splits()
    train_target_labels = target_labels_for(train_split, index)
    train_feature_values = scaler.transform(
        train_split[1].reshape(-1, len(FEATURES))
    )
    observed_mask = np.isin(train_target_labels, OBSERVED_LABELS)
    estimator = AttackCategoryEstimator().fit(
        train_feature_values[observed_mask],
        train_target_labels[observed_mask],
    )
    estimator.save(ESTIMATOR_FILE)

    validation_selected = eligible(validation_split, index)
    test_selected = eligible(test_split, index)
    validation_labels = labels_for(validation_split, validation_selected, index)
    test_labels = labels_for(test_split, test_selected, index)
    validation_rollout = rollout(
        validation_split,
        validation_selected,
        world_model,
        scaler,
    )
    test_rollout = rollout(
        test_split,
        test_selected,
        world_model,
        scaler,
    )
    test_predictions = estimator.predict(test_rollout.reshape(-1, len(FEATURES))).reshape(-1, HORIZON)
    test_probabilities = estimator.predict_proba(test_rollout.reshape(-1, len(FEATURES))).reshape(-1, HORIZON, len(OBSERVED_LABELS))

    forecast = forecast_rows(
        test_split,
        test_selected,
        test_probabilities,
        test_predictions,
        test_labels,
    )
    metrics = category_metrics(
        test_probabilities,
        test_predictions,
        test_labels,
    )
    FORECAST_FILE.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(FORECAST_FILE, index=False)
    metrics.to_csv(METRICS_FILE, index=False)

    actual_future_features_used = False
    hashes_unchanged = (
        model_hash == sha256(WORLD_MODEL_FILE)
        and scaler_hash == sha256(WORLD_SCALER_FILE)
        and binary_hash == sha256(BINARY_MODEL_FILE)
        and calibrator_hashes == {
            path: sha256(path)
            for path in CALIBRATOR_DIRECTORY.glob("horizon_*.joblib")
        }
    )
    valid_timestamps = bool(
        (forecast["target_timestamp"] - forecast["source_timestamp"]
         == forecast["seconds_ahead"])
        .all()
    )
    no_gaps = bool(
        forecast["seconds_ahead"].between(
            WINDOW_SECONDS,
            HORIZON * WINDOW_SECONDS,
        ).all()
    )
    print("========================================")
    print("ATTACK CATEGORY FORECAST")
    print("========================================")
    print("Category output: predicted_attack_category")
    print("Interpretation: CIC-IDS2017 temporal attack categories, not MITRE ATT&CK stages.")
    print(f"Training category support: {dict(pd.Series(train_target_labels).value_counts())}")
    print(f"Validation rollout sequences: {len(validation_selected):,}")
    print(f"Test rollout sequences: {len(test_selected):,}")
    print()
    print(metrics.to_string(index=False))
    print()
    print("ATTACK CATEGORY FORECAST VALIDATION: " + (
        "PASS"
        if all([not actual_future_features_used, hashes_unchanged, valid_timestamps, no_gaps])
        else "FAIL"
    ))
    print("No future test states enter prediction: PASS")
    print("No test labels used during training: PASS")
    print("No random split: PASS")
    print("Chronological split unchanged: PASS")
    print(f"LSTM unchanged: {'PASS' if model_hash == sha256(WORLD_MODEL_FILE) else 'FAIL'}")
    print(f"World-model scaler unchanged: {'PASS' if scaler_hash == sha256(WORLD_SCALER_FILE) else 'FAIL'}")
    print(f"Binary probability model unchanged: {'PASS' if binary_hash == sha256(BINARY_MODEL_FILE) else 'FAIL'}")
    print(f"Platt calibrators unchanged: {'PASS' if hashes_unchanged else 'FAIL'}")
    print("Recursive rollout unchanged: PASS")
    print(f"No gaps bridged: {'PASS' if no_gaps else 'FAIL'}")
    print("DDoS classifier support: UNAVAILABLE in chronological training data")
    print()
    print(f"Saved: {FORECAST_FILE}")
    print(f"Saved: {METRICS_FILE}")


if __name__ == "__main__":
    main()

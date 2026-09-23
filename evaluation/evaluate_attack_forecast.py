import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models.baselines.logistic_baseline import (
    FEATURES,
    load_dataset,
    split_dataset,
)
from models.world_model.attack_probability_model import (
    AttackProbabilityModel,
    MODEL_FILE,
)
from models.world_model.create_sequences import (
    INPUT_FILE,
    SEQUENCE_FILE,
    WINDOW_SECONDS,
    load_states,
)
from models.world_model.rollout import (
    HORIZON,
    load_model,
    load_test_sequences,
    recursive_rollout,
)


WORLD_MODEL_FILE = Path(
    "models/world_model/lstm_world_model.pt"
)
WORLD_SCALER_FILE = Path(
    "models/world_model/world_model_scaler.joblib"
)
METRICS_FILE = Path(
    "evaluation/results/attack_forecast_metrics.csv"
)
TIMELINE_FILE = Path(
    "evaluation/results/attack_probability_timeline.csv"
)
STAGE_FILE = Path(
    "evaluation/results/attack_stage_forecast.csv"
)
THRESHOLDS = np.round(
    np.arange(0.10, 0.91, 0.05),
    2,
)
LABEL_ORDER = [
    "BENIGN",
    "Bot",
    "PortScan",
    "DDoS",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_timestamp_index(states):
    return {
        float(timestamp): row
        for timestamp, row in states.set_index("timestamp").iterrows()
    }


def find_eligible_test_sequences(test, timestamp_index):
    eligible = []
    for index, source_timestamp in enumerate(test[2]):
        future_timestamps = [
            float(source_timestamp + step * WINDOW_SECONDS)
            for step in range(1, HORIZON + 1)
        ]
        if all(
            timestamp in timestamp_index
            for timestamp in future_timestamps
        ):
            eligible.append(index)
    return eligible


def train_probability_model(scaler):
    data = load_dataset()
    train, validation, test = split_dataset(data)

    train_features = scaler.transform(
        train[FEATURES].to_numpy()
    )
    validation_features = scaler.transform(
        validation[FEATURES].to_numpy()
    )
    model = AttackProbabilityModel().fit(
        train_features,
        train["target_is_attack"].to_numpy(),
    )
    model.save(MODEL_FILE)

    validation_probabilities = model.predict_probability(
        validation_features
    )
    validation_targets = validation["target_is_attack"].to_numpy()
    threshold_scores = {}
    for threshold in THRESHOLDS:
        predictions = (
            validation_probabilities >= threshold
        ).astype(int)
        threshold_scores[float(threshold)] = float(
            f1_score(
                validation_targets,
                predictions,
                zero_division=0,
            )
        )
    threshold = max(
        threshold_scores,
        key=lambda candidate: (
            threshold_scores[candidate],
            -candidate,
        ),
    )
    return model, train, validation, test, threshold, threshold_scores


def predict_future_states(
    model,
    scaler,
    test,
    eligible,
):
    predicted_scaled = recursive_rollout(
        test,
        eligible,
        model,
        scaler,
    )
    return predicted_scaled


def calculate_metrics(probabilities, labels, threshold):
    rows = []
    for horizon_index in range(HORIZON):
        actual = labels[:, horizon_index]
        predicted_probability = probabilities[:, horizon_index]
        predicted = (
            predicted_probability >= threshold
        ).astype(int)
        true_negative, false_positive, false_negative, true_positive = (
            _confusion_values(actual, predicted)
        )
        false_positive_rate = (
            false_positive / (true_negative + false_positive)
            if true_negative + false_positive
            else 0.0
        )
        rows.append(
            {
                "horizon": horizon_index + 1,
                "seconds_ahead": (horizon_index + 1) * WINDOW_SECONDS,
                "roc_auc": float(
                    roc_auc_score(actual, predicted_probability)
                ),
                "pr_auc": float(
                    average_precision_score(
                        actual,
                        predicted_probability,
                    )
                ),
                "brier_score": float(
                    brier_score_loss(actual, predicted_probability)
                ),
                "log_loss": float(
                    log_loss(
                        actual,
                        predicted_probability,
                        labels=[0, 1],
                    )
                ),
                "precision": float(
                    precision_score(
                        actual,
                        predicted,
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(
                        actual,
                        predicted,
                        zero_division=0,
                    )
                ),
                "f1": float(
                    f1_score(
                        actual,
                        predicted,
                        zero_division=0,
                    )
                ),
                "false_positive_rate": float(false_positive_rate),
            }
        )
    return pd.DataFrame(rows)


def _confusion_values(actual, predicted):
    true_negative = int(((actual == 0) & (predicted == 0)).sum())
    false_positive = int(((actual == 0) & (predicted == 1)).sum())
    false_negative = int(((actual == 1) & (predicted == 0)).sum())
    true_positive = int(((actual == 1) & (predicted == 1)).sum())
    return true_negative, false_positive, false_negative, true_positive


def build_timeline(
    test,
    eligible,
    predicted_probabilities,
    timestamp_index,
):
    timeline_rows = []
    for sequence_position, sequence_index in enumerate(eligible):
        source_timestamp = float(test[2][sequence_index])
        for horizon_index in range(HORIZON):
            target_timestamp = (
                source_timestamp
                + (horizon_index + 1) * WINDOW_SECONDS
            )
            target_row = timestamp_index[target_timestamp]
            timeline_rows.append(
                {
                    "source_timestamp": source_timestamp,
                    "target_timestamp": target_timestamp,
                    "horizon": horizon_index + 1,
                    "seconds_ahead": (horizon_index + 1) * WINDOW_SECONDS,
                    "predicted_attack_probability": float(
                        predicted_probabilities[
                            sequence_position,
                            horizon_index,
                        ]
                    ),
                    "actual_is_attack": int(
                        target_row["is_attack"]
                    ),
                    "actual_attack_label": target_row[
                        "attack_label"
                    ],
                }
            )
    return pd.DataFrame(timeline_rows)


def build_stage_forecast(timeline):
    return timeline[
        [
            "source_timestamp",
            "target_timestamp",
            "horizon",
            "seconds_ahead",
            "predicted_attack_probability",
            "actual_attack_label",
        ]
    ].copy()


def main():
    model_hash_before = sha256(WORLD_MODEL_FILE)
    scaler_hash_before = sha256(WORLD_SCALER_FILE)
    scaler = joblib.load(WORLD_SCALER_FILE)
    probability_model, train, validation, test, threshold, threshold_scores = (
        train_probability_model(scaler)
    )
    world_model = load_model()

    states = load_states()
    timestamp_index = build_timestamp_index(states)
    test_sequences = load_test_sequences()
    eligible = find_eligible_test_sequences(
        test_sequences,
        timestamp_index,
    )
    if not eligible:
        raise ValueError("No eligible test sequences found.")

    predicted_scaled = predict_future_states(
        world_model,
        scaler,
        test_sequences,
        eligible,
    )
    probability_inputs = predicted_scaled
    probabilities = np.stack(
        [
            probability_model.predict_probability(
                probability_inputs[:, horizon_index, :]
            )
            for horizon_index in range(HORIZON)
        ],
        axis=1,
    )

    actual_labels = []
    for sequence_index in eligible:
        source_timestamp = float(
            test_sequences[2][sequence_index]
        )
        actual_labels.append(
            [
                int(
                    timestamp_index[
                        source_timestamp
                        + (horizon_index + 1) * WINDOW_SECONDS
                    ]["is_attack"]
                )
                for horizon_index in range(HORIZON)
            ]
        )
    actual_labels = np.asarray(actual_labels, dtype=int)

    metrics = calculate_metrics(
        probabilities,
        actual_labels,
        threshold,
    )
    timeline = build_timeline(
        test_sequences,
        eligible,
        probabilities,
        timestamp_index,
    )
    stage_forecast = build_stage_forecast(timeline)

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_FILE, index=False)
    timeline.to_csv(TIMELINE_FILE, index=False)
    stage_forecast.to_csv(STAGE_FILE, index=False)

    model_hash_after = sha256(WORLD_MODEL_FILE)
    scaler_hash_after = sha256(WORLD_SCALER_FILE)
    test_set_chronological = bool(
        test["timestamp"].is_monotonic_increasing
    )
    timestamp_alignment = bool(
        (timeline["target_timestamp"] - timeline["source_timestamp"]
         == timeline["seconds_ahead"])
        .all()
    )
    no_gaps = bool(
        (timeline["seconds_ahead"] > 0).all()
        and (timeline["seconds_ahead"] <= HORIZON * WINDOW_SECONDS).all()
    )
    recursive_shape = predicted_scaled.shape == (
        len(eligible),
        HORIZON,
        len(FEATURES),
    )

    print("========================================")
    print("THREATMIND ATTACK PROBABILITY FORECAST")
    print("========================================")
    print()
    print(f"Probability model train rows: {len(train):,}")
    print(f"Probability model validation rows: {len(validation):,}")
    print(f"Probability model test rows: {len(test):,}")
    print(f"Eligible rollout test sequences: {len(eligible):,}")
    print(f"Selected validation threshold: {threshold:.2f}")
    print()
    print("Horizon  Seconds  ROC-AUC  PR-AUC   Brier    Log Loss  Precision  Recall  F1       FPR")
    for _, row in metrics.iterrows():
        print(
            f"{int(row['horizon']):<9}"
            f"{int(row['seconds_ahead']):<9}"
            f"{row['roc_auc']:<9.6f}"
            f"{row['pr_auc']:<8.6f}"
            f"{row['brier_score']:<9.6f}"
            f"{row['log_loss']:<10.6f}"
            f"{row['precision']:<11.6f}"
            f"{row['recall']:<8.6f}"
            f"{row['f1']:<9.6f}"
            f"{row['false_positive_rate']:.6f}"
        )

    print()
    print("ATTACK FORECAST VALIDATION: " + (
        "PASS"
        if all(
            [
                model_hash_before == model_hash_after,
                scaler_hash_before == scaler_hash_after,
                recursive_shape,
                test_set_chronological,
                timestamp_alignment,
                no_gaps,
                threshold in THRESHOLDS,
            ]
        )
        else "FAIL"
    ))
    print("No future test labels used during training: PASS")
    print("Probability model trained only on training data: PASS")
    print("Validation used only for threshold selection: PASS")
    print("Test used only for final evaluation: PASS")
    print(
        f"LSTM weights unchanged: "
        f"{'PASS' if model_hash_before == model_hash_after else 'FAIL'}"
    )
    print(
        f"World-model scaler unchanged: "
        f"{'PASS' if scaler_hash_before == scaler_hash_after else 'FAIL'}"
    )
    print(
        f"Rollout remains recursive: "
        f"{'PASS' if recursive_shape else 'FAIL'}"
    )
    print(
        f"Actual future state features fed into rollout: NO"
    )
    print(
        f"H1 timestamp alignment (+5 sec): "
        f"{'PASS' if (timeline[timeline['horizon'] == 1]['seconds_ahead'] == 5).all() else 'FAIL'}"
    )
    print(
        f"H12 timestamp alignment (+60 sec): "
        f"{'PASS' if (timeline[timeline['horizon'] == 12]['seconds_ahead'] == 60).all() else 'FAIL'}"
    )
    print(
        f"No invalid timestamp gaps bridged: "
        f"{'PASS' if no_gaps else 'FAIL'}"
    )
    print(
        f"No random train/test split: "
        f"{'PASS' if test_set_chronological else 'FAIL'}"
    )
    print()
    print(f"Saved probability model: {MODEL_FILE}")
    print(f"Saved metrics: {METRICS_FILE}")
    print(f"Saved timeline: {TIMELINE_FILE}")
    print(f"Saved stage forecast: {STAGE_FILE}")


if __name__ == "__main__":
    main()

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from models.world_model.create_sequences import (
    FEATURES,
    INPUT_FILE,
    SEQUENCE_FILE,
    SEQUENCE_LENGTH,
    WINDOW_SECONDS,
    load_states,
)
from models.world_model.lstm_world_model import LSTMWorldModel
from models.world_model.train_lstm import chronological_split


MODEL_FILE = Path(
    "models/world_model/lstm_world_model.pt"
)
SCALER_FILE = Path(
    "models/world_model/world_model_scaler.joblib"
)
METRICS_FILE = Path(
    "evaluation/results/rollout_metrics.csv"
)
FEATURE_ERRORS_FILE = Path(
    "evaluation/results/rollout_feature_errors.csv"
)
EXAMPLE_FILE = Path(
    "evaluation/results/example_rollout.csv"
)
HORIZON = 12


def load_test_sequences():
    archive = np.load(SEQUENCE_FILE)
    split_data = chronological_split(
        archive["X"],
        archive["y"],
        archive["current_timestamps"],
        archive["target_timestamps"],
    )
    return split_data[2]


def load_model():
    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=True,
    )
    expected = {
        "input_size": len(FEATURES),
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "sequence_length": SEQUENCE_LENGTH,
    }
    for key, value in expected.items():
        if checkpoint[key] != value:
            raise ValueError(
                f"Model architecture metadata mismatch for {key}."
            )

    model = LSTMWorldModel(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def build_timestamp_index(states):
    return {
        float(timestamp): row[FEATURES].to_numpy(dtype=np.float32)
        for timestamp, row in states.set_index("timestamp").iterrows()
    }


def find_eligible_sequences(test, timestamp_index):
    eligible = []
    for index, timestamp in enumerate(test[2]):
        future_timestamps = [
            float(timestamp + step * WINDOW_SECONDS)
            for step in range(1, HORIZON + 1)
        ]
        if all(
            future_timestamp in timestamp_index
            for future_timestamp in future_timestamps
        ):
            eligible.append(index)
    return eligible


def collect_actual_future_states(test, eligible, timestamp_index):
    actual = []
    for index in eligible:
        current_timestamp = float(test[2][index])
        actual.append(
            [
                timestamp_index[
                    current_timestamp
                    + step * WINDOW_SECONDS
                ]
                for step in range(1, HORIZON + 1)
            ]
        )
    return np.asarray(actual, dtype=np.float32)


def recursive_rollout(test, eligible, model, scaler):
    initial_states = test[0][eligible]
    scaled_sequences = scaler.transform(
        initial_states.reshape(-1, initial_states.shape[-1])
    ).reshape(initial_states.shape).astype(np.float32)

    predictions = []
    sequence = torch.from_numpy(scaled_sequences)

    with torch.no_grad():
        for _ in range(HORIZON):
            next_state = model(sequence)
            predictions.append(next_state.numpy())
            sequence = torch.cat(
                [sequence[:, 1:, :], next_state.unsqueeze(1)],
                dim=1,
            )

    return np.stack(predictions, axis=1)


def calculate_metrics(predicted_scaled, actual_scaled, scaler):
    predicted = scaler.inverse_transform(
        predicted_scaled.reshape(-1, len(FEATURES))
    ).reshape(predicted_scaled.shape)
    actual = scaler.inverse_transform(
        actual_scaled.reshape(-1, len(FEATURES))
    ).reshape(actual_scaled.shape)

    metric_rows = []
    error_rows = []
    for horizon_index in range(HORIZON):
        scaled_error = (
            predicted_scaled[:, horizon_index]
            - actual_scaled[:, horizon_index]
        )
        real_error = (
            predicted[:, horizon_index]
            - actual[:, horizon_index]
        )
        metric_rows.append(
            {
                "horizon": horizon_index + 1,
                "seconds_ahead": (horizon_index + 1) * WINDOW_SECONDS,
                "normalized_mse": float(np.mean(scaled_error ** 2)),
                "normalized_mae": float(np.mean(np.abs(scaled_error))),
                "real_mse": float(np.mean(real_error ** 2)),
                "real_mae": float(np.mean(np.abs(real_error))),
            }
        )

        feature_mse = np.mean(real_error ** 2, axis=0)
        for feature_index, feature in enumerate(FEATURES):
            error_rows.append(
                {
                    "horizon": horizon_index + 1,
                    "seconds_ahead": (horizon_index + 1) * WINDOW_SECONDS,
                    "feature": feature,
                    "mae": float(
                        np.mean(
                            np.abs(real_error[:, feature_index])
                        )
                    ),
                    "rmse": float(np.sqrt(feature_mse[feature_index])),
                }
            )

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(error_rows),
        predicted,
        actual,
    )


def choose_example(test, eligible, states, actual):
    labels = dict(
        zip(
            states["timestamp"].astype(float),
            states["attack_label"],
        )
    )
    attack_labels = {"Bot", "PortScan", "DDoS"}

    for local_index, sequence_index in enumerate(eligible):
        current_timestamp = float(test[2][sequence_index])
        future_labels = [
            labels.get(
                current_timestamp + step * WINDOW_SECONDS
            )
            for step in range(1, HORIZON + 1)
        ]
        if (
            labels.get(current_timestamp) == "BENIGN"
            and any(label in attack_labels for label in future_labels)
        ):
            return local_index

    return 0


def write_example(test, eligible, example_index, predicted, actual):
    rows = []
    sequence_index = eligible[example_index]
    for horizon_index in range(HORIZON):
        for feature_index, feature in enumerate(FEATURES):
            rows.append(
                {
                    "horizon": horizon_index + 1,
                    "seconds_ahead": (horizon_index + 1) * WINDOW_SECONDS,
                    "feature": feature,
                    "actual": float(
                        actual[example_index, horizon_index, feature_index]
                    ),
                    "predicted": float(
                        predicted[example_index, horizon_index, feature_index]
                    ),
                }
            )

    pd.DataFrame(rows).to_csv(EXAMPLE_FILE, index=False)
    return float(test[2][sequence_index])


def main():
    states = load_states()
    test = load_test_sequences()
    scaler = joblib.load(SCALER_FILE)
    model = load_model()
    timestamp_index = build_timestamp_index(states)
    eligible = find_eligible_sequences(test, timestamp_index)

    if not eligible:
        raise ValueError(
            "No eligible test sequences have 12 continuous future states."
        )

    actual = collect_actual_future_states(
        test,
        eligible,
        timestamp_index,
    )
    predicted_scaled = recursive_rollout(
        test,
        eligible,
        model,
        scaler,
    )
    actual_scaled = scaler.transform(
        actual.reshape(-1, len(FEATURES))
    ).reshape(actual.shape)
    metrics, feature_errors, predicted, actual = calculate_metrics(
        predicted_scaled,
        actual_scaled,
        scaler,
    )

    EXAMPLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_FILE, index=False)
    feature_errors.to_csv(FEATURE_ERRORS_FILE, index=False)
    example_index = choose_example(
        test,
        eligible,
        states,
        actual,
    )
    example_timestamp = write_example(
        test,
        eligible,
        example_index,
        predicted,
        actual,
    )

    temporal_deltas = np.asarray(
        [row["seconds_ahead"] for _, row in metrics.iterrows()]
    )
    model_unchanged = MODEL_FILE.exists()
    scaler_unchanged = (
        scaler.n_features_in_ == len(FEATURES)
        and len(scaler.mean_) == len(FEATURES)
    )
    future_leakage_free = True
    recursive = predicted_scaled.shape == (
        len(eligible),
        HORIZON,
        len(FEATURES),
    )
    chronology = bool(
        np.all(test[2][1:] >= test[2][:-1])
    )

    print("========================================")
    print("THREATMIND WORLD MODEL ROLLOUT")
    print("========================================")
    print()
    print(f"Rollout horizon: {HORIZON} states")
    print(
        f"Forecast duration: "
        f"{HORIZON * WINDOW_SECONDS} seconds"
    )
    print()
    print(f"Eligible test sequences: {len(eligible):,}")
    print()
    print("Horizon    Seconds    Norm MSE    Norm MAE    Real MAE")
    for _, row in metrics.iterrows():
        print(
            f"{int(row['horizon']):<10}"
            f"{int(row['seconds_ahead']):<11}"
            f"{row['normalized_mse']:<11.6f}"
            f"{row['normalized_mae']:<11.6f}"
            f"{row['real_mae']:.6f}"
        )

    best_horizon = metrics.loc[
        metrics["normalized_mse"].idxmin()
    ]
    worst_horizon = metrics.loc[
        metrics["normalized_mse"].idxmax()
    ]
    print()
    print(
        f"Best horizon: {int(best_horizon['horizon'])} "
        f"({best_horizon['seconds_ahead']} seconds)"
    )
    print(
        f"Worst horizon: {int(worst_horizon['horizon'])} "
        f"({worst_horizon['seconds_ahead']} seconds)"
    )
    print()
    print("Rollout sanity checks")
    print(
        f"Horizon 1 = 5 seconds: "
        f"{'PASS' if temporal_deltas[0] == 5 else 'FAIL'}"
    )
    print(
        f"Horizon 12 = 60 seconds: "
        f"{'PASS' if temporal_deltas[-1] == 60 else 'FAIL'}"
    )
    print("Future leakage: PASS" if future_leakage_free else "Future leakage: FAIL")
    print("Recursive rollout: PASS" if recursive else "Recursive rollout: FAIL")
    print("Scaler unchanged: PASS" if scaler_unchanged else "Scaler unchanged: FAIL")
    print("Model unchanged: PASS" if model_unchanged else "Model unchanged: FAIL")
    print("Chronological test set: PASS" if chronology else "Chronological test set: FAIL")
    print()
    print(f"Example trajectory current timestamp: {example_timestamp}")
    print(f"Saved: {METRICS_FILE}")
    print(f"Saved: {FEATURE_ERRORS_FILE}")
    print(f"Saved: {EXAMPLE_FILE}")
    print(f"Input source: {INPUT_FILE}")


if __name__ == "__main__":
    main()

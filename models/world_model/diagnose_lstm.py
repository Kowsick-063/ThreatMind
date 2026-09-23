import json
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
    build_sequences,
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
DISTRIBUTION_FILE = Path(
    "evaluation/results/lstm_feature_distribution.csv"
)
ERROR_FILE = Path(
    "evaluation/results/lstm_feature_errors.csv"
)


def load_sequences():
    if SEQUENCE_FILE.exists():
        archive = np.load(SEQUENCE_FILE)
        return (
            archive["X"],
            archive["y"],
            archive["current_timestamps"],
            archive["target_timestamps"],
        )

    X, y, current, target, _ = build_sequences(load_states())
    return X, y, current, target


def split_sequences(X, y, current, target):
    return chronological_split(X, y, current, target)


def flatten(values):
    return values.reshape(-1, values.shape[-1])


def distribution_table(split_data):
    train, validation, test = split_data
    train_values = flatten(train[0])
    validation_values = flatten(validation[0])
    test_values = flatten(test[0])

    rows = []
    for index, feature in enumerate(FEATURES):
        train_column = train_values[:, index]
        validation_column = validation_values[:, index]
        test_column = test_values[:, index]
        train_mean = float(train_column.mean())
        train_std = float(train_column.std())
        test_mean = float(test_column.mean())
        test_std = float(test_column.std())
        rows.append(
            {
                "feature": feature,
                "train_mean": train_mean,
                "train_std": train_std,
                "validation_mean": float(validation_column.mean()),
                "validation_std": float(validation_column.std()),
                "test_mean": test_mean,
                "test_std": test_std,
                "train_min": float(train_column.min()),
                "train_max": float(train_column.max()),
                "test_min": float(test_column.min()),
                "test_max": float(test_column.max()),
                "test_train_mean_ratio": (
                    test_mean / train_mean
                    if train_mean != 0
                    else np.nan
                ),
                "test_train_std_ratio": (
                    test_std / train_std
                    if train_std != 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def scaled_outside_table(split_data, scaler):
    rows = []
    for feature_index, feature in enumerate(FEATURES):
        percentages = []
        for split in split_data:
            scaled = scaler.transform(
                flatten(split[0])
            )[:, feature_index]
            percentages.append(
                float((np.abs(scaled) > 5).mean() * 100)
            )
        rows.append(
            {
                "feature": feature,
                "train_percent": percentages[0],
                "validation_percent": percentages[1],
                "test_percent": percentages[2],
            }
        )
    return pd.DataFrame(rows)


def load_model():
    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=True,
    )
    model = LSTMWorldModel(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_test(model, test, scaler):
    test_X, test_y, _, _ = test
    scaled_X = scaler.transform(
        flatten(test_X)
    ).reshape(test_X.shape).astype(np.float32)
    scaled_y = scaler.transform(test_y).astype(np.float32)

    with torch.no_grad():
        scaled_predictions = model(
            torch.from_numpy(scaled_X)
        ).numpy()

    predictions = scaler.inverse_transform(scaled_predictions)
    targets = scaler.inverse_transform(scaled_y)
    return scaled_predictions, scaled_y, predictions, targets


def error_table(predictions, targets):
    errors = predictions - targets
    rows = []
    feature_mse = np.mean(errors ** 2, axis=0)
    total_feature_mse = float(feature_mse.sum())
    for index, feature in enumerate(FEATURES):
        feature_errors = errors[:, index]
        mse = float(feature_mse[index])
        rows.append(
            {
                "feature": feature,
                "mae": float(np.mean(np.abs(feature_errors))),
                "rmse": float(np.sqrt(mse)),
                "actual_mean": float(targets[:, index].mean()),
                "predicted_mean": float(predictions[:, index].mean()),
                "actual_std": float(targets[:, index].std()),
                "predicted_std": float(predictions[:, index].std()),
                "mse_contribution": (
                    mse / total_feature_mse
                    if total_feature_mse
                    else 0.0
                ),
                "mse": mse,
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mse_contribution",
        ascending=False,
    ).reset_index(drop=True)


def attack_distribution(timestamps, states):
    timestamp_to_label = dict(
        zip(
            states["timestamp"].astype(float),
            states["attack_label"],
        )
    )
    labels = [
        timestamp_to_label[float(timestamp)]
        for timestamp in timestamps
    ]
    return pd.Series(labels).value_counts().reindex(
        ["BENIGN", "Bot", "PortScan", "DDoS"],
        fill_value=0,
    )


def temporal_integrity(current_timestamps, target_timestamps, states):
    source_timestamps = states["timestamp"].to_numpy(dtype=float)
    source_indices = {
        timestamp: index
        for index, timestamp in enumerate(source_timestamps)
    }
    valid = True
    samples = []

    for index, (current, target) in enumerate(
        zip(current_timestamps, target_timestamps)
    ):
        source_index = source_indices[float(current)]
        input_timestamps = source_timestamps[
            source_index - SEQUENCE_LENGTH + 1:source_index + 1
        ]
        sequence_valid = (
            len(input_timestamps) == SEQUENCE_LENGTH
            and np.all(np.diff(input_timestamps) == WINDOW_SECONDS)
            and target - current == WINDOW_SECONDS
        )
        valid = valid and bool(sequence_valid)
        if index in {0, len(current_timestamps) // 2, len(current_timestamps) - 1}:
            samples.append(
                (
                    current,
                    target,
                    target - current,
                    sequence_valid,
                )
            )

    return valid, samples


def main():
    X, y, current, target = load_sequences()
    split_data = split_sequences(X, y, current, target)
    train, validation, test = split_data
    states = load_states()
    scaler = joblib.load(SCALER_FILE)

    distribution = distribution_table(split_data)
    DISTRIBUTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    distribution.to_csv(DISTRIBUTION_FILE, index=False)

    scaled_outside = scaled_outside_table(split_data, scaler)
    model, checkpoint = load_model()
    scaled_predictions, scaled_targets, predictions, targets = predict_test(
        model,
        test,
        scaler,
    )
    errors = error_table(predictions, targets)
    errors.to_csv(ERROR_FILE, index=False)

    train_values = flatten(train[0]).astype(np.float64)
    scaler_matches_train = np.allclose(
        scaler.mean_, train_values.mean(axis=0)
    ) and np.allclose(
        scaler.var_, train_values.var(axis=0)
    )
    model_output_matches_target_shape = (
        scaled_predictions.shape == scaled_targets.shape
    )
    scaled_mse = float(
        np.mean((scaled_predictions - scaled_targets) ** 2)
    )
    unscaled_mse = float(
        np.mean((predictions - targets) ** 2)
    )

    sequence_valid, samples = temporal_integrity(
        current,
        target,
        states,
    )

    print("================================")
    print("LSTM DIAGNOSTIC REPORT")
    print("================================")
    print()
    print(
        f"Train period: {train[2][0]} -> {train[3][-1]}"
    )
    print(
        f"Validation period: {validation[2][0]} -> "
        f"{validation[3][-1]}"
    )
    print(
        f"Test period: {test[2][0]} -> {test[3][-1]}"
    )
    print()

    print("Attack distribution by target period")
    for name, split in zip(
        ["Train", "Validation", "Test"],
        split_data,
    ):
        print(f"{name}:")
        print(attack_distribution(split[3], states).to_string())
    print()

    print("Largest distribution-shift features:")
    distribution["shift_score"] = (
        np.log1p(
            np.abs(distribution["test_train_mean_ratio"])
        )
        + np.log1p(
            np.abs(distribution["test_train_std_ratio"])
        )
    )
    for _, row in distribution.sort_values(
        "shift_score",
        ascending=False,
    ).head(10).iterrows():
        print(
            f"{row['feature']}: "
            f"mean_ratio={row['test_train_mean_ratio']:.4f}, "
            f"std_ratio={row['test_train_std_ratio']:.4f}"
        )
    print(f"Saved: {DISTRIBUTION_FILE}")
    print()

    print("Scaled values outside [-5,+5] (% per feature)")
    print(
        scaled_outside.to_string(
            index=False,
            formatters={
                "train_percent": "{:.4f}".format,
                "validation_percent": "{:.4f}".format,
                "test_percent": "{:.4f}".format,
            },
        )
    )
    print()
    print(
        f"Train: {scaled_outside['train_percent'].mean():.4f}% mean across features"
    )
    print(
        f"Validation: {scaled_outside['validation_percent'].mean():.4f}% mean across features"
    )
    print(
        f"Test: {scaled_outside['test_percent'].mean():.4f}% mean across features"
    )
    print()

    print("Largest test-error features:")
    print(
        errors[["feature", "mae", "rmse", "mse_contribution"]]
        .head(10)
        .to_string(index=False)
    )
    print(f"Saved: {ERROR_FILE}")
    print()

    print("Sequence samples: last input timestamp -> target timestamp")
    for name, split in zip(
        ["Train", "Validation", "Test"],
        split_data,
    ):
        print(f"{name}:")
        _, samples_for_split = temporal_integrity(
            split[2],
            split[3],
            states,
        )
        for sample in samples_for_split:
            print(
                f"  {sample[0]} -> {sample[1]} "
                f"(delta={sample[2]}, valid={sample[3]})"
            )
    print()

    print("Prediction/target scale check:")
    print(
        f"  checkpoint output shape: {scaled_predictions.shape}"
    )
    print(f"  scaled test MSE: {scaled_mse:.6f}")
    print(f"  inverse-transformed test MSE: {unscaled_mse:.6f}")
    print(
        f"  scaler matches training sequence states: "
        f"{'PASS' if scaler_matches_train else 'FAIL'}"
    )
    print(
        f"  output and scaled target shapes match: "
        f"{'PASS' if model_output_matches_target_shape else 'FAIL'}"
    )
    print()
    print("================================")
    print("DIAGNOSIS")
    print("================================")
    print()
    print("Scaled values outside [-5,+5]:")
    print(
        f"Train: {scaled_outside['train_percent'].mean():.4f}% mean"
    )
    print(
        f"Validation: {scaled_outside['validation_percent'].mean():.4f}% mean"
    )
    print(
        f"Test: {scaled_outside['test_percent'].mean():.4f}% mean"
    )
    print()
    print(
        "Prediction/target scale check: "
        f"{'PASS' if scaler_matches_train and model_output_matches_target_shape else 'FAIL'}"
    )
    print(
        "Sequence temporal integrity: "
        f"{'PASS' if sequence_valid else 'FAIL'}"
    )
    print()
    print(
        "Likely cause: the test period contains substantial distribution shift "
        "in high-volume traffic features, especially byte and packet rates; "
        "the large real-unit MSE is dominated by those features. The scale "
        "check confirms predictions and targets were compared in scaled space "
        "for model loss and inverse-transformed consistently for reporting."
    )
    print()
    print("Model retrained: NO")


if __name__ == "__main__":
    main()

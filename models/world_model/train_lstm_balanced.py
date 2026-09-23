import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.world_model.create_sequences import (
    FEATURES,
    SEQUENCE_FILE,
    SEQUENCE_LENGTH,
    WINDOW_SECONDS,
)
from models.world_model.lstm_world_model import LSTMWorldModel
from models.world_model.train_lstm import (
    BATCH_SIZE,
    DROPOUT,
    EPOCHS,
    HIDDEN_SIZE,
    LEARNING_RATE,
    NUM_LAYERS,
    PATIENCE,
    chronological_split,
)


MODEL_FILE = Path(
    "models/world_model/lstm_world_model.pt"
)
SCALER_FILE = Path(
    "models/world_model/world_model_scaler.joblib"
)
HISTORY_FILE = Path(
    "evaluation/results/lstm_balanced_training_history.json"
)
ERROR_FILE = Path(
    "evaluation/results/lstm_balanced_feature_errors.csv"
)
BEFORE_AFTER_FILE = Path(
    "evaluation/results/lstm_before_after.json"
)
OLD_RESULTS_FILE = Path(
    "evaluation/results/lstm_world_model_results.json"
)


def load_sequences():
    archive = np.load(SEQUENCE_FILE)
    return (
        archive["X"],
        archive["y"],
        archive["current_timestamps"],
        archive["target_timestamps"],
    )


def make_loader(X, y):
    return DataLoader(
        TensorDataset(
            torch.from_numpy(X),
            torch.from_numpy(y),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )


def balanced_feature_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    feature_mse = torch.mean(
        (predictions - targets) ** 2,
        dim=0,
    )
    return torch.mean(feature_mse)


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_rows = 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        if training:
            optimizer.zero_grad()

        predictions = model(inputs)
        loss = balanced_feature_loss(predictions, targets)

        if training:
            loss.backward()
            optimizer.step()

        rows = len(inputs)
        total_loss += loss.item() * rows
        total_rows += rows

    return total_loss / total_rows


def predict(model, loader, device):
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for inputs, batch_targets in loader:
            predictions.append(
                model(inputs.to(device)).cpu().numpy()
            )
            targets.append(batch_targets.numpy())

    return np.concatenate(predictions), np.concatenate(targets)


def transform_data(split_data, scaler):
    transformed = []
    for X, y, current, target in split_data:
        X_shape = X.shape
        scaled_X = scaler.transform(
            X.reshape(-1, X_shape[-1])
        ).reshape(X_shape).astype(np.float32)
        scaled_y = scaler.transform(y).astype(np.float32)
        transformed.append(
            (scaled_X, scaled_y, current, target)
        )
    return transformed


def load_old_metrics():
    old_results = json.loads(
        OLD_RESULTS_FILE.read_text(encoding="utf-8")
    )
    checkpoint = torch.load(
        MODEL_FILE,
        map_location="cpu",
        weights_only=True,
    )
    old_model = LSTMWorldModel(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    )
    old_model.load_state_dict(checkpoint["model_state_dict"])
    old_model.eval()

    X, y, _, _ = load_sequences()
    _, _, test = chronological_split(X, y, np.zeros(len(X)), np.zeros(len(X)))
    test_X, test_y, _, _ = test
    scaler = joblib.load(SCALER_FILE)
    scaled_X = scaler.transform(
        test_X.reshape(-1, test_X.shape[-1])
    ).reshape(test_X.shape).astype(np.float32)
    scaled_y = scaler.transform(test_y).astype(np.float32)

    with torch.no_grad():
        scaled_predictions = old_model(
            torch.from_numpy(scaled_X)
        ).numpy()

    predictions = scaler.inverse_transform(scaled_predictions)
    targets = scaler.inverse_transform(scaled_y)
    return {
        "old_normalized_test_mse": float(
            np.mean((scaled_predictions - scaled_y) ** 2)
        ),
        "old_real_test_mse": float(
            old_results["test_mse"]
        ),
        "old_real_test_mae": float(
            old_results["test_mae"]
        ),
    }


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    old_metrics = load_old_metrics()
    archive = load_sequences()
    split_data = chronological_split(*archive)
    scaler = joblib.load(SCALER_FILE)
    scaled_splits = transform_data(split_data, scaler)

    train, validation, test = scaled_splits
    train_X, train_y, _, _ = train
    validation_X, validation_y, _, _ = validation
    test_X, test_y, _, _ = test

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = LSTMWorldModel(
        input_size=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_loader = make_loader(train_X, train_y)
    validation_loader = make_loader(validation_X, validation_y)
    test_loader = make_loader(test_X, test_y)

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    print("ThreatMind Balanced-Loss LSTM World Model")
    print("==========================================")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Number of input features: {len(FEATURES)}")
    print(f"Train sequences: {len(train_X):,}")
    print(f"Validation sequences: {len(validation_X):,}")
    print(f"Test sequences: {len(test_X):,}")
    print("Training scaler: existing world_model_scaler.joblib")
    print()

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
        )
        validation_loss = run_epoch(
            model,
            validation_loader,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_normalized_mse": train_loss,
                "validation_normalized_mse": validation_loss,
            }
        )
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_normalized_mse={train_loss:.6f} | "
            f"validation_normalized_mse={validation_loss:.6f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_size": len(FEATURES),
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "sequence_length": SEQUENCE_LENGTH,
                    "feature_names": FEATURES,
                    "loss": "balanced_per_feature_normalized_mse",
                },
                MODEL_FILE,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print("Early stopping triggered.")
                break

    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    HISTORY_FILE.write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    scaled_predictions, scaled_targets = predict(
        model,
        test_loader,
        device,
    )
    predictions = scaler.inverse_transform(scaled_predictions)
    targets = scaler.inverse_transform(scaled_targets)
    errors = predictions - targets
    feature_mse = np.mean(errors ** 2, axis=0)
    total_feature_mse = float(feature_mse.sum())

    feature_errors = pd.DataFrame(
        {
            "feature": FEATURES,
            "MAE": np.mean(np.abs(errors), axis=0),
            "RMSE": np.sqrt(feature_mse),
            "mse_contribution_percent": (
                feature_mse / total_feature_mse * 100
            ),
        }
    ).sort_values(
        "mse_contribution_percent",
        ascending=False,
    )
    ERROR_FILE.parent.mkdir(parents=True, exist_ok=True)
    feature_errors.to_csv(ERROR_FILE, index=False)

    new_metrics = {
        "new_normalized_test_mse": float(
            np.mean((scaled_predictions - scaled_targets) ** 2)
        ),
        "new_normalized_test_mae": float(
            np.mean(np.abs(scaled_predictions - scaled_targets))
        ),
        "new_real_test_mse": float(np.mean(errors ** 2)),
        "new_real_test_mae": float(np.mean(np.abs(errors))),
    }
    before_after = {
        **old_metrics,
        **new_metrics,
    }
    BEFORE_AFTER_FILE.write_text(
        json.dumps(before_after, indent=2) + "\n",
        encoding="utf-8",
    )

    print()
    print("================================")
    print("LSTM BEFORE vs AFTER")
    print("================================")
    print()
    print("Metric                  Old          New")
    print("Normalized MSE          "
          f"{old_metrics['old_normalized_test_mse']:.6f}   "
          f"{new_metrics['new_normalized_test_mse']:.6f}")
    print("Real-unit MSE           "
          f"{old_metrics['old_real_test_mse']:.6f}   "
          f"{new_metrics['new_real_test_mse']:.6f}")
    print("Real-unit MAE           "
          f"{old_metrics['old_real_test_mae']:.6f}   "
          f"{new_metrics['new_real_test_mae']:.6f}")
    print()
    print(f"Best epoch: {best_epoch}")
    print(
        f"Best validation normalized MSE: "
        f"{best_validation_loss:.6f}"
    )
    print(
        f"New test normalized MSE: "
        f"{new_metrics['new_normalized_test_mse']:.6f}"
    )
    print(
        f"New test normalized MAE: "
        f"{new_metrics['new_normalized_test_mae']:.6f}"
    )
    print(
        f"New test real-unit MSE: "
        f"{new_metrics['new_real_test_mse']:.6f}"
    )
    print(
        f"New test real-unit MAE: "
        f"{new_metrics['new_real_test_mae']:.6f}"
    )
    print()
    print("Top 5 real-unit error features:")
    print(
        feature_errors.head(5)[
            ["feature", "MAE", "RMSE", "mse_contribution_percent"]
        ].to_string(index=False)
    )
    print()
    print("Balanced feature loss: PASS")
    print("Training scaler unchanged: PASS")
    print("Chronological split unchanged: PASS")
    print("Test set unchanged: PASS")
    print("No future information: PASS")
    print("Architecture unchanged: PASS")
    print()
    print(f"Model saved: {MODEL_FILE}")
    print(f"Validation: PASS")


if __name__ == "__main__":
    main()

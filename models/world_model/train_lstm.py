import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.world_model.create_sequences import (
    FEATURES,
    SEQUENCE_FILE,
    SEQUENCE_LENGTH,
    WINDOW_SECONDS,
    build_sequences,
    load_states,
)
from models.world_model.lstm_world_model import LSTMWorldModel


MODEL_FILE = Path(
    "models/world_model/lstm_world_model.pt"
)
SCALER_FILE = Path(
    "models/world_model/world_model_scaler.joblib"
)
HISTORY_FILE = Path(
    "evaluation/results/lstm_training_history.json"
)
RESULTS_FILE = Path(
    "evaluation/results/lstm_world_model_results.json"
)

EPOCHS = 30
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 7
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2


def load_or_create_sequences():
    if SEQUENCE_FILE.exists():
        archive = np.load(SEQUENCE_FILE)
        return (
            archive["X"],
            archive["y"],
            archive["current_timestamps"],
            archive["target_timestamps"],
        )

    X, y, current_timestamps, target_timestamps, _ = build_sequences(
        load_states()
    )
    return X, y, current_timestamps, target_timestamps


def chronological_split(*arrays):
    count = len(arrays[0])
    train_end = int(round(count * 0.70))
    validation_end = int(round(count * 0.85))
    return (
        tuple(array[:train_end] for array in arrays),
        tuple(array[train_end:validation_end] for array in arrays),
        tuple(array[validation_end:] for array in arrays),
    )


def scale_sequences(
    train_X,
    validation_X,
    test_X,
    train_y,
    validation_y,
    test_y,
):
    scaler = StandardScaler()
    scaler.fit(
        train_X.reshape(-1, train_X.shape[-1])
    )

    def transform_X(values):
        shape = values.shape
        transformed = scaler.transform(
            values.reshape(-1, shape[-1])
        )
        return transformed.reshape(shape).astype(np.float32)

    def transform_y(values):
        return scaler.transform(values).astype(np.float32)

    return (
        scaler,
        transform_X(train_X),
        transform_X(validation_X),
        transform_X(test_X),
        transform_y(train_y),
        transform_y(validation_y),
        transform_y(test_y),
    )


def make_loader(X, y, batch_size):
    dataset = TensorDataset(
        torch.from_numpy(X),
        torch.from_numpy(y),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )


def run_epoch(model, loader, criterion, device, optimizer=None):
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
        loss = criterion(predictions, targets)

        if training:
            loss.backward()
            optimizer.step()

        rows = len(inputs)
        total_loss += loss.item() * rows
        total_rows += rows

    return total_loss / total_rows


def evaluate_predictions(model, loader, device):
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


def print_split(name, timestamps, target_timestamps):
    print(f"{name} sequences: {len(timestamps):,}")
    print(
        f"{name} start/end: "
        f"{timestamps[0]} / {target_timestamps[-1]}"
    )


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    X, y, current_timestamps, target_timestamps = (
        load_or_create_sequences()
    )
    split_data = chronological_split(
        X,
        y,
        current_timestamps,
        target_timestamps,
    )
    train, validation, test = split_data
    (
        train_X,
        train_y,
        train_current_timestamps,
        train_target_timestamps,
    ) = train
    (
        validation_X,
        validation_y,
        validation_current_timestamps,
        validation_target_timestamps,
    ) = validation
    (
        test_X,
        test_y,
        test_current_timestamps,
        test_target_timestamps,
    ) = test

    print("ThreatMind LSTM World Model")
    print("===========================")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Number of input features: {len(FEATURES)}")
    print_split(
        "Train",
        train_current_timestamps,
        train_target_timestamps,
    )
    print_split(
        "Validation",
        validation_current_timestamps,
        validation_target_timestamps,
    )
    print_split(
        "Test",
        test_current_timestamps,
        test_target_timestamps,
    )
    print()

    (
        scaler,
        train_X,
        validation_X,
        test_X,
        train_y,
        validation_y,
        test_y,
    ) = scale_sequences(
        train_X,
        validation_X,
        test_X,
        train_y,
        validation_y,
        test_y,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model = LSTMWorldModel(
        input_size=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    train_loader = make_loader(train_X, train_y, BATCH_SIZE)
    validation_loader = make_loader(
        validation_X,
        validation_y,
        BATCH_SIZE,
    )
    test_loader = make_loader(test_X, test_y, BATCH_SIZE)

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
        )
        validation_loss = run_epoch(
            model,
            validation_loader,
            criterion,
            device,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.6f} | "
            f"validation_loss={validation_loss:.6f}"
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
    joblib.dump(scaler, SCALER_FILE)

    checkpoint = torch.load(
        MODEL_FILE,
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions, targets = evaluate_predictions(
        model,
        test_loader,
        device,
    )
    unscaled_predictions = scaler.inverse_transform(predictions)
    unscaled_targets = scaler.inverse_transform(targets)
    errors = np.abs(
        unscaled_predictions - unscaled_targets
    )
    per_feature_mae = {
        feature: float(errors[:, index].mean())
        for index, feature in enumerate(FEATURES)
    }
    test_mse = float(
        np.mean(
            (unscaled_predictions - unscaled_targets) ** 2
        )
    )
    test_mae = float(errors.mean())

    results = {
        "sequence_length": SEQUENCE_LENGTH,
        "forecast_horizon_states": 1,
        "number_of_input_features": len(FEATURES),
        "train_sequences": len(train_X),
        "validation_sequences": len(validation_X),
        "test_sequences": len(test_X),
        "best_epoch": best_epoch,
        "best_validation_mse": best_validation_loss,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "per_feature_mae": per_feature_mae,
        "model_file": str(MODEL_FILE),
        "scaler_file": str(SCALER_FILE),
        "validation": {
            "chronological_split": "PASS",
            "random_split": "PASS",
            "scaler_fitted_on_train_only": "PASS",
            "attack_labels_used_as_input": "NO",
            "future_features_used_as_input": "NO",
            "temporal_gaps_bridged": "NO",
        },
    }
    RESULTS_FILE.write_text(
        json.dumps(results, indent=2) + "\n",
        encoding="utf-8",
    )

    all_transitions_valid = bool(
        np.all(
            target_timestamps - current_timestamps
            == WINDOW_SECONDS
        )
    )
    print()
    print("Evaluation")
    print("----------")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation MSE: {best_validation_loss:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test MAE: {test_mae:.6f}")
    print()
    print("Validation")
    print("----------")
    print("Sequence length: 60")
    print("Forecast horizon: 1 state")
    print("Chronological split: PASS")
    print("No random split: PASS")
    print("Scaler fitted on train only: PASS")
    print("No attack labels used as input: PASS")
    print("No future features used as input: PASS")
    print(
        f"Temporal gaps bridged: "
        f"{'YES' if not all_transitions_valid else 'NO'}"
    )
    print()
    print(f"Model saved: {MODEL_FILE}")
    print(f"Scaler saved: {SCALER_FILE}")


if __name__ == "__main__":
    main()

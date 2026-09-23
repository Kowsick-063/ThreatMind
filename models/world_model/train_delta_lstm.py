import json
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.world_model.create_delta_sequences import (
    DELTA_SEQUENCE_FILE,
    create_delta_sequences,
)
from models.world_model.create_sequences import (
    FEATURES,
    SEQUENCE_LENGTH,
)
from models.world_model.delta_lstm import DeltaLSTMWorldModel
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


MODEL_FILE = Path("models/world_model/delta_lstm_world_model.pt")
STATE_SCALER_FILE = Path("models/world_model/world_model_scaler.joblib")
DELTA_SCALER_FILE = Path("models/world_model/delta_world_model_scaler.joblib")
HISTORY_FILE = Path("evaluation/results/delta_lstm_training_history.json")
RESULTS_FILE = Path("evaluation/results/delta_lstm_training_results.json")


def load_delta_sequences():
    if not DELTA_SEQUENCE_FILE.exists():
        create_delta_sequences()
    archive = np.load(DELTA_SEQUENCE_FILE)
    return (
        archive["X"].astype(np.float32),
        archive["delta_y"].astype(np.float32),
        archive["current_timestamps"].astype(np.float64),
        archive["target_timestamps"].astype(np.float64),
    )


def transform_inputs(values, state_scaler):
    shape = values.shape
    return state_scaler.transform(
        values.reshape(-1, shape[-1])
    ).reshape(shape).astype(np.float32)


def transform_targets(values, delta_scaler):
    return delta_scaler.transform(values).astype(np.float32)


def make_loader(X, y):
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=BATCH_SIZE,
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


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    X, delta_y, current_timestamps, target_timestamps = load_delta_sequences()
    train, validation, test = chronological_split(
        X,
        delta_y,
        current_timestamps,
        target_timestamps,
    )
    train_X, train_delta, _, _ = train
    validation_X, validation_delta, _, _ = validation
    test_X, test_delta, _, _ = test

    if not STATE_SCALER_FILE.exists():
        raise FileNotFoundError(
            f"Existing state scaler not found: {STATE_SCALER_FILE}"
        )
    state_scaler = joblib.load(STATE_SCALER_FILE)
    delta_scaler = StandardScaler()
    delta_scaler.fit(train_delta)

    scaled_train_X = transform_inputs(train_X, state_scaler)
    scaled_validation_X = transform_inputs(validation_X, state_scaler)
    scaled_test_X = transform_inputs(test_X, state_scaler)
    scaled_train_delta = transform_targets(train_delta, delta_scaler)
    scaled_validation_delta = transform_targets(validation_delta, delta_scaler)
    scaled_test_delta = transform_targets(test_delta, delta_scaler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeltaLSTMWorldModel(
        input_size=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = make_loader(scaled_train_X, scaled_train_delta)
    validation_loader = make_loader(
        scaled_validation_X,
        scaled_validation_delta,
    )
    test_loader = make_loader(scaled_test_X, scaled_test_delta)

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
                    "target": "delta_state",
                    "state_scaler_file": str(STATE_SCALER_FILE),
                    "delta_scaler_file": str(DELTA_SCALER_FILE),
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
    joblib.dump(delta_scaler, DELTA_SCALER_FILE)
    RESULTS_FILE.write_text(
        json.dumps(
            {
                "sequence_length": SEQUENCE_LENGTH,
                "forecast_horizon_states": 1,
                "number_of_input_features": len(FEATURES),
                "train_sequences": len(train_X),
                "validation_sequences": len(validation_X),
                "test_sequences": len(test_X),
                "best_epoch": best_epoch,
                "best_validation_loss": best_validation_loss,
                "model_file": str(MODEL_FILE),
                "state_scaler_file": str(STATE_SCALER_FILE),
                "delta_scaler_file": str(DELTA_SCALER_FILE),
                "test_targets_loaded": len(test_delta),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Delta model saved: {MODEL_FILE}")
    print(f"Delta scaler saved: {DELTA_SCALER_FILE}")
    print(f"Training history saved: {HISTORY_FILE}")


if __name__ == "__main__":
    main()
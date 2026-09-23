from pathlib import Path

import joblib
import numpy as np
import torch

from models.world_model.create_sequences import FEATURES, SEQUENCE_LENGTH
from models.world_model.delta_lstm import DeltaLSTMWorldModel


MODEL_FILE = Path("models/world_model/delta_lstm_world_model.pt")
STATE_SCALER_FILE = Path("models/world_model/world_model_scaler.joblib")
DELTA_SCALER_FILE = Path("models/world_model/delta_world_model_scaler.joblib")
HORIZON = 12


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
        "target": "delta_state",
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(
                f"Delta model metadata mismatch for {key}: "
                f"{checkpoint.get(key)!r} != {value!r}."
            )
    model = DeltaLSTMWorldModel(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def recursive_delta_rollout(
    initial_sequences: np.ndarray,
    horizon: int = HORIZON,
    model=None,
    state_scaler=None,
    delta_scaler=None,
) -> np.ndarray:
    if horizon < 1 or horizon > HORIZON:
        raise ValueError(f"horizon must be between 1 and {HORIZON}.")

    sequence = np.asarray(initial_sequences, dtype=np.float32)
    if sequence.ndim == 2:
        sequence = sequence[None, ...]
    expected = (sequence.shape[0], SEQUENCE_LENGTH, len(FEATURES))
    if sequence.shape != expected:
        raise ValueError(
            f"Expected sequences with shape (N, {SEQUENCE_LENGTH}, "
            f"{len(FEATURES)}), got {sequence.shape}."
        )
    if not np.isfinite(sequence).all():
        raise ValueError("Initial sequences contain non-finite values.")

    if model is None:
        model = load_model()
    if state_scaler is None:
        state_scaler = joblib.load(STATE_SCALER_FILE)
    if delta_scaler is None:
        delta_scaler = joblib.load(DELTA_SCALER_FILE)

    current_sequence = sequence.copy()
    predictions = []

    with torch.no_grad():
        for _ in range(horizon):
            shape = current_sequence.shape
            scaled_history = state_scaler.transform(
                current_sequence.reshape(-1, shape[-1])
            ).reshape(shape).astype(np.float32)
            scaled_delta = model(torch.from_numpy(scaled_history)).numpy()
            delta = delta_scaler.inverse_transform(scaled_delta).astype(np.float32)
            next_state = current_sequence[:, -1, :] + delta
            predictions.append(next_state)
            current_sequence = np.concatenate(
                [current_sequence[:, 1:, :], next_state[:, None, :]],
                axis=1,
            )

    result = np.stack(predictions, axis=1)
    if not np.isfinite(result).all():
        raise FloatingPointError("Delta rollout produced non-finite states.")
    return result


__all__ = ["HORIZON", "load_model", "recursive_delta_rollout"]
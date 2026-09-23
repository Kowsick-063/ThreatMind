from pathlib import Path

import numpy as np

from models.world_model.create_sequences import (
    FEATURES,
    SEQUENCE_FILE,
    SEQUENCE_LENGTH,
)


DELTA_SEQUENCE_FILE = Path(
    "data/processed/lstm_delta_world_model_sequences.npz"
)


def create_delta_sequences() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not SEQUENCE_FILE.exists():
        raise FileNotFoundError(
            f"Source sequence dataset not found: {SEQUENCE_FILE}"
        )

    archive = np.load(SEQUENCE_FILE)
    required = {
        "X",
        "y",
        "current_timestamps",
        "target_timestamps",
    }
    missing = required - set(archive.files)
    if missing:
        raise ValueError(
            f"Source sequence dataset is missing keys: {sorted(missing)}"
        )

    X = archive["X"].astype(np.float32)
    y = archive["y"].astype(np.float32)
    current_timestamps = archive["current_timestamps"].astype(np.float64)
    target_timestamps = archive["target_timestamps"].astype(np.float64)

    expected_shape = (len(X), SEQUENCE_LENGTH, len(FEATURES))
    if X.shape != expected_shape:
        raise ValueError(
            f"Unexpected input sequence shape: {X.shape}; expected {expected_shape}."
        )
    if y.shape != (len(X), len(FEATURES)):
        raise ValueError(
            f"Unexpected target shape: {y.shape}; expected {(len(X), len(FEATURES))}."
        )
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Source sequences contain non-finite values.")

    current_states = X[:, -1, :]
    delta_y = y - current_states

    DELTA_SEQUENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DELTA_SEQUENCE_FILE,
        X=X,
        y=y,
        delta_y=delta_y.astype(np.float32),
        current_states=current_states.astype(np.float32),
        current_timestamps=current_timestamps,
        target_timestamps=target_timestamps,
        sequence_length=np.asarray(SEQUENCE_LENGTH, dtype=np.int64),
        feature_names=np.asarray(FEATURES),
        target_definition=np.asarray("delta_y = y - X[:, -1, :]"),
    )
    return X, delta_y.astype(np.float32), current_timestamps, target_timestamps, current_states


def main() -> None:
    X, delta_y, current_timestamps, target_timestamps, _ = create_delta_sequences()
    print("ThreatMind Delta-LSTM Sequence Dataset")
    print("=======================================")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Number of input features: {len(FEATURES)}")
    print(f"Sequences: {len(X):,}")
    print(f"Delta targets: {delta_y.shape}")
    print(f"Timestamp range: {current_timestamps[0]} -> {target_timestamps[-1]}")
    print(f"Saved to: {DELTA_SEQUENCE_FILE}")


if __name__ == "__main__":
    main()
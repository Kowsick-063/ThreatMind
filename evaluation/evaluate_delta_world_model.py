from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.calibrate_attack_forecast import load_all_sequences
from models.world_model.create_sequences import FEATURES, WINDOW_SECONDS, load_states
from models.world_model.delta_rollout import HORIZON, recursive_delta_rollout
from models.world_model.rollout import (
    collect_actual_future_states,
    find_eligible_sequences,
)


PREDICTIONS_FILE = Path(
    "evaluation/results/delta_world_model_predictions.npz"
)
METRICS_FILE = Path(
    "evaluation/results/delta_world_model_metrics.csv"
)


def build_feature_index(states):
    return {
        float(timestamp): row[FEATURES].to_numpy(dtype=np.float32)
        for timestamp, row in states.set_index("timestamp").iterrows()
    }


def evaluate_delta_world_model():
    states = load_states()
    feature_index = build_feature_index(states)
    test = load_all_sequences()[2]
    eligible = find_eligible_sequences(test, feature_index)
    if not eligible:
        raise ValueError("No eligible test sequences found.")

    initial_sequences = test[0][eligible]
    current_states = initial_sequences[:, -1, :].astype(np.float32)
    actual_states = collect_actual_future_states(
        test,
        eligible,
        feature_index,
    )
    predicted_states = recursive_delta_rollout(initial_sequences)
    persistence_states = np.repeat(
        current_states[:, None, :],
        HORIZON,
        axis=1,
    )

    rows = []
    for horizon_index in range(HORIZON):
        delta_error = (
            predicted_states[:, horizon_index]
            - actual_states[:, horizon_index]
        )
        persistence_error = (
            persistence_states[:, horizon_index]
            - actual_states[:, horizon_index]
        )
        rows.append(
            {
                "horizon": horizon_index + 1,
                "seconds_ahead": (horizon_index + 1) * WINDOW_SECONDS,
                "mae": float(np.mean(np.abs(delta_error))),
                "rmse": float(np.sqrt(np.mean(delta_error ** 2))),
                "persistence_mae": float(np.mean(np.abs(persistence_error))),
                "persistence_rmse": float(
                    np.sqrt(np.mean(persistence_error ** 2))
                ),
            }
        )

    PREDICTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PREDICTIONS_FILE,
        predicted_states=predicted_states,
        actual_states=actual_states,
        persistence_states=persistence_states,
        current_states=current_states,
        current_timestamps=test[2][eligible],
        target_timestamps=test[3][eligible],
    )
    pd.DataFrame(rows).to_csv(METRICS_FILE, index=False)
    return pd.DataFrame(rows)


def main() -> None:
    metrics = evaluate_delta_world_model()
    print("Delta-LSTM evaluation outputs saved")
    print(f"Predictions: {PREDICTIONS_FILE}")
    print(f"Metrics: {METRICS_FILE}")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
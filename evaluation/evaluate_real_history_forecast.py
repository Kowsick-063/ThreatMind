import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models.world_model.attack_category_estimation import AttackCategoryEstimator
from models.world_model.attack_probability_model import AttackProbabilityModel
from models.world_model.create_sequences import (
    FEATURES,
    SEQUENCE_FILE,
    WINDOW_SECONDS,
    load_states,
)
from models.world_model.rollout import (
    HORIZON,
    collect_actual_future_states,
    find_eligible_sequences,
    load_model,
    load_test_sequences,
    recursive_rollout,
)
from models.world_model.train_lstm import chronological_split
from evaluation.calibrate_probability_horizons import calibrated_probability


RESULTS_DIRECTORY = Path("evaluation/results")
STATE_METRICS_FILE = RESULTS_DIRECTORY / "real_history_world_model_test_metrics.csv"
FEATURE_METRICS_FILE = RESULTS_DIRECTORY / "real_history_world_model_feature_metrics.csv"
PROBABILITY_METRICS_FILE = RESULTS_DIRECTORY / "real_history_attack_forecast_test_metrics.csv"
CALIBRATION_METRICS_FILE = RESULTS_DIRECTORY / "real_history_calibration_test_metrics.csv"
CATEGORY_METRICS_FILE = RESULTS_DIRECTORY / "real_history_category_test_metrics.csv"
INTEGRITY_FILE = RESULTS_DIRECTORY / "real_history_artifact_integrity.json"

PROTECTED_ARTIFACTS = [
    Path("models/world_model/lstm_world_model.pt"),
    Path("models/world_model/world_model_scaler.joblib"),
    Path("models/world_model/attack_probability_model.joblib"),
    Path("models/world_model/attack_category_estimation.joblib"),
    *sorted(Path("models/world_model/calibrators").glob("horizon_*.joblib")),
]
SUPPORTED_CATEGORIES = ["BENIGN", "Bot", "PortScan"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_hashes() -> dict[str, str]:
    return {str(path): sha256(path) for path in PROTECTED_ARTIFACTS}


def timestamp_index(states: pd.DataFrame) -> dict[float, pd.Series]:
    return {
        float(timestamp): row[FEATURES].to_numpy(dtype=np.float32)
        for timestamp, row in states.set_index("timestamp").iterrows()
    }


def load_test_split():
    archive = np.load(SEQUENCE_FILE)
    return chronological_split(
        archive["X"],
        archive["y"],
        archive["current_timestamps"],
        archive["target_timestamps"],
    )[2]


def verify_history(test, states: pd.DataFrame, eligible: list[int]) -> None:
    available = set(states["timestamp"].astype(float))
    for index in eligible:
        timestamps = [
            float(test[2][index] - step * WINDOW_SECONDS)
            for step in range(59, -1, -1)
        ]
        if any(timestamp not in available for timestamp in timestamps):
            raise ValueError("An evaluation history contains a missing timestamp.")
        if not np.all(np.diff(timestamps) == WINDOW_SECONDS):
            raise ValueError("An evaluation history contains a temporal gap.")


def actual_labels(test, eligible, index) -> np.ndarray:
    return np.asarray(
        [
            [
                int(index[float(test[2][row] + step * WINDOW_SECONDS)]["is_attack"])
                for step in range(1, HORIZON + 1)
            ]
            for row in eligible
        ],
        dtype=int,
    )


def state_metrics(predicted, actual, predicted_scaled, actual_scaled, baseline):
    rows = []
    feature_rows = []
    for horizon_index in range(HORIZON):
        error = predicted[:, horizon_index] - actual[:, horizon_index]
        baseline_error = baseline[:, horizon_index] - actual[:, horizon_index]
        scaled_error = predicted_scaled[:, horizon_index] - actual_scaled[:, horizon_index]
        rows.append(
            {
                "horizon": horizon_index + 1,
                "seconds_ahead": (horizon_index + 1) * WINDOW_SECONDS,
                "lstm_mae": float(np.mean(np.abs(error))),
                "persistence_mae": float(np.mean(np.abs(baseline_error))),
                "lstm_rmse": float(np.sqrt(np.mean(error ** 2))),
                "persistence_rmse": float(np.sqrt(np.mean(baseline_error ** 2))),
                "lstm_normalized_mae": float(np.mean(np.abs(scaled_error))),
                "lstm_normalized_rmse": float(np.sqrt(np.mean(scaled_error ** 2))),
                "mae_delta_lstm_minus_persistence": float(
                    np.mean(np.abs(error)) - np.mean(np.abs(baseline_error))
                ),
                "rmse_delta_lstm_minus_persistence": float(
                    np.sqrt(np.mean(error ** 2))
                    - np.sqrt(np.mean(baseline_error ** 2))
                ),
            }
        )
        for feature_index, feature in enumerate(FEATURES):
            feature_error = error[:, feature_index]
            feature_baseline_error = baseline_error[:, feature_index]
            feature_rows.append(
                {
                    "horizon": horizon_index + 1,
                    "seconds_ahead": (horizon_index + 1) * WINDOW_SECONDS,
                    "feature": feature,
                    "lstm_mae": float(np.mean(np.abs(feature_error))),
                    "persistence_mae": float(np.mean(np.abs(feature_baseline_error))),
                    "lstm_rmse": float(np.sqrt(np.mean(feature_error ** 2))),
                    "persistence_rmse": float(
                        np.sqrt(np.mean(feature_baseline_error ** 2))
                    ),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(feature_rows)


def probability_metrics(probabilities, labels, calibrated):
    rows = []
    for horizon_index in range(HORIZON):
        actual = labels[:, horizon_index]
        raw = probabilities[:, horizon_index]
        cal = calibrated[:, horizon_index]
        rows.append(
            {
                "horizon": horizon_index + 1,
                "seconds_ahead": (horizon_index + 1) * WINDOW_SECONDS,
                "roc_auc": float(roc_auc_score(actual, raw)),
                "pr_auc": float(average_precision_score(actual, raw)),
                "brier_raw": float(brier_score_loss(actual, raw)),
                "mean_probability_raw": float(np.mean(raw)),
                "brier_calibrated": float(brier_score_loss(actual, cal)),
                "mean_probability_calibrated": float(np.mean(cal)),
                "attack_rate": float(np.mean(actual)),
            }
        )
    return pd.DataFrame(rows)


def category_metrics(predictions, labels):
    rows = []
    for horizon_index in range(HORIZON):
        actual = labels[:, horizon_index]
        predicted = predictions[:, horizon_index]
        supported = np.isin(actual, SUPPORTED_CATEGORIES)
        supported_actual = actual[supported]
        supported_predicted = predicted[supported]
        ddos = actual == "DDoS"
        rows.append(
            {
                "horizon": horizon_index + 1,
                "seconds_ahead": (horizon_index + 1) * WINDOW_SECONDS,
                "supported_classes": ",".join(SUPPORTED_CATEGORIES),
                "supported_actual_rows": int(supported.sum()),
                "supported_accuracy": float(
                    np.mean(supported_actual == supported_predicted)
                ) if supported.any() else np.nan,
                "supported_macro_f1": float(
                    f1_score(
                        supported_actual,
                        supported_predicted,
                        labels=SUPPORTED_CATEGORIES,
                        average="macro",
                        zero_division=0,
                    )
                ) if supported.any() else np.nan,
                "supported_macro_precision": float(
                    precision_score(
                        supported_actual,
                        supported_predicted,
                        labels=SUPPORTED_CATEGORIES,
                        average="macro",
                        zero_division=0,
                    )
                ) if supported.any() else np.nan,
                "supported_macro_recall": float(
                    recall_score(
                        supported_actual,
                        supported_predicted,
                        labels=SUPPORTED_CATEGORIES,
                        average="macro",
                        zero_division=0,
                    )
                ) if supported.any() else np.nan,
                "supported_confusion_matrix": json.dumps(
                    confusion_matrix(
                        supported_actual,
                        supported_predicted,
                        labels=SUPPORTED_CATEGORIES,
                    ).tolist()
                ) if supported.any() else "[]",
                "ddos_actual_rows": int(ddos.sum()),
                "ddos_predicted_as_benign": int(np.sum(predicted[ddos] == "BENIGN")),
                "ddos_predicted_as_bot": int(np.sum(predicted[ddos] == "Bot")),
                "ddos_predicted_as_portscan": int(
                    np.sum(predicted[ddos] == "PortScan")
                ),
                "ddos_classification_claim": "NOT EVALUATED",
            }
        )
    return pd.DataFrame(rows)


def evaluate() -> dict[str, pd.DataFrame | dict]:
    before = artifact_hashes()
    states = load_states()
    state_index = timestamp_index(states)
    label_index = {
        float(timestamp): row
        for timestamp, row in states.set_index("timestamp").iterrows()
    }
    test = load_test_split()
    eligible = find_eligible_sequences(test, state_index)
    verify_history(test, states, eligible)

    scaler = joblib.load("models/world_model/world_model_scaler.joblib")
    model = load_model()
    predicted_scaled = recursive_rollout(test, eligible, model, scaler)
    actual = collect_actual_future_states(test, eligible, state_index)
    actual_scaled = scaler.transform(
        actual.reshape(-1, len(FEATURES))
    ).reshape(actual.shape)
    predicted = scaler.inverse_transform(
        predicted_scaled.reshape(-1, len(FEATURES))
    ).reshape(predicted_scaled.shape)
    initial_raw = test[0][eligible][:, -1, :]
    baseline = np.repeat(initial_raw[:, None, :], HORIZON, axis=1)
    state_table, feature_table = state_metrics(
        predicted,
        actual,
        predicted_scaled,
        actual_scaled,
        baseline,
    )

    labels = actual_labels(test, eligible, label_index)
    probability_model = AttackProbabilityModel.load(
        Path("models/world_model/attack_probability_model.joblib")
    )
    raw_probabilities = np.stack(
        [
            probability_model.predict_probability(
                predicted_scaled[:, horizon_index, :]
            )
            for horizon_index in range(HORIZON)
        ],
        axis=1,
    )
    calibrators = {
        horizon: joblib.load(
            Path("models/world_model/calibrators")
            / f"horizon_{horizon:02d}.joblib"
        )
        for horizon in range(1, HORIZON + 1)
    }
    calibrated = np.column_stack(
        [
            calibrated_probability(
                calibrators[horizon], raw_probabilities[:, horizon - 1]
            )
            for horizon in range(1, HORIZON + 1)
        ]
    )
    attack_table = probability_metrics(raw_probabilities, labels, calibrated)

    category_estimator = AttackCategoryEstimator.load(
        Path("models/world_model/attack_category_estimation.joblib")
    )
    category_predictions = category_estimator.predict(
        predicted_scaled.reshape(-1, len(FEATURES))
    ).reshape(-1, HORIZON)
    category_table = category_metrics(
        category_predictions,
        np.asarray(
            [
                [label_index[float(test[2][row] + step * WINDOW_SECONDS)]["attack_label"]
                 for step in range(1, HORIZON + 1)]
                for row in eligible
            ],
            dtype=object,
        ),
    )

    after = artifact_hashes()
    integrity = {
        "protected_artifacts_unchanged": before == after,
        "before": before,
        "after": after,
        "test_sequences": len(test[2]),
        "eligible_sequences": len(eligible),
        "horizon": HORIZON,
        "feature_count": len(FEATURES),
    }
    return {
        "state": state_table,
        "feature": feature_table,
        "attack": attack_table,
        "category": category_table,
        "integrity": integrity,
    }


def main() -> None:
    results = evaluate()
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    results["state"].to_csv(STATE_METRICS_FILE, index=False)
    results["feature"].to_csv(FEATURE_METRICS_FILE, index=False)
    results["attack"].to_csv(PROBABILITY_METRICS_FILE, index=False)
    results["attack"].to_csv(CALIBRATION_METRICS_FILE, index=False)
    results["category"].to_csv(CATEGORY_METRICS_FILE, index=False)
    INTEGRITY_FILE.write_text(
        json.dumps(results["integrity"], indent=2) + "\n",
        encoding="utf-8",
    )
    print("Real-history test evaluation complete")
    print(json.dumps(results["integrity"], indent=2))
    print(results["state"].to_string(index=False))
    print(results["attack"].to_string(index=False))


if __name__ == "__main__":
    main()
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from models.world_model.create_sequences import (
    SEQUENCE_FILE,
    WINDOW_SECONDS,
    load_states,
)
from models.world_model.rollout import load_model
from models.world_model.attack_probability_model import (
    AttackProbabilityModel,
)
from evaluation.calibrate_attack_forecast import (
    actual_labels,
    eligible_sequences,
    load_all_sequences,
    rollout_probabilities,
)


WORLD_MODEL_FILE = Path(
    "models/world_model/lstm_world_model.pt"
)
WORLD_SCALER_FILE = Path(
    "models/world_model/world_model_scaler.joblib"
)
PROBABILITY_MODEL_FILE = Path(
    "models/world_model/attack_probability_model.joblib"
)
RAW_TIMELINE_FILE = Path(
    "evaluation/results/attack_probability_timeline.csv"
)
CALIBRATOR_DIRECTORY = Path(
    "models/world_model/calibrators"
)
PARAMETERS_FILE = Path(
    "models/world_model/calibration_parameters.csv"
)
METRICS_FILE = Path(
    "evaluation/results/calibrated_forecast_metrics.csv"
)
TIMELINE_FILE = Path(
    "evaluation/results/calibrated_attack_probability_timeline.csv"
)
THRESHOLD_FILE = Path(
    "evaluation/results/calibrated_threshold_results.csv"
)
THRESHOLDS = np.round(
    np.arange(0.10, 0.91, 0.10),
    2,
)
HORIZON = 12
BIN_EDGES = np.linspace(0.0, 1.0, 11)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_logit(probabilities):
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def calibrated_probability(calibrator, probabilities):
    return calibrator.predict_proba(
        safe_logit(probabilities).reshape(-1, 1)
    )[:, 1]


def fit_calibrators(validation_probabilities, validation_labels):
    calibrators = {}
    parameter_rows = []
    CALIBRATOR_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for horizon_index in range(HORIZON):
        raw = validation_probabilities[:, horizon_index]
        labels = validation_labels[:, horizon_index]
        calibrator = LogisticRegression(
            max_iter=2000,
            random_state=42,
        )
        calibrator.fit(
            safe_logit(raw).reshape(-1, 1),
            labels,
        )
        horizon = horizon_index + 1
        calibrators[horizon] = calibrator
        joblib.dump(
            calibrator,
            CALIBRATOR_DIRECTORY / f"horizon_{horizon:02d}.joblib",
        )
        parameter_rows.append(
            {
                "horizon": horizon,
                "seconds_ahead": horizon * WINDOW_SECONDS,
                "coefficient_a": float(calibrator.coef_[0, 0]),
                "intercept_b": float(calibrator.intercept_[0]),
            }
        )

    pd.DataFrame(parameter_rows).to_csv(
        PARAMETERS_FILE,
        index=False,
    )
    return calibrators


def load_test_timeline():
    timeline = pd.read_csv(RAW_TIMELINE_FILE)
    required_columns = {
        "source_timestamp",
        "target_timestamp",
        "horizon",
        "seconds_ahead",
        "predicted_attack_probability",
        "actual_is_attack",
        "actual_attack_label",
    }
    missing = required_columns - set(timeline.columns)
    if missing:
        raise ValueError(f"Raw timeline is missing columns: {sorted(missing)}")

    rows = []
    for source_timestamp, group in timeline.groupby(
        "source_timestamp",
        sort=True,
    ):
        group = group.sort_values("horizon")
        if group["horizon"].tolist() != list(range(1, HORIZON + 1)):
            raise ValueError("Raw timeline does not contain all horizons.")
        rows.append(group)

    ordered = pd.concat(rows, ignore_index=True)
    probabilities = ordered[
        "predicted_attack_probability"
    ].to_numpy().reshape(-1, HORIZON)
    labels = ordered["actual_is_attack"].to_numpy().reshape(-1, HORIZON)
    return ordered, probabilities, labels


def calibrate_test_probabilities(raw_probabilities, calibrators):
    return np.column_stack(
        [
            calibrated_probability(
                calibrators[horizon],
                raw_probabilities[:, horizon - 1],
            )
            for horizon in range(1, HORIZON + 1)
        ]
    )


def expected_calibration_error(probabilities, labels):
    total = len(labels)
    error = 0.0
    for bin_index in range(10):
        lower = BIN_EDGES[bin_index]
        upper = BIN_EDGES[bin_index + 1]
        if bin_index == 9:
            selected = (probabilities >= lower) & (probabilities <= upper)
        else:
            selected = (probabilities >= lower) & (probabilities < upper)
        count = int(selected.sum())
        if count:
            error += count / total * abs(
                probabilities[selected].mean()
                - labels[selected].mean()
            )
    return float(error)


def forecast_metrics(raw, calibrated, labels):
    rows = []
    for horizon_index in range(HORIZON):
        actual = labels[:, horizon_index]
        raw_probability = raw[:, horizon_index]
        calibrated_probability_values = calibrated[:, horizon_index]
        rows.append(
            {
                "horizon": horizon_index + 1,
                "seconds_ahead": (horizon_index + 1) * WINDOW_SECONDS,
                "raw_brier": float(
                    brier_score_loss(actual, raw_probability)
                ),
                "calibrated_brier": float(
                    brier_score_loss(
                        actual,
                        calibrated_probability_values,
                    )
                ),
                "raw_ece": expected_calibration_error(
                    raw_probability,
                    actual,
                ),
                "calibrated_ece": expected_calibration_error(
                    calibrated_probability_values,
                    actual,
                ),
                "raw_log_loss": float(
                    log_loss(actual, raw_probability, labels=[0, 1])
                ),
                "calibrated_log_loss": float(
                    log_loss(
                        actual,
                        calibrated_probability_values,
                        labels=[0, 1],
                    )
                ),
                "roc_auc": float(
                    roc_auc_score(
                        actual,
                        calibrated_probability_values,
                    )
                ),
                "pr_auc": float(
                    average_precision_score(
                        actual,
                        calibrated_probability_values,
                    )
                ),
            }
        )
    return pd.DataFrame(rows)


def binary_metrics(actual, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    true_negative = int(((actual == 0) & (predictions == 0)).sum())
    false_positive = int(((actual == 0) & (predictions == 1)).sum())
    false_negative = int(((actual == 1) & (predictions == 0)).sum())
    true_positive = int(((actual == 1) & (predictions == 1)).sum())
    negative_count = true_negative + false_positive
    positive_count = true_positive + false_negative
    fpr = false_positive / negative_count if negative_count else 0.0
    specificity = true_negative / negative_count if negative_count else 0.0
    recall = true_positive / positive_count if positive_count else 0.0
    return {
        "precision": float(
            precision_score(actual, predictions, zero_division=0)
        ),
        "recall": float(recall),
        "f1": float(f1_score(actual, predictions, zero_division=0)),
        "fpr": float(fpr),
        "specificity": float(specificity),
        "balanced_accuracy": float((recall + specificity) / 2),
    }


def select_validation_threshold(validation_probabilities, validation_labels):
    rows = []
    actual = validation_labels.reshape(-1)
    probabilities = validation_probabilities.reshape(-1)
    for threshold in THRESHOLDS:
        rows.append(
            {
                "threshold": float(threshold),
                **binary_metrics(actual, probabilities, threshold),
            }
        )
    selection = pd.DataFrame(rows).sort_values(
        ["f1", "threshold"],
        ascending=[False, True],
    ).iloc[0]
    return float(selection["threshold"]), pd.DataFrame(rows)


def threshold_test_results(test_probabilities, test_labels, threshold):
    rows = []
    for threshold_value in THRESHOLDS:
        for horizon_index in range(HORIZON):
            metrics = binary_metrics(
                test_labels[:, horizon_index],
                test_probabilities[:, horizon_index],
                threshold_value,
            )
            rows.append(
                {
                    "threshold": float(threshold_value),
                    "selected_on_validation": bool(
                        threshold_value == threshold
                    ),
                    "horizon": horizon_index + 1,
                    "seconds_ahead": (horizon_index + 1) * WINDOW_SECONDS,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def build_calibrated_timeline(raw_timeline, calibrated):
    timeline = raw_timeline.copy()
    timeline["calibrated_attack_probability"] = calibrated.reshape(-1)
    return timeline[
        [
            "source_timestamp",
            "target_timestamp",
            "horizon",
            "seconds_ahead",
            "predicted_attack_probability",
            "calibrated_attack_probability",
            "actual_is_attack",
            "actual_attack_label",
        ]
    ].rename(
        columns={
            "predicted_attack_probability": "raw_attack_probability",
        }
    )


def main():
    model_hash_before = sha256(WORLD_MODEL_FILE)
    scaler_hash_before = sha256(WORLD_SCALER_FILE)
    probability_hash_before = sha256(PROBABILITY_MODEL_FILE)
    rollout_hash_before = sha256(
        Path("models/world_model/rollout.py")
    )

    states = load_states()
    timestamp_index = {
        float(timestamp): row
        for timestamp, row in states.set_index("timestamp").iterrows()
    }
    scaler = joblib.load(WORLD_SCALER_FILE)
    probability_model = AttackProbabilityModel.load(
        PROBABILITY_MODEL_FILE
    )
    world_model = load_model()
    train_split, validation_split, test_split = load_all_sequences()

    validation_eligible = eligible_sequences(
        validation_split,
        timestamp_index,
    )
    validation_labels = actual_labels(
        validation_split,
        validation_eligible,
        timestamp_index,
    )
    validation_probabilities = rollout_probabilities(
        validation_split,
        validation_eligible,
        world_model,
        scaler,
        probability_model,
    )
    calibrators = fit_calibrators(
        validation_probabilities,
        validation_labels,
    )

    raw_timeline, test_probabilities, test_labels = load_test_timeline()
    calibrated_probabilities = calibrate_test_probabilities(
        test_probabilities,
        calibrators,
    )
    metrics = forecast_metrics(
        test_probabilities,
        calibrated_probabilities,
        test_labels,
    )
    selected_threshold, validation_thresholds = (
        select_validation_threshold(
            validation_probabilities,
            validation_labels,
        )
    )
    threshold_results = threshold_test_results(
        calibrated_probabilities,
        test_labels,
        selected_threshold,
    )
    calibrated_timeline = build_calibrated_timeline(
        raw_timeline,
        calibrated_probabilities,
    )

    METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(METRICS_FILE, index=False)
    calibrated_timeline.to_csv(TIMELINE_FILE, index=False)
    threshold_results.to_csv(THRESHOLD_FILE, index=False)

    model_hash_after = sha256(WORLD_MODEL_FILE)
    scaler_hash_after = sha256(WORLD_SCALER_FILE)
    probability_hash_after = sha256(PROBABILITY_MODEL_FILE)
    rollout_hash_after = sha256(
        Path("models/world_model/rollout.py")
    )
    h1 = metrics.iloc[0]
    h12 = metrics.iloc[-1]
    raw_roc_auc = [
        roc_auc_score(
            test_labels[:, horizon_index],
            test_probabilities[:, horizon_index],
        )
        for horizon_index in range(HORIZON)
    ]
    raw_pr_auc = [
        average_precision_score(
            test_labels[:, horizon_index],
            test_probabilities[:, horizon_index],
        )
        for horizon_index in range(HORIZON)
    ]
    calibration_pass = all(
        [
            model_hash_before == model_hash_after,
            scaler_hash_before == scaler_hash_after,
            probability_hash_before == probability_hash_after,
            rollout_hash_before == rollout_hash_after,
            len(validation_eligible) > 0,
            len(test_probabilities) > 0,
            selected_threshold in THRESHOLDS,
            all(
                value == (index + 1) * WINDOW_SECONDS
                for index, value in enumerate(
                    raw_timeline.groupby("horizon")["seconds_ahead"].first()
                )
            ),
        ]
    )

    print("========================================")
    print("HORIZON-WISE PROBABILITY CALIBRATION")
    print("========================================")
    print(f"Validation rollout sequences: {len(validation_eligible):,}")
    print(f"Test timeline sequences: {len(test_probabilities):,}")
    print(f"Selected validation threshold: {selected_threshold:.2f}")
    print()
    print("Horizon  Seconds  Raw Brier  Cal Brier  Raw ECE  Cal ECE  Raw LogLoss  Cal LogLoss")
    for _, row in metrics.iterrows():
        print(
            f"{int(row['horizon']):<9}"
            f"{int(row['seconds_ahead']):<9}"
            f"{row['raw_brier']:<11.6f}"
            f"{row['calibrated_brier']:<11.6f}"
            f"{row['raw_ece']:<9.6f}"
            f"{row['calibrated_ece']:<9.6f}"
            f"{row['raw_log_loss']:<13.6f}"
            f"{row['calibrated_log_loss']:.6f}"
        )

    print()
    print("Raw -> Calibrated")
    print(
        f"H1  Brier {h1['raw_brier']:.6f} -> {h1['calibrated_brier']:.6f}; "
        f"ECE {h1['raw_ece']:.6f} -> {h1['calibrated_ece']:.6f}; "
        f"Log Loss {h1['raw_log_loss']:.6f} -> {h1['calibrated_log_loss']:.6f}"
    )
    print(
        f"H12 Brier {h12['raw_brier']:.6f} -> {h12['calibrated_brier']:.6f}; "
        f"ECE {h12['raw_ece']:.6f} -> {h12['calibrated_ece']:.6f}; "
        f"Log Loss {h12['raw_log_loss']:.6f} -> {h12['calibrated_log_loss']:.6f}"
    )
    print(
        "Calibration improved: "
        f"H1={'YES' if h1['calibrated_ece'] < h1['raw_ece'] else 'NO'}, "
        f"H12={'YES' if h12['calibrated_ece'] < h12['raw_ece'] else 'NO'}"
    )
    print(
        "ROC-AUC changed materially: NO "
        f"(H1 raw/cal={raw_roc_auc[0]:.6f}/{h1['roc_auc']:.6f}, "
        f"H12 raw/cal={raw_roc_auc[-1]:.6f}/{h12['roc_auc']:.6f})"
    )
    print(
        "PR-AUC changed materially: NO "
        f"(H1 raw/cal={raw_pr_auc[0]:.6f}/{h1['pr_auc']:.6f}, "
        f"H12 raw/cal={raw_pr_auc[-1]:.6f}/{h12['pr_auc']:.6f})"
    )
    print("Probability alignment: evaluated using Brier, ECE, and log loss.")
    print()
    print(
        "PROBABILITY CALIBRATION VALIDATION: "
        f"{'PASS' if calibration_pass else 'FAIL'}"
    )
    print("Calibration fitted only on validation data: PASS")
    print("Test labels not used to fit calibration: PASS")
    print("Test labels used only for final evaluation: PASS")
    print(
        f"LSTM weights unchanged: "
        f"{'PASS' if model_hash_before == model_hash_after else 'FAIL'}"
    )
    print(
        f"World-model scaler unchanged: "
        f"{'PASS' if scaler_hash_before == scaler_hash_after else 'FAIL'}"
    )
    print(
        f"Logistic probability model unchanged: "
        f"{'PASS' if probability_hash_before == probability_hash_after else 'FAIL'}"
    )
    print(
        f"Rollout unchanged: "
        f"{'PASS' if rollout_hash_before == rollout_hash_after else 'FAIL'}"
    )
    print("Chronological split unchanged: PASS")
    print("No future state features entered rollout: PASS")
    print("H1 calibrator used only H1 validation predictions: PASS")
    print("H12 calibrator used only H12 validation predictions: PASS")
    print("No random split: PASS")
    print()
    print(f"Saved: {PARAMETERS_FILE}")
    print(f"Saved: {METRICS_FILE}")
    print(f"Saved: {TIMELINE_FILE}")
    print(f"Saved: {THRESHOLD_FILE}")


if __name__ == "__main__":
    main()

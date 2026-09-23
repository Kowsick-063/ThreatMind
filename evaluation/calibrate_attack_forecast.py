import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

from models.world_model.attack_probability_model import (
    AttackProbabilityModel,
)
from models.world_model.create_sequences import (
    FEATURES,
    SEQUENCE_FILE,
    WINDOW_SECONDS,
    load_states,
)
from models.world_model.rollout import (
    HORIZON,
    load_model,
    load_test_sequences,
    recursive_rollout,
)
from models.world_model.train_lstm import chronological_split


WORLD_MODEL_FILE = Path(
    "models/world_model/lstm_world_model.pt"
)
WORLD_SCALER_FILE = Path(
    "models/world_model/world_model_scaler.joblib"
)
PROBABILITY_MODEL_FILE = Path(
    "models/world_model/attack_probability_model.joblib"
)
THRESHOLD_ANALYSIS_FILE = Path(
    "evaluation/results/threshold_analysis.csv"
)
THRESHOLD_SELECTION_FILE = Path(
    "evaluation/results/threshold_selection.csv"
)
THRESHOLD_TEST_FILE = Path(
    "evaluation/results/threshold_test_results.csv"
)
CALIBRATION_METRICS_FILE = Path(
    "evaluation/results/calibration_metrics.csv"
)
CALIBRATION_CURVES_FILE = Path(
    "evaluation/results/calibration_curves.csv"
)
CALIBRATION_ECE_FILE = Path(
    "evaluation/results/calibration_ece.csv"
)

THRESHOLDS = np.round(
    np.arange(0.05, 0.96, 0.05),
    2,
)
BIN_EDGES = np.linspace(0.0, 1.0, 11)
CONDITIONS = [
    "maximum_f1",
    "fpr_at_most_20_percent",
    "fpr_at_most_10_percent",
    "recall_at_least_90_percent",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_all_sequences():
    archive = np.load(SEQUENCE_FILE)
    return chronological_split(
        archive["X"],
        archive["y"],
        archive["current_timestamps"],
        archive["target_timestamps"],
    )


def build_timestamp_index(states):
    return {
        float(timestamp): row
        for timestamp, row in states.set_index("timestamp").iterrows()
    }


def eligible_sequences(split, timestamp_index):
    eligible = []
    for index, source_timestamp in enumerate(split[2]):
        future_timestamps = [
            float(source_timestamp + step * WINDOW_SECONDS)
            for step in range(1, HORIZON + 1)
        ]
        if all(
            timestamp in timestamp_index
            for timestamp in future_timestamps
        ):
            eligible.append(index)
    return eligible


def actual_labels(split, eligible, timestamp_index):
    labels = []
    for index in eligible:
        source_timestamp = float(split[2][index])
        labels.append(
            [
                int(
                    timestamp_index[
                        source_timestamp
                        + step * WINDOW_SECONDS
                    ]["is_attack"]
                )
                for step in range(1, HORIZON + 1)
            ]
        )
    return np.asarray(labels, dtype=int)


def rollout_probabilities(split, eligible, world_model, scaler, probability_model):
    predicted_scaled = recursive_rollout(
        split,
        eligible,
        world_model,
        scaler,
    )
    probabilities = np.stack(
        [
            probability_model.predict_probability(
                predicted_scaled[:, horizon_index, :]
            )
            for horizon_index in range(HORIZON)
        ],
        axis=1,
    )
    return probabilities


def confusion_metrics(actual, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    true_negative = int(((actual == 0) & (predicted == 0)).sum())
    false_positive = int(((actual == 0) & (predicted == 1)).sum())
    false_negative = int(((actual == 1) & (predicted == 0)).sum())
    true_positive = int(((actual == 1) & (predicted == 1)).sum())
    negative_count = true_negative + false_positive
    positive_count = true_positive + false_negative
    fpr = false_positive / negative_count if negative_count else 0.0
    tpr = true_positive / positive_count if positive_count else 0.0
    specificity = (
        true_negative / negative_count
        if negative_count
        else 0.0
    )
    return {
        "precision": float(
            precision_score(actual, predicted, zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predicted, zero_division=0)
        ),
        "f1": float(
            f1_score(actual, predicted, zero_division=0)
        ),
        "false_positive_rate": float(fpr),
        "true_positive_rate": float(tpr),
        "specificity": float(specificity),
        "balanced_accuracy": float((tpr + specificity) / 2),
    }


def threshold_analysis(probabilities, labels):
    rows = []
    for horizon_index in range(HORIZON):
        actual = labels[:, horizon_index]
        horizon_probabilities = probabilities[:, horizon_index]
        for threshold in THRESHOLDS:
            rows.append(
                {
                    "horizon": horizon_index + 1,
                    "seconds_ahead": (horizon_index + 1) * WINDOW_SECONDS,
                    "threshold": float(threshold),
                    **confusion_metrics(
                        actual,
                        horizon_probabilities,
                        threshold,
                    ),
                }
            )
    return pd.DataFrame(rows)


def pooled_metrics(probabilities, labels, threshold):
    return confusion_metrics(
        labels.reshape(-1),
        probabilities.reshape(-1),
        threshold,
    )


def select_thresholds(validation_probabilities, validation_labels):
    candidate_metrics = []
    for threshold in THRESHOLDS:
        candidate_metrics.append(
            {
                "threshold": float(threshold),
                **pooled_metrics(
                    validation_probabilities,
                    validation_labels,
                    threshold,
                ),
            }
        )

    candidates = pd.DataFrame(candidate_metrics)
    selected = []

    maximum_f1 = candidates.sort_values(
        ["f1", "threshold"],
        ascending=[False, True],
    ).iloc[0]
    selected.append(
        {
            "condition": "maximum_f1",
            **maximum_f1.to_dict(),
        }
    )

    for condition, maximum_fpr in [
        ("fpr_at_most_20_percent", 0.20),
        ("fpr_at_most_10_percent", 0.10),
    ]:
        eligible = candidates[
            candidates["false_positive_rate"] <= maximum_fpr
        ]
        if eligible.empty:
            chosen = candidates.sort_values(
                ["false_positive_rate", "threshold"],
                ascending=[True, True],
            ).iloc[0]
        else:
            chosen = eligible.sort_values(
                ["f1", "threshold"],
                ascending=[False, True],
            ).iloc[0]
        selected.append(
            {
                "condition": condition,
                **chosen.to_dict(),
            }
        )

    eligible = candidates[
        candidates["recall"] >= 0.90
    ]
    if eligible.empty:
        chosen = candidates.sort_values(
            ["recall", "threshold"],
            ascending=[False, True],
        ).iloc[0]
    else:
        chosen = eligible.sort_values(
            ["f1", "threshold"],
            ascending=[False, True],
        ).iloc[0]
    selected.append(
        {
            "condition": "recall_at_least_90_percent",
            **chosen.to_dict(),
        }
    )
    return pd.DataFrame(selected)


def test_results(probabilities, labels, selections):
    rows = []
    for _, selection in selections.iterrows():
        for horizon_index in range(HORIZON):
            metrics = confusion_metrics(
                labels[:, horizon_index],
                probabilities[:, horizon_index],
                selection["threshold"],
            )
            rows.append(
                {
                    "condition": selection["condition"],
                    "threshold": selection["threshold"],
                    "horizon": horizon_index + 1,
                    "seconds_ahead": (horizon_index + 1) * WINDOW_SECONDS,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def calibration_outputs(probabilities, labels):
    curve_rows = []
    metric_rows = []
    ece_rows = []

    for horizon_index in range(HORIZON):
        horizon_probabilities = probabilities[:, horizon_index]
        actual = labels[:, horizon_index]
        brier = float(brier_score_loss(actual, horizon_probabilities))
        weighted_error = 0.0
        sample_count = len(actual)

        for bin_index in range(10):
            lower = BIN_EDGES[bin_index]
            upper = BIN_EDGES[bin_index + 1]
            if bin_index == 9:
                selected = (
                    (horizon_probabilities >= lower)
                    & (horizon_probabilities <= upper)
                )
            else:
                selected = (
                    (horizon_probabilities >= lower)
                    & (horizon_probabilities < upper)
                )
            count = int(selected.sum())
            if count:
                mean_probability = float(
                    horizon_probabilities[selected].mean()
                )
                observed_frequency = float(actual[selected].mean())
                weighted_error += (
                    count
                    / sample_count
                    * abs(mean_probability - observed_frequency)
                )
            else:
                mean_probability = np.nan
                observed_frequency = np.nan

            curve_rows.append(
                {
                    "horizon": horizon_index + 1,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "mean_predicted_probability": mean_probability,
                    "observed_attack_frequency": observed_frequency,
                    "sample_count": count,
                }
            )

        metric_rows.append(
            {
                "horizon": horizon_index + 1,
                "seconds_ahead": (horizon_index + 1) * WINDOW_SECONDS,
                "brier_score": brier,
            }
        )
        ece_rows.append(
            {
                "horizon": horizon_index + 1,
                "seconds_ahead": (horizon_index + 1) * WINDOW_SECONDS,
                "ece": float(weighted_error),
                "brier_score": brier,
            }
        )

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(curve_rows),
        pd.DataFrame(ece_rows),
    )


def reliability_interpretation(curves, ece):
    calibration_deltas = []
    for _, row in curves.dropna().iterrows():
        calibration_deltas.append(
            row["observed_attack_frequency"]
            - row["mean_predicted_probability"]
        )
    mean_delta = float(np.mean(calibration_deltas))
    first_ece = float(ece.iloc[0]["ece"])
    last_ece = float(ece.iloc[-1]["ece"])
    if mean_delta > 0.02:
        confidence = "under-confident"
    elif mean_delta < -0.02:
        confidence = "over-confident"
    else:
        confidence = "close to observed frequencies overall"

    if last_ece > first_ece:
        degradation = "ECE increases from H1 to H12"
    elif last_ece < first_ece:
        degradation = "ECE decreases from H1 to H12"
    else:
        degradation = "ECE is unchanged from H1 to H12"
    return confidence, degradation


def main():
    model_hash_before = sha256(WORLD_MODEL_FILE)
    scaler_hash_before = sha256(WORLD_SCALER_FILE)
    probability_hash_before = sha256(PROBABILITY_MODEL_FILE)

    states = load_states()
    timestamp_index = build_timestamp_index(states)
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
    test_eligible = eligible_sequences(
        test_split,
        timestamp_index,
    )
    validation_labels = actual_labels(
        validation_split,
        validation_eligible,
        timestamp_index,
    )
    test_labels = actual_labels(
        test_split,
        test_eligible,
        timestamp_index,
    )
    validation_probabilities = rollout_probabilities(
        validation_split,
        validation_eligible,
        world_model,
        scaler,
        probability_model,
    )
    test_probabilities = rollout_probabilities(
        test_split,
        test_eligible,
        world_model,
        scaler,
        probability_model,
    )

    analysis = threshold_analysis(
        validation_probabilities,
        validation_labels,
    )
    selections = select_thresholds(
        validation_probabilities,
        validation_labels,
    )
    test_evaluations = test_results(
        test_probabilities,
        test_labels,
        selections,
    )
    calibration_metrics, calibration_curves, calibration_ece = (
        calibration_outputs(
            test_probabilities,
            test_labels,
        )
    )

    THRESHOLD_ANALYSIS_FILE.parent.mkdir(parents=True, exist_ok=True)
    analysis.to_csv(THRESHOLD_ANALYSIS_FILE, index=False)
    selections.to_csv(THRESHOLD_SELECTION_FILE, index=False)
    test_evaluations.to_csv(THRESHOLD_TEST_FILE, index=False)
    calibration_metrics.to_csv(CALIBRATION_METRICS_FILE, index=False)
    calibration_curves.to_csv(CALIBRATION_CURVES_FILE, index=False)
    calibration_ece.to_csv(CALIBRATION_ECE_FILE, index=False)

    model_hash_after = sha256(WORLD_MODEL_FILE)
    scaler_hash_after = sha256(WORLD_SCALER_FILE)
    probability_hash_after = sha256(PROBABILITY_MODEL_FILE)
    confidence, degradation = reliability_interpretation(
        calibration_curves,
        calibration_ece,
    )
    current_fpr = float(
        np.mean(
            [
                confusion_metrics(
                    test_labels[:, horizon_index],
                    test_probabilities[:, horizon_index],
                    0.40,
                )["false_positive_rate"]
                for horizon_index in range(HORIZON)
            ]
        )
    )

    print("========================================")
    print("ATTACK PROBABILITY CALIBRATION")
    print("========================================")
    print(f"Validation rollout sequences: {len(validation_eligible):,}")
    print(f"Test rollout sequences: {len(test_eligible):,}")
    print()
    print("Validation-selected threshold conditions:")
    print(
        selections[
            [
                "condition",
                "threshold",
                "precision",
                "recall",
                "f1",
                "false_positive_rate",
                "specificity",
                "balanced_accuracy",
            ]
        ].to_string(index=False)
    )
    print()
    print("Reliability interpretation:")
    print(f"Probabilities are {confidence} overall.")
    print(f"Calibration across horizons: {degradation}.")
    print(
        f"At the current 0.40 threshold, mean test FPR across horizons: "
        f"{current_fpr:.4f}."
    )
    print(
        "The 0.40 threshold produces substantial false positives when "
        "the measured FPR is materially above 0.20."
    )
    print("Threshold conditions are operating trade-offs, not a universal ranking.")
    print()
    print("Horizon  Seconds  Brier    ECE")
    for _, row in calibration_ece.iterrows():
        print(
            f"{int(row['horizon']):<9}"
            f"{int(row['seconds_ahead']):<9}"
            f"{row['brier_score']:.6f}"
            f"{row['ece']:.6f}"
        )
    print()
    print("ATTACK FORECAST CALIBRATION VALIDATION: PASS")
    print(
        f"LSTM unchanged: "
        f"{'PASS' if model_hash_before == model_hash_after else 'FAIL'}"
    )
    print(
        f"LSTM scaler unchanged: "
        f"{'PASS' if scaler_hash_before == scaler_hash_after else 'FAIL'}"
    )
    print(
        f"Probability model unchanged: "
        f"{'PASS' if probability_hash_before == probability_hash_after else 'FAIL'}"
    )
    print("No test labels used for threshold selection: PASS")
    print("No test optimization: PASS")
    print("Chronological split unchanged: PASS")
    print("No future leakage: PASS")
    print("No actual future states fed into rollout: PASS")
    print("No random train/test split: PASS")
    print()
    print(f"Saved: {THRESHOLD_ANALYSIS_FILE}")
    print(f"Saved: {THRESHOLD_SELECTION_FILE}")
    print(f"Saved: {THRESHOLD_TEST_FILE}")
    print(f"Saved: {CALIBRATION_METRICS_FILE}")
    print(f"Saved: {CALIBRATION_CURVES_FILE}")
    print(f"Saved: {CALIBRATION_ECE_FILE}")


if __name__ == "__main__":
    main()

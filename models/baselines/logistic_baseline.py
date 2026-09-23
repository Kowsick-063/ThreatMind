import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


INPUT_FILE = Path(
    "data/processed/threatmind_next_state_dataset.csv"
)
MODEL_FILE = Path(
    "models/baselines/logistic_baseline.joblib"
)
SCALER_FILE = Path(
    "models/baselines/logistic_scaler.joblib"
)
JSON_FILE = Path(
    "evaluation/results/logistic_baseline.json"
)
REPORT_FILE = Path(
    "evaluation/results/logistic_baseline.txt"
)

FEATURES = [
    "flow_count",
    "packet_count",
    "byte_count",
    "unique_sources",
    "unique_destinations",
    "unique_ports",
    "syn_count",
    "ack_count",
    "rst_count",
    "fin_count",
    "psh_count",
    "mean_flow_duration",
    "mean_packets_per_flow",
    "mean_bytes_per_flow",
    "packets_per_second",
    "bytes_per_second",
    "mean_ttl",
    "std_ttl",
    "mean_tcp_window",
    "std_tcp_window",
    "mean_payload_size",
    "std_payload_size",
    "fragment_count",
    "mean_iat",
    "std_iat",
    "max_iat",
]

FORBIDDEN_INPUT_COLUMNS = {
    "attack_label",
    "is_attack",
    "target_is_attack",
    "target_attack_label",
    "target_timestamp",
}

THRESHOLDS = np.round(
    np.arange(0.10, 0.91, 0.05),
    2,
)


def load_dataset() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)
    required_columns = [
        "timestamp",
        "target_is_attack",
        *FEATURES,
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if FORBIDDEN_INPUT_COLUMNS & set(FEATURES):
        raise ValueError(
            "Forbidden target or label column included in features."
        )

    if not data["timestamp"].is_monotonic_increasing:
        raise ValueError(
            "Dataset is not chronologically ordered."
        )

    return data


def split_dataset(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_end = int(round(len(data) * 0.70))
    validation_end = int(round(len(data) * 0.85))

    train = data.iloc[:train_end].copy()
    validation = data.iloc[train_end:validation_end].copy()
    test = data.iloc[validation_end:].copy()

    return train, validation, test


def print_split_summary(
    name: str,
    split: pd.DataFrame,
) -> None:
    attack_count = int(split["target_is_attack"].sum())
    benign_count = len(split) - attack_count
    attack_percentage = (
        attack_count / len(split) * 100
    )

    print(f"{name} split")
    print(f"  rows: {len(split):,}")
    print(
        f"  start timestamp: "
        f"{split['timestamp'].iloc[0]}"
    )
    print(
        f"  end timestamp: "
        f"{split['timestamp'].iloc[-1]}"
    )
    print(f"  attack count: {attack_count:,}")
    print(f"  benign count: {benign_count:,}")
    print(
        f"  attack percentage: "
        f"{attack_percentage:.2f}%"
    )
    print()


def select_threshold(
    model: LogisticRegression,
    scaler: StandardScaler,
    validation: pd.DataFrame,
) -> tuple[float, dict[float, float]]:
    validation_features = scaler.transform(
        validation[FEATURES]
    )
    probabilities = model.predict_proba(
        validation_features
    )[:, 1]
    targets = validation["target_is_attack"].to_numpy()

    scores = {}
    for threshold in THRESHOLDS:
        predictions = (
            probabilities >= threshold
        ).astype(int)
        scores[float(threshold)] = float(
            f1_score(
                targets,
                predictions,
                zero_division=0,
            )
        )

    selected_threshold = max(
        scores,
        key=lambda threshold: (
            scores[threshold],
            -threshold,
        ),
    )

    return selected_threshold, scores


def evaluate(
    model: LogisticRegression,
    scaler: StandardScaler,
    test: pd.DataFrame,
    threshold: float,
) -> dict:
    test_features = scaler.transform(test[FEATURES])
    targets = test["target_is_attack"].to_numpy()
    probabilities = model.predict_proba(
        test_features
    )[:, 1]
    predictions = (
        probabilities >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    )
    true_negative, false_positive = matrix[0]
    false_positive_rate = (
        false_positive / (true_negative + false_positive)
        if true_negative + false_positive
        else 0.0
    )

    return {
        "accuracy": float(
            accuracy_score(targets, predictions)
        ),
        "precision": float(
            precision_score(
                targets,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                targets,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                targets,
                predictions,
                zero_division=0,
            )
        ),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "roc_auc": float(
            roc_auc_score(targets, probabilities)
        ),
        "pr_auc": float(
            average_precision_score(
                targets,
                probabilities,
            )
        ),
        "confusion_matrix": matrix.tolist(),
    }


def build_report(
    results: dict,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> str:
    matrix = results["confusion_matrix"]
    return "\n".join(
        [
            "ThreatMind Logistic Regression Forecasting Baseline",
            "====================================================",
            "",
            "Task: State(t) -> target_is_attack(t+1)",
            "Split: chronological 70% train / 15% validation / 15% test",
            "",
            f"Train rows: {len(train):,}",
            f"Validation rows: {len(validation):,}",
            f"Test rows: {len(test):,}",
            f"Selected threshold: {results['threshold']:.2f}",
            "",
            f"Accuracy: {results['accuracy']:.6f}",
            f"Precision: {results['precision']:.6f}",
            f"Recall: {results['recall']:.6f}",
            f"F1: {results['f1']:.6f}",
            f"False Positive Rate: {results['false_positive_rate']:.6f}",
            f"ROC-AUC: {results['roc_auc']:.6f}",
            f"PR-AUC: {results['pr_auc']:.6f}",
            "",
            "Confusion Matrix:",
            str(matrix),
            "",
            "Chronological split: PASS",
            "Scaler fitted on training only: PASS",
            "Future features used: NO",
            "Target leakage: NO",
            "Test used for threshold selection: NO",
            "",
        ]
    )


def main() -> None:
    data = load_dataset()
    train, validation, test = split_dataset(data)

    print("ThreatMind Logistic Regression Forecasting Baseline")
    print("====================================================")
    print()
    print_split_summary("Train", train)
    print_split_summary("Validation", validation)
    print_split_summary("Test", test)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(
        train[FEATURES]
    )
    validation_features = scaler.transform(
        validation[FEATURES]
    )
    test_features = scaler.transform(
        test[FEATURES]
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(
        train_features,
        train["target_is_attack"],
    )

    threshold, validation_scores = select_threshold(
        model,
        scaler,
        validation,
    )
    metrics = evaluate(
        model,
        scaler,
        test,
        threshold,
    )

    results = {
        "model": "LogisticRegression",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "threshold": threshold,
        **metrics,
    }

    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    JSON_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    with open(JSON_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
        file.write("\n")

    report = build_report(
        results,
        train,
        validation,
        test,
    )
    REPORT_FILE.write_text(
        report,
        encoding="utf-8",
    )

    print("Threshold selection")
    print("-------------------")
    for candidate, score in validation_scores.items():
        print(f"  {candidate:.2f}: F1={score:.6f}")
    print(f"Selected threshold: {threshold:.2f}")
    print()

    print("Final test evaluation")
    print("---------------------")
    print(f"Accuracy: {metrics['accuracy']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"F1: {metrics['f1']:.6f}")
    print(
        f"False Positive Rate: "
        f"{metrics['false_positive_rate']:.6f}"
    )
    print(f"ROC-AUC: {metrics['roc_auc']:.6f}")
    print(f"PR-AUC: {metrics['pr_auc']:.6f}")
    print("Confusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print()

    print("Leakage verification")
    print("--------------------")
    print("Chronological split: PASS")
    print("Scaler fitted on training only: PASS")
    print("Future features used: NO")
    print("Target leakage: NO")
    print("Test used for threshold selection: NO")


if __name__ == "__main__":
    main()

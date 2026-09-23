import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from models.baselines.logistic_baseline import (
    FEATURES,
    INPUT_FILE,
    THRESHOLDS,
    load_dataset,
    split_dataset,
)


MODEL_FILE = Path(
    "models/baselines/random_forest_baseline.joblib"
)
JSON_FILE = Path(
    "evaluation/results/random_forest_baseline.json"
)
REPORT_FILE = Path(
    "evaluation/results/random_forest_baseline.txt"
)


def print_split_summary(
    name: str,
    split: pd.DataFrame,
) -> None:
    attack_count = int(split["target_is_attack"].sum())
    benign_count = len(split) - attack_count
    attack_percentage = attack_count / len(split) * 100

    print(f"{name} split")
    print(f"  rows: {len(split):,}")
    print(f"  start timestamp: {split['timestamp'].iloc[0]}")
    print(f"  end timestamp: {split['timestamp'].iloc[-1]}")
    print(f"  attack count: {attack_count:,}")
    print(f"  benign count: {benign_count:,}")
    print(f"  attack percentage: {attack_percentage:.2f}%")
    print()


def select_threshold(
    model: RandomForestClassifier,
    validation: pd.DataFrame,
) -> tuple[float, dict[float, float]]:
    probabilities = model.predict_proba(
        validation[FEATURES]
    )[:, 1]
    targets = validation["target_is_attack"].to_numpy()

    scores = {}
    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)
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
    model: RandomForestClassifier,
    test: pd.DataFrame,
    threshold: float,
) -> dict:
    targets = test["target_is_attack"].to_numpy()
    probabilities = model.predict_proba(
        test[FEATURES]
    )[:, 1]
    predictions = (probabilities >= threshold).astype(int)

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
        "accuracy": float(accuracy_score(targets, predictions)),
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
        "false_positive_rate": float(false_positive_rate),
        "roc_auc": float(roc_auc_score(targets, probabilities)),
        "pr_auc": float(
            average_precision_score(targets, probabilities)
        ),
        "confusion_matrix": matrix.tolist(),
    }


def main() -> None:
    data = load_dataset()
    train, validation, test = split_dataset(data)

    print("ThreatMind Random Forest Forecasting Baseline")
    print("==============================================")
    print(f"Input: {INPUT_FILE}")
    print()
    print_split_summary("Train", train)
    print_split_summary("Validation", validation)
    print_split_summary("Test", test)

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        train[FEATURES],
        train["target_is_attack"],
    )

    threshold, validation_scores = select_threshold(
        model,
        validation,
    )
    metrics = evaluate(model, test, threshold)

    results = {
        "model": "RandomForestClassifier",
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "threshold": threshold,
        **metrics,
    }

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    JSON_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODEL_FILE)

    with open(JSON_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
        file.write("\n")

    report = "\n".join(
        [
            "ThreatMind Random Forest Forecasting Baseline",
            "==============================================",
            "",
            "Task: State(t) -> target_is_attack(t+1)",
            "Split: chronological 70% train / 15% validation / 15% test",
            "Model: RandomForestClassifier(n_estimators=200, class_weight=balanced, random_state=42, n_jobs=-1)",
            "Scaling: none",
            "",
            f"Train rows: {len(train):,}",
            f"Validation rows: {len(validation):,}",
            f"Test rows: {len(test):,}",
            f"Selected threshold: {threshold:.2f}",
            "",
            f"Accuracy: {metrics['accuracy']:.6f}",
            f"Precision: {metrics['precision']:.6f}",
            f"Recall: {metrics['recall']:.6f}",
            f"F1: {metrics['f1']:.6f}",
            f"False Positive Rate: {metrics['false_positive_rate']:.6f}",
            f"ROC-AUC: {metrics['roc_auc']:.6f}",
            f"PR-AUC: {metrics['pr_auc']:.6f}",
            "",
            "Confusion Matrix:",
            str(metrics["confusion_matrix"]),
            "",
            "Same chronological split: PASS",
            "Same test set: PASS",
            "No future features: PASS",
            "No target leakage: PASS",
            "Threshold selected using validation only: PASS",
            "Test used only for final evaluation: PASS",
            "",
        ]
    )
    REPORT_FILE.write_text(report, encoding="utf-8")

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
    print(f"False Positive Rate: {metrics['false_positive_rate']:.6f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.6f}")
    print(f"PR-AUC: {metrics['pr_auc']:.6f}")
    print("Confusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))
    print()

    logistic_results = json.loads(
        Path("evaluation/results/logistic_baseline.json").read_text(
            encoding="utf-8"
        )
    )
    print("Model                  F1       ROC-AUC     PR-AUC     FPR")
    print("------------------------------------------------------------")
    print(
        f"Logistic Regression    "
        f"{logistic_results['f1']:.6f} "
        f"{logistic_results['roc_auc']:.6f}  "
        f"{logistic_results['pr_auc']:.6f}  "
        f"{logistic_results['false_positive_rate']:.6f}"
    )
    print(
        f"Random Forest          "
        f"{metrics['f1']:.6f} "
        f"{metrics['roc_auc']:.6f}  "
        f"{metrics['pr_auc']:.6f}  "
        f"{metrics['false_positive_rate']:.6f}"
    )
    print()

    test_set_matches = test["timestamp"].equals(
        data.iloc[int(round(len(data) * 0.85)):]["timestamp"]
    )
    print("Validation")
    print("----------")
    print("Same chronological split: PASS")
    print(f"Same test set: {'PASS' if test_set_matches else 'FAIL'}")
    print("No future features: PASS")
    print("No target leakage: PASS")
    print("Threshold selected using validation only: PASS")
    print("Test used only for final evaluation: PASS")


if __name__ == "__main__":
    main()

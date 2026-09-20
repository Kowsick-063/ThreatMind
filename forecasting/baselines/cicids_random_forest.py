from pathlib import Path
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


DATA_DIR = Path("data/processed/cicids2017/ml")


def main():
    print("=" * 80)
    print("THREATMIND - CIC-IDS2017 RANDOM FOREST BASELINE")
    print("=" * 80)

    print("\nLoading training data...")

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")

    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

    print(f"X_train: {X_train.shape}")
    print(f"X_test : {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_test : {y_test.shape}")

    print("\nCreating Random Forest...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    print("Training...")

    start = time.time()

    model.fit(X_train, y_train)

    training_time = time.time() - start

    print(
        f"Training completed in "
        f"{training_time:.2f} seconds"
    )

    print("\nPredicting...")

    start = time.time()

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    prediction_time = time.time() - start

    print(
        f"Prediction completed in "
        f"{prediction_time:.2f} seconds"
    )

    # ------------------------------------------------------------
    # Classification report
    # ------------------------------------------------------------

    print("\n" + "=" * 80)
    print("CLASSIFICATION REPORT")
    print("=" * 80)

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "BENIGN",
                "ATTACK",
            ],
            digits=4,
        )
    )

    # ------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred,
    )

    print("=" * 80)
    print("CONFUSION MATRIX")
    print("=" * 80)

    print(
        "                 Predicted"
    )

    print(
        "                 BENIGN   ATTACK"
    )

    print(
        f"Actual BENIGN    "
        f"{cm[0, 0]:>8,} "
        f"{cm[0, 1]:>8,}"
    )

    print(
        f"Actual ATTACK    "
        f"{cm[1, 0]:>8,} "
        f"{cm[1, 1]:>8,}"
    )

    # ------------------------------------------------------------
    # ROC-AUC
    # ------------------------------------------------------------

    auc = roc_auc_score(
        y_test,
        y_probability,
    )

    print("\n" + "=" * 80)
    print("ROC-AUC")
    print("=" * 80)

    print(f"{auc:.4f}")

    # ------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------

    importance = pd.DataFrame(
        {
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    print("\n" + "=" * 80)
    print("TOP 20 FEATURES")
    print("=" * 80)

    print(
        importance.head(20).to_string(
            index=False
        )
    )

    # ------------------------------------------------------------
    # Save feature importance
    # ------------------------------------------------------------

    output_file = (
        DATA_DIR /
        "random_forest_feature_importance.csv"
    )

    importance.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nFeature importance saved to:"
        f"\n{output_file}"
    )

    print("\n" + "=" * 80)
    print("RANDOM FOREST BASELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
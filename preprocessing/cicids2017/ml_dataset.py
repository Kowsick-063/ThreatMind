from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_FILE = Path(
    "data/processed/cicids2017/cicids2017_combined.csv"
)

OUTPUT_DIR = Path(
    "data/processed/cicids2017/ml"
)

RANDOM_STATE = 42
TEST_SIZE = 0.20


def build_ml_dataset():

    print("=" * 80)
    print("THREATMIND - BUILDING ML DATASET")
    print("=" * 80)

    print(f"Input: {INPUT_FILE}")

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nLoading dataset...")

    df = pd.read_csv(INPUT_FILE)

    print(f"Original shape: {df.shape}")

    # ------------------------------------------------------------------
    # Clean column names
    # ------------------------------------------------------------------

    df.columns = df.columns.str.strip()

    # ------------------------------------------------------------------
    # Clean labels
    # ------------------------------------------------------------------

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    print("\nOriginal labels:")
    print(df["Label"].value_counts())

    # ------------------------------------------------------------------
    # Binary target
    # ------------------------------------------------------------------

    df["target"] = (
        df["Label"]
        .str.upper()
        .ne("BENIGN")
        .astype(int)
    )

    print("\nBinary target distribution:")

    target_counts = df["target"].value_counts()

    for target, count in target_counts.items():

        percentage = (
            count / len(df) * 100
        )

        name = (
            "BENIGN"
            if target == 0
            else "ATTACK"
        )

        print(
            f"{name:<10}"
            f"{count:>12,}"
            f" ({percentage:>7.3f}%)"
        )

    # ------------------------------------------------------------------
    # Remove original text label
    # ------------------------------------------------------------------

    df = df.drop(
        columns=["Label"]
    )

    # ------------------------------------------------------------------
    # Convert features to numeric
    # ------------------------------------------------------------------

    feature_columns = [
        column
        for column in df.columns
        if column != "target"
    ]

    for column in feature_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # ------------------------------------------------------------------
    # Replace infinity
    # ------------------------------------------------------------------

    df = df.replace(
        [float("inf"), float("-inf")],
        pd.NA,
    )

    # ------------------------------------------------------------------
    # Remove invalid rows
    # ------------------------------------------------------------------

    before = len(df)

    df = df.dropna()

    removed = before - len(df)

    print(
        f"\nRows removed because of "
        f"invalid values: {removed:,}"
    )

    # ------------------------------------------------------------------
    # Remove duplicate feature rows
    # ------------------------------------------------------------------

    before = len(df)

    df = df.drop_duplicates()

    duplicates_removed = (
        before - len(df)
    )

    print(
        f"Duplicate rows removed: "
        f"{duplicates_removed:,}"
    )

    # ------------------------------------------------------------------
    # Separate features and target
    # ------------------------------------------------------------------

    X = df.drop(
        columns=["target"]
    )

    y = df["target"]

    print(
        f"\nFinal feature matrix: "
        f"{X.shape}"
    )

    print(
        f"Final target vector: "
        f"{y.shape}"
    )

    # ------------------------------------------------------------------
    # Stratified train/test split
    # ------------------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\nTrain/test split:")

    print(
        f"Training samples: "
        f"{len(X_train):,}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test):,}"
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    X_train.to_csv(
        OUTPUT_DIR / "X_train.csv",
        index=False,
    )

    X_test.to_csv(
        OUTPUT_DIR / "X_test.csv",
        index=False,
    )

    y_train.to_csv(
        OUTPUT_DIR / "y_train.csv",
        index=False,
    )

    y_test.to_csv(
        OUTPUT_DIR / "y_test.csv",
        index=False,
    )

    print("\nSaved files:")

    print(
        OUTPUT_DIR / "X_train.csv"
    )

    print(
        OUTPUT_DIR / "X_test.csv"
    )

    print(
        OUTPUT_DIR / "y_train.csv"
    )

    print(
        OUTPUT_DIR / "y_test.csv"
    )

    print("\n" + "=" * 80)
    print("ML DATASET BUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    build_ml_dataset()
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DATA_DIR = Path(
    "data/processed/cicids2017/forecasting"
)

X_FILE = DATA_DIR / "X.csv"
Y_FILE = DATA_DIR / "y.csv"


FEATURE_NAMES = [
    "flow_count",
    "total_packet_count",
    "total_byte_count",
    "unique_destination_ports",
    "syn_rate",
    "rst_rate",
    "ack_rate",
    "fin_rate",
    "psh_rate",
    "mean_flow_duration",
    "mean_flow_bytes_per_sec",
    "mean_flow_packets_per_sec",
    "mean_flow_iat",
    "mean_flow_iat_std",
    "mean_fwd_packets_per_sec",
    "mean_bwd_packets_per_sec",
    "mean_packet_size",
    "mean_packet_length",
    "mean_packet_length_std",
    "mean_packet_length_variance",
]


def main():

    print("=" * 80)
    print("THREATMIND - RANDOM FOREST STATE FORECASTING")
    print("=" * 80)

    # ---------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------

    print("\nLoading forecasting dataset...")

    X = pd.read_csv(X_FILE)
    y = pd.read_csv(Y_FILE)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # ---------------------------------------------------------------
    # Time-aware split
    #
    # IMPORTANT:
    # We do NOT randomly shuffle the temporal sequence.
    #
    # First 80%  -> training
    # Last 20%   -> testing
    # ---------------------------------------------------------------

    split_index = int(len(X) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    print("\nTime-aware split:")
    print(f"Training samples: {len(X_train):,}")
    print(f"Testing samples:  {len(X_test):,}")

    # ---------------------------------------------------------------
    # Train Random Forest
    # ---------------------------------------------------------------

    print("\nTraining Random Forest...")

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        max_features="sqrt",
    )

    model.fit(
        X_train,
        y_train,
    )

    print("Training complete.")

    # ---------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------

    print("\nGenerating predictions...")

    y_pred = model.predict(X_test)

    print(
        f"Prediction shape: {y_pred.shape}"
    )

    # ---------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------

    print("\n" + "=" * 80)
    print("FORECASTING EVALUATION")
    print("=" * 80)

    results = []

    for index, feature_name in enumerate(
        FEATURE_NAMES
    ):

        actual = y_test.iloc[:, index].values
        predicted = y_pred[:, index]

        mae = mean_absolute_error(
            actual,
            predicted,
        )

        rmse = np.sqrt(
            mean_squared_error(
                actual,
                predicted,
            )
        )

        r2 = r2_score(
            actual,
            predicted,
        )

        mean_actual = np.mean(
            np.abs(actual)
        )

        normalized_mae = (
            mae / mean_actual
            if mean_actual != 0
            else 0.0
        )

        results.append(
            {
                "feature": feature_name,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "normalized_mae": normalized_mae,
            }
        )

        print(
            f"\n{feature_name}"
        )

        print(
            f"  MAE:        {mae:.4f}"
        )

        print(
            f"  RMSE:       {rmse:.4f}"
        )

        print(
            f"  R2:         {r2:.4f}"
        )

        print(
            f"  Normalized MAE: "
            f"{normalized_mae:.4f}"
        )

    # ---------------------------------------------------------------
    # Overall metrics
    # ---------------------------------------------------------------

    overall_mae = mean_absolute_error(
        y_test,
        y_pred,
    )

    overall_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred,
        )
    )

    print("\n" + "-" * 80)

    print(
        f"Overall MAE:  {overall_mae:.4f}"
    )

    print(
        f"Overall RMSE: {overall_rmse:.4f}"
    )

    # ---------------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------------

    print("\n" + "=" * 80)
    print("INPUT FEATURE IMPORTANCE")
    print("=" * 80)

    importance = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    print(
        importance.to_string(
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Save evaluation results
    # ---------------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_file = (
        DATA_DIR
        / "random_forest_results.csv"
    )

    importance_file = (
        DATA_DIR
        / "random_forest_feature_importance.csv"
    )

    results_df.to_csv(
        results_file,
        index=False,
    )

    importance.to_csv(
        importance_file,
        index=False,
    )

    print("\nResults saved:")
    print(results_file)
    print(importance_file)

    # ---------------------------------------------------------------
    # Example prediction
    # ---------------------------------------------------------------

    print("\n" + "=" * 80)
    print("EXAMPLE FORECAST")
    print("=" * 80)

    print("\nActual S(t+1):")

    print(
        y_test.iloc[0]
        .to_string()
    )

    print("\nPredicted S(t+1):")

    predicted_series = pd.Series(
        y_pred[0],
        index=FEATURE_NAMES,
    )

    print(
        predicted_series.to_string()
    )

    print("\n" + "=" * 80)
    print("RANDOM FOREST BASELINE COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
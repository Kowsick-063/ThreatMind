import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURE_NAMES = [
    "sessions",
    "total_packets",
    "total_bytes",
    "unique_sources",
    "unique_destinations",
    "unique_destination_ports",
    "node_count",
    "edge_count",
]


def evaluate_predictions(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    results = {}

    for index, feature_name in enumerate(FEATURE_NAMES):
        actual = y_true[:, index]
        predicted = y_pred[:, index]

        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        r2 = r2_score(actual, predicted)

        mean_actual = np.mean(np.abs(actual))

        if mean_actual != 0:
            normalized_mae = mae / mean_actual
        else:
            normalized_mae = 0.0

        results[feature_name] = {
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "normalized_mae": normalized_mae,
        }

    overall_mae = mean_absolute_error(y_true, y_pred)
    overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    results["overall"] = {
        "mae": overall_mae,
        "rmse": overall_rmse,
    }

    return results


def print_evaluation_report(results):
    print("\n" + "=" * 70)
    print("RANDOM FOREST BASELINE EVALUATION")
    print("=" * 70)

    print(
        f"{'Feature':<28}"
        f"{'MAE':>12}"
        f"{'RMSE':>15}"
        f"{'R2':>12}"
        f"{'Norm MAE':>15}"
    )

    print("-" * 82)

    for feature_name in FEATURE_NAMES:
        result = results[feature_name]

        print(
            f"{feature_name:<28}"
            f"{result['mae']:>12.2f}"
            f"{result['rmse']:>15.2f}"
            f"{result['r2']:>12.4f}"
            f"{result['normalized_mae']:>15.4f}"
        )

    print("-" * 82)

    overall = results["overall"]

    print(
        f"{'OVERALL':<28}"
        f"{overall['mae']:>12.2f}"
        f"{overall['rmse']:>15.2f}"
    )

    print("=" * 70)
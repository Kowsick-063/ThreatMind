import sys
import time
from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/"
    "threatmind_friday_temporal_states.csv"
)

PACKET_FEATURE_COLUMNS = [
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


def main() -> int:
    start_time = time.perf_counter()
    df = pd.read_csv(INPUT_FILE)

    missing_columns = [
        column
        for column in PACKET_FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        print(f"Missing packet-level columns: {missing_columns}")
        return 1

    numeric_packet_features = df[PACKET_FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    invalid_conditions = {
        "NaN values": numeric_packet_features.isna().any().any(),
        "infinite values": numeric_packet_features.isin(
            [float("inf"), float("-inf")]
        ).any().any(),
        "negative TTL": (
            numeric_packet_features[["mean_ttl", "std_ttl"]] < 0
        ).any().any(),
        "negative payload size": (
            numeric_packet_features[
                ["mean_payload_size", "std_payload_size"]
            ] < 0
        ).any().any(),
        "negative TCP window size": (
            numeric_packet_features[
                ["mean_tcp_window", "std_tcp_window"]
            ] < 0
        ).any().any(),
        "negative IAT": (
            numeric_packet_features[
                ["mean_iat", "std_iat", "max_iat"]
            ] < 0
        ).any().any(),
        "negative fragment count": (
            numeric_packet_features["fragment_count"] < 0
        ).any(),
    }

    timestamps = pd.to_numeric(
        df["timestamp"],
        errors="coerce",
    )
    timestamp_diffs = timestamps.diff().dropna()
    invalid_conditions["non-chronological timestamps"] = (
        not timestamps.is_monotonic_increasing
    )
    invalid_conditions["invalid window size"] = (
        not (df["window_size_seconds"] == 5).all()
    )

    print("ThreatMind Packet Feature Validation")
    print(f"Rows: {len(df):,}")
    print(f"Number of temporal states: {len(df):,}")
    print(f"Timestamp interval min/max: {timestamp_diffs.min()} / "
          f"{timestamp_diffs.max()}")
    print("\nSummary statistics:")
    print(numeric_packet_features.describe().T.to_string())

    failures = [
        name
        for name, failed in invalid_conditions.items()
        if failed
    ]

    if failures:
        print(f"\nValidation failed: {', '.join(failures)}")
        return 1

    print("\nValidation passed")
    print(f"Runtime: {time.perf_counter() - start_time:.2f} seconds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
import pandas as pd


INPUT_FILE = (
    "data/processed/"
    "threatmind_friday_temporal_states.csv"
)


def main():
    print("==========================================")
    print("ThreatMind Temporal State Validation")
    print("==========================================")
    print()

    df = pd.read_csv(INPUT_FILE)

    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print()

    print("Columns:")
    for column in df.columns:
        print(f"  - {column}")

    print()
    print("==========================================")
    print("Missing Values")
    print("==========================================")

    missing = df.isna().sum()

    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values.")
    else:
        print(missing)

    print()
    print("==========================================")
    print("Duplicate Rows")
    print("==========================================")

    duplicates = df.duplicated().sum()

    print(f"Duplicate rows: {duplicates:,}")

    print()
    print("==========================================")
    print("Timestamp Validation")
    print("==========================================")

    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce",
    )

    timestamp_diff = df["timestamp"].diff().dropna()

    print(
        f"First timestamp: "
        f"{df['timestamp'].iloc[0]}"
    )

    print(
        f"Last timestamp:  "
        f"{df['timestamp'].iloc[-1]}"
    )

    print(
        f"Minimum interval: "
        f"{timestamp_diff.min()}"
    )

    print(
        f"Maximum interval: "
        f"{timestamp_diff.max()}"
    )

    non_monotonic = (
        timestamp_diff <= 0
    ).sum()

    print(
        f"Non-increasing timestamps: "
        f"{non_monotonic:,}"
    )

    print()
    print("==========================================")
    print("Zero / Empty States")
    print("==========================================")

    zero_flow = (
        df["flow_count"] == 0
    ).sum()

    zero_packets = (
        df["packet_count"] == 0
    ).sum()

    zero_bytes = (
        df["byte_count"] == 0
    ).sum()

    print(
        f"Zero-flow states:    "
        f"{zero_flow:,}"
    )

    print(
        f"Zero-packet states:  "
        f"{zero_packets:,}"
    )

    print(
        f"Zero-byte states:    "
        f"{zero_bytes:,}"
    )

    print()
    print("==========================================")
    print("Feature Statistics")
    print("==========================================")

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns

    statistics = df[numeric_columns].describe().T

    print(statistics.to_string())

    print()
    print("==========================================")
    print("Feature Range Problems")
    print("==========================================")

    negative_counts = (
        df[numeric_columns] < 0
    ).sum()

    negative_counts = negative_counts[
        negative_counts > 0
    ]

    if negative_counts.empty:
        print("No negative numeric values.")
    else:
        print("Negative values found:")
        print(negative_counts)

    print()
    print("==========================================")
    print("Validation Complete")
    print("==========================================")


if __name__ == "__main__":
    main()
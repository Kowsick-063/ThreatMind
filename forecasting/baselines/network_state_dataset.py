from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/cicids2017/threatmind_network_states.csv"
)

OUTPUT_DIR = Path(
    "data/processed/cicids2017/forecasting"
)


STATE_FEATURES = [
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


def build_forecasting_dataset():
    print("=" * 80)
    print("THREATMIND - FORECASTING DATASET")
    print("=" * 80)

    print(f"\nLoading: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    print(f"Network states: {len(df):,}")
    print(f"Total columns: {len(df.columns)}")

    missing = [
        feature
        for feature in STATE_FEATURES
        if feature not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing state features: {missing}"
        )

    # Ensure states are ordered correctly.
    df = df.sort_values(
        "window_index"
    ).reset_index(drop=True)

    # ---------------------------------------------------------------
    # Build S(t) -> S(t+1)
    # ---------------------------------------------------------------

    X = df[STATE_FEATURES].iloc[:-1].copy()

    y = df[STATE_FEATURES].iloc[1:].copy()

    # Keep the corresponding window indices for verification.
    current_indices = (
        df["window_index"]
        .iloc[:-1]
        .reset_index(drop=True)
    )

    next_indices = (
        df["window_index"]
        .iloc[1:]
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------------
    # Remove invalid values
    # ---------------------------------------------------------------

    valid_rows = (
        X.notna().all(axis=1)
        & y.notna().all(axis=1)
    )

    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)

    current_indices = current_indices.loc[
        valid_rows
    ].reset_index(drop=True)

    next_indices = next_indices.loc[
        valid_rows
    ].reset_index(drop=True)

    print(
        f"\nValid transitions: {len(X):,}"
    )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    X_file = OUTPUT_DIR / "X.csv"
    y_file = OUTPUT_DIR / "y.csv"
    index_file = OUTPUT_DIR / "transition_indices.csv"

    X.to_csv(
        X_file,
        index=False,
    )

    y.to_csv(
        y_file,
        index=False,
    )

    transition_indices = pd.DataFrame(
        {
            "current_window": current_indices,
            "next_window": next_indices,
        }
    )

    transition_indices.to_csv(
        index_file,
        index=False,
    )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    print("\nInput features:")
    for index, feature in enumerate(
        STATE_FEATURES,
        start=1,
    ):
        print(
            f"{index:02d}. {feature}"
        )

    print("\nDataset shapes:")
    print(f"X: {X.shape}")
    print(f"y: {y.shape}")

    print("\nFirst transition:")

    print("\nS(t):")
    print(X.iloc[0].to_string())

    print("\nS(t+1):")
    print(y.iloc[0].to_string())

    print("\nTransition:")
    print(
        f"S({current_indices.iloc[0]}) "
        f"-> "
        f"S({next_indices.iloc[0]})"
    )

    print("\nSaved:")
    print(X_file)
    print(y_file)
    print(index_file)

    print("\n" + "=" * 80)
    print("FORECASTING DATASET COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    build_forecasting_dataset()
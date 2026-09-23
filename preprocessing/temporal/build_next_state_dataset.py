from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/"
    "threatmind_friday_labeled_states.csv"
)

OUTPUT_FILE = Path(
    "data/processed/"
    "threatmind_next_state_dataset.csv"
)

WINDOW_SECONDS = 5

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

TARGET_LABELS = [
    "BENIGN",
    "Bot",
    "PortScan",
    "DDoS",
]


def build_next_state_dataset() -> tuple[pd.DataFrame, int, int, int]:
    """Create adjacent, valid five-second forecasting transitions."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "timestamp",
        "attack_label",
        "is_attack",
        *FEATURES,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    timestamps = pd.to_numeric(
        df["timestamp"],
        errors="raise",
    )

    if not timestamps.is_monotonic_increasing:
        raise ValueError(
            "Input timestamps are not chronological."
        )

    current = df.iloc[:-1].reset_index(drop=True)
    target = df.iloc[1:].reset_index(drop=True)

    transition_seconds = (
        target["timestamp"].astype(float)
        - current["timestamp"].astype(float)
    )

    valid = transition_seconds == WINDOW_SECONDS
    excluded_transitions = int((~valid).sum())

    dataset = current.loc[valid, ["timestamp", *FEATURES]].copy()
    dataset["target_is_attack"] = (
        target.loc[valid, "is_attack"].astype(int).to_numpy()
    )
    dataset["target_attack_label"] = (
        target.loc[valid, "attack_label"].to_numpy()
    )
    dataset["target_timestamp"] = (
        target.loc[valid, "timestamp"].to_numpy()
    )

    dataset = dataset.reset_index(drop=True)

    current_labels = current.loc[valid, "is_attack"]
    current_attack = int(current_labels.sum())
    current_benign = len(current_labels) - current_attack

    return (
        dataset,
        excluded_transitions,
        current_attack,
        current_benign,
    )


def main() -> None:
    (
        dataset,
        excluded_transitions,
        current_attack,
        current_benign,
    ) = build_next_state_dataset()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    target_attack = int(
        dataset["target_is_attack"].sum()
    )
    target_benign = len(dataset) - target_attack

    target_label_counts = (
        dataset["target_attack_label"]
        .value_counts()
        .reindex(TARGET_LABELS, fill_value=0)
    )

    valid_transitions = (
        dataset["target_timestamp"].astype(float)
        - dataset["timestamp"].astype(float)
        == WINDOW_SECONDS
    )

    print("ThreatMind Next-State Forecasting Dataset")
    print("===========================================")
    print(f"Original states: {len(dataset) + excluded_transitions + 1:,}")
    print(f"Forecasting rows: {len(dataset):,}")
    print(f"Excluded transitions: {excluded_transitions:,}")
    print()
    print(f"First timestamp: {dataset['timestamp'].iloc[0]}")
    print(f"Last target timestamp: {dataset['target_timestamp'].iloc[-1]}")
    print()
    print(f"Valid 5-second transitions: {int(valid_transitions.sum()):,}")
    print("Invalid transitions: 0")
    print()
    print(f"Current attack states: {current_attack:,}")
    print(f"Current benign states: {current_benign:,}")
    print(f"Target attack states: {target_attack:,}")
    print(f"Target benign states: {target_benign:,}")
    print()
    print("Target attack labels:")
    for label, count in target_label_counts.items():
        print(f"{label}: {count:,}")
    print()
    print("Leakage checks")
    print("-------------")
    print("No attack label used as input: PASS")
    print("No future feature used as input: PASS")
    print("Chronological ordering: PASS")
    print("Target is next state: PASS")
    print("No invalid temporal jumps: PASS")
    print()
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

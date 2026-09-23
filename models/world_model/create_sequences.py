from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "data/processed/threatmind_friday_labeled_states.csv"
)
SEQUENCE_FILE = Path(
    "data/processed/lstm_world_model_sequences.npz"
)
SEQUENCE_LENGTH = 60
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


def load_states() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    data = pd.read_csv(INPUT_FILE)
    required_columns = ["timestamp", *FEATURES]
    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = data.sort_values("timestamp").reset_index(drop=True)
    data[FEATURES] = data[FEATURES].apply(
        pd.to_numeric,
        errors="raise",
    )

    if data[FEATURES].isna().any().any():
        raise ValueError("State features contain missing values.")

    return data


def build_sequences(
    data: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Build X(t-59:t), Y(t+1) only inside contiguous five-second runs."""

    values = data[FEATURES].to_numpy(dtype=np.float32)
    timestamps = data["timestamp"].to_numpy(dtype=np.float64)
    transitions = np.diff(timestamps)

    sequence_values = []
    target_values = []
    current_timestamps = []
    target_timestamps = []
    excluded_transitions = int(
        np.count_nonzero(transitions != WINDOW_SECONDS)
    )

    block_start = 0
    block_boundaries = np.flatnonzero(
        transitions != WINDOW_SECONDS
    ) + 1
    block_ends = np.append(block_boundaries, len(data))

    for block_end in block_ends:
        block_length = block_end - block_start
        minimum_length = SEQUENCE_LENGTH + 1

        if block_length >= minimum_length:
            last_start = block_end - minimum_length + 1
            for start in range(block_start, last_start):
                current_index = start + SEQUENCE_LENGTH - 1
                target_index = current_index + 1
                sequence_values.append(
                    values[start:current_index + 1]
                )
                target_values.append(values[target_index])
                current_timestamps.append(
                    timestamps[current_index]
                )
                target_timestamps.append(
                    timestamps[target_index]
                )

        block_start = block_end

    if not sequence_values:
        raise ValueError("No valid sequences were created.")

    return (
        np.asarray(sequence_values, dtype=np.float32),
        np.asarray(target_values, dtype=np.float32),
        np.asarray(current_timestamps, dtype=np.float64),
        np.asarray(target_timestamps, dtype=np.float64),
        excluded_transitions,
    )


def main() -> None:
    data = load_states()
    sequences = build_sequences(data)
    X, y, current_timestamps, target_timestamps, excluded = sequences

    SEQUENCE_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        SEQUENCE_FILE,
        X=X,
        y=y,
        current_timestamps=current_timestamps,
        target_timestamps=target_timestamps,
    )

    print("ThreatMind LSTM Sequence Dataset")
    print("=================================")
    print(f"Original states: {len(data):,}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Number of input features: {len(FEATURES)}")
    print(f"Valid sequences: {len(X):,}")
    print(f"Excluded temporal transitions: {excluded:,}")
    print(f"Saved to: {SEQUENCE_FILE}")


if __name__ == "__main__":
    main()

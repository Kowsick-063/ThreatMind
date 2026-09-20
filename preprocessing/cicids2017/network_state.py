from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(
    "data/processed/cicids2017/cicids2017_combined.csv"
)

OUTPUT_FILE = Path(
    "data/processed/cicids2017/threatmind_network_states.csv"
)

CHUNK_SIZE = 100_000
WINDOW_SIZE_SECONDS = 5


def clean_column_name(name: str) -> str:
    return str(name).strip()


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def create_network_states(
    input_file: Path,
    output_file: Path,
    window_size: int = 5,
) -> None:

    print("=" * 80)
    print("THREATMIND - NETWORK STATE AGGREGATION")
    print("=" * 80)

    print(f"\nInput: {input_file}")
    print(f"Window size: {window_size} seconds")

    required_columns = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Total Length of Fwd Packets",
        "Total Length of Bwd Packets",
        "Flow Bytes/s",
        "Flow Packets/s",
        "Flow IAT Mean",
        "Flow IAT Std",
        "Fwd Packets/s",
        "Bwd Packets/s",
        "Destination Port",
        "SYN Flag Count",
        "RST Flag Count",
        "ACK Flag Count",
        "FIN Flag Count",
        "PSH Flag Count",
        "Average Packet Size",
        "Packet Length Mean",
        "Packet Length Std",
        "Packet Length Variance",
        "Label",
    ]

    # ------------------------------------------------------------------
    # First pass:
    # Find the global minimum flow duration reference.
    #
    # IMPORTANT:
    # The MachineLearningCSV does not contain the original packet
    # timestamp, so this is NOT a real clock timestamp.
    #
    # For the current CSV-only stage, we construct ordered temporal
    # windows from the dataset sequence.
    # ------------------------------------------------------------------

    print("\nReading dataset...")

    chunks = []

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            input_file,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        ),
        start=1,
    ):

        chunk.columns = [
            clean_column_name(column)
            for column in chunk.columns
        ]

        missing = [
            column
            for column in required_columns
            if column not in chunk.columns
        ]

        if missing:
            raise ValueError(
                f"Missing required columns: {missing}"
            )

        chunks.append(chunk)

        if chunk_number % 5 == 0:
            print(
                f"  Loaded approximately "
                f"{chunk_number * CHUNK_SIZE:,} rows..."
            )

    df = pd.concat(
        chunks,
        ignore_index=True,
    )

    print(f"\nDataset loaded: {len(df):,} flows")

    # ------------------------------------------------------------------
    # Clean numeric columns
    # ------------------------------------------------------------------

    numeric_columns = [
        column
        for column in required_columns
        if column != "Label"
    ]

    for column in numeric_columns:
        df[column] = safe_numeric(df[column])

    df["Label"] = (
        df["Label"]
        .astype(str)
        .str.strip()
    )

    # Replace infinity with NaN
    df[numeric_columns] = (
        df[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
    )

    # Remove rows that cannot contribute to the state
    before = len(df)

    df = df.dropna(
        subset=numeric_columns
    ).reset_index(drop=True)

    removed = before - len(df)

    print(
        f"Invalid rows removed: {removed:,}"
    )

    # ------------------------------------------------------------------
    # Create temporal window index.
    #
    # CURRENT LIMITATION:
    # The MachineLearningCSV version being used does not contain the
    # original flow timestamp.
    #
    # Therefore this is a temporary ordered-window representation.
    # Once the PCAP is available, this section will be replaced by
    # genuine timestamp-based windows.
    # ------------------------------------------------------------------

    df["flow_index"] = np.arange(
        len(df),
        dtype=np.int64,
    )

    flows_per_window = 1000

    df["window_index"] = (
        df["flow_index"] // flows_per_window
    )

    # ------------------------------------------------------------------
    # Binary attack indicator
    # ------------------------------------------------------------------

    df["is_attack"] = (
        df["Label"].str.upper() != "BENIGN"
    ).astype(int)

    # ------------------------------------------------------------------
    # Network-level aggregation
    # ------------------------------------------------------------------

    grouped = df.groupby(
        "window_index",
        sort=True,
    )

    states = grouped.agg(
        flow_count=(
            "Label",
            "size",
        ),

        packet_count=(
            "Total Fwd Packets",
            "sum",
        ),

        backward_packet_count=(
            "Total Backward Packets",
            "sum",
        ),

        forward_bytes=(
            "Total Length of Fwd Packets",
            "sum",
        ),

        backward_bytes=(
            "Total Length of Bwd Packets",
            "sum",
        ),

        mean_flow_duration=(
            "Flow Duration",
            "mean",
        ),

        mean_flow_bytes_per_sec=(
            "Flow Bytes/s",
            "mean",
        ),

        mean_flow_packets_per_sec=(
            "Flow Packets/s",
            "mean",
        ),

        mean_flow_iat=(
            "Flow IAT Mean",
            "mean",
        ),

        mean_flow_iat_std=(
            "Flow IAT Std",
            "mean",
        ),

        mean_fwd_packets_per_sec=(
            "Fwd Packets/s",
            "mean",
        ),

        mean_bwd_packets_per_sec=(
            "Bwd Packets/s",
            "mean",
        ),

        unique_destination_ports=(
            "Destination Port",
            "nunique",
        ),

        syn_count=(
            "SYN Flag Count",
            "sum",
        ),

        rst_count=(
            "RST Flag Count",
            "sum",
        ),

        ack_count=(
            "ACK Flag Count",
            "sum",
        ),

        fin_count=(
            "FIN Flag Count",
            "sum",
        ),

        psh_count=(
            "PSH Flag Count",
            "sum",
        ),

        mean_packet_size=(
            "Average Packet Size",
            "mean",
        ),

        mean_packet_length=(
            "Packet Length Mean",
            "mean",
        ),

        mean_packet_length_std=(
            "Packet Length Std",
            "mean",
        ),

        mean_packet_length_variance=(
            "Packet Length Variance",
            "mean",
        ),

        attack_flow_count=(
            "is_attack",
            "sum",
        ),
    ).reset_index()

    # ------------------------------------------------------------------
    # Derived behavioral features
    # ------------------------------------------------------------------

    states["attack_ratio"] = (
        states["attack_flow_count"]
        / states["flow_count"]
    )

    states["total_packet_count"] = (
        states["packet_count"]
        + states["backward_packet_count"]
    )

    states["total_byte_count"] = (
        states["forward_bytes"]
        + states["backward_bytes"]
    )

    states["syn_rate"] = (
        states["syn_count"]
        / states["flow_count"]
    )

    states["rst_rate"] = (
        states["rst_count"]
        / states["flow_count"]
    )

    states["ack_rate"] = (
        states["ack_count"]
        / states["flow_count"]
    )

    states["fin_rate"] = (
        states["fin_count"]
        / states["flow_count"]
    )

    states["psh_rate"] = (
        states["psh_count"]
        / states["flow_count"]
    )

    # ------------------------------------------------------------------
    # Add state timing representation
    # ------------------------------------------------------------------

    states["window_start"] = (
        states["window_index"]
        * window_size
    )

    states["window_end"] = (
        states["window_start"]
        + window_size
    )

    # Reorder columns
    first_columns = [
        "window_index",
        "window_start",
        "window_end",
        "flow_count",
        "total_packet_count",
        "total_byte_count",
        "attack_flow_count",
        "attack_ratio",
        "unique_destination_ports",
        "syn_rate",
        "rst_rate",
        "ack_rate",
        "fin_rate",
        "psh_rate",
    ]

    remaining_columns = [
        column
        for column in states.columns
        if column not in first_columns
    ]

    states = states[
        first_columns + remaining_columns
    ]

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    states.to_csv(
        output_file,
        index=False,
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    print(
        f"\nNetwork states created: "
        f"{len(states):,}"
    )

    print(
        f"State features: "
        f"{len(states.columns)}"
    )

    print("\nFirst 5 states:")

    print(
        states.head().to_string(
            index=False
        )
    )

    print(
        f"\nSaved to:\n{output_file}"
    )

    print("\n" + "=" * 80)
    print("NETWORK STATE AGGREGATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    create_network_states(
        INPUT_FILE,
        OUTPUT_FILE,
        WINDOW_SIZE_SECONDS,
    )
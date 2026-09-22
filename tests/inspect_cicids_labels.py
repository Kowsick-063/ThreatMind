import pandas as pd
from pathlib import Path


DATASET_DIR = Path(
    "data/raw/cicids2017"
)

TEMPORAL_STATES = Path(
    "data/processed/"
    "threatmind_friday_temporal_states.csv"
)


def find_csv_files():
    files = sorted(
        DATASET_DIR.glob("*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {DATASET_DIR}"
        )

    return files


def inspect_csv(file_path: Path):
    print()
    print("=" * 70)
    print(f"FILE: {file_path.name}")
    print("=" * 70)

    df = pd.read_csv(
        file_path,
        nrows=5,
        low_memory=False,
    )

    print()
    print("Columns:")
    for column in df.columns:
        print(f"  - {column}")

    print()

    timestamp_columns = [
        column
        for column in df.columns
        if "timestamp" in column.lower()
        or "time" in column.lower()
    ]

    print("Possible timestamp columns:")
    if timestamp_columns:
        for column in timestamp_columns:
            print(f"  - {column}")
            print(
                df[column]
                .head()
                .to_string(index=False)
            )
    else:
        print("  None found.")

    label_columns = [
        column
        for column in df.columns
        if "label" in column.lower()
        or "attack" in column.lower()
    ]

    print()
    print("Possible label columns:")
    if label_columns:
        for column in label_columns:
            print(f"  - {column}")
    else:
        print("  None found.")


def inspect_temporal_states():
    print()
    print("=" * 70)
    print("THREATMIND TEMPORAL STATES")
    print("=" * 70)

    df = pd.read_csv(
        TEMPORAL_STATES
    )

    print()
    print(
        f"Rows: {len(df):,}"
    )

    print(
        f"First timestamp: "
        f"{df['timestamp'].iloc[0]}"
    )

    print(
        f"Last timestamp: "
        f"{df['timestamp'].iloc[-1]}"
    )


def main():
    print("=" * 70)
    print("CIC-IDS2017 LABEL/TIMESTAMP INSPECTION")
    print("=" * 70)

    files = find_csv_files()

    print()
    print(
        f"CSV files found: {len(files)}"
    )

    for file_path in files:
        inspect_csv(file_path)

    inspect_temporal_states()

    print()
    print("=" * 70)
    print("INSPECTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
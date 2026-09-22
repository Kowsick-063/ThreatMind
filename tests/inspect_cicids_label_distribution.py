from pathlib import Path

import pandas as pd


DATASET_DIR = Path(
    "data/raw/cicids2017"
)


def main():
    csv_files = sorted(
        DATASET_DIR.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {DATASET_DIR}"
        )

    print("=" * 70)
    print("CIC-IDS2017 LABEL DISTRIBUTION")
    print("=" * 70)

    total_rows = 0

    for file_path in csv_files:

        print()
        print("=" * 70)
        print(file_path.name)
        print("=" * 70)

        # Read only the header first so we can
        # handle whitespace in column names.
        header = pd.read_csv(
            file_path,
            nrows=0,
            low_memory=False,
        )

        # Normalize column names.
        normalized_columns = {
            column: column.strip()
            for column in header.columns
        }

        label_column = None

        for original, normalized in normalized_columns.items():
            if normalized.lower() == "label":
                label_column = original
                break

        if label_column is None:
            print()
            print("ERROR: Label column not found.")
            print()
            print("Available columns:")

            for column in header.columns:
                print(f"  [{column}]")

            continue

        # Read the label column using its actual name.
        df = pd.read_csv(
            file_path,
            usecols=[label_column],
            low_memory=False,
        )

        # Normalize label values.
        labels = (
            df[label_column]
            .astype(str)
            .str.strip()
        )

        counts = labels.value_counts()

        file_total = len(df)
        total_rows += file_total

        print()
        print(
            f"Total rows: {file_total:,}"
        )

        print()
        print("Labels:")

        for label, count in counts.items():

            percentage = (
                count / file_total
            ) * 100

            print(
                f"  {label:<35} "
                f"{count:>10,} "
                f"({percentage:6.2f}%)"
            )

    print()
    print("=" * 70)
    print("TOTAL")
    print("=" * 70)

    print(
        f"Rows across all files: "
        f"{total_rows:,}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
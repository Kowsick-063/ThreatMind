from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

DATA_DIR = Path("data/processed/cicids2017")
CHUNK_SIZE = 100_000


def row_hash(row) -> str:
    """
    Create a stable hash for detecting duplicate rows.
    """
    row_string = "|".join(
        "" if pd.isna(value) else str(value)
        for value in row
    )

    return hashlib.md5(
        row_string.encode("utf-8")
    ).hexdigest()


def inspect_file(file_path: Path):
    print("\n" + "=" * 90)
    print(f"FILE: {file_path.name}")
    print("=" * 90)

    total_rows = 0
    missing_values = None
    infinite_values = None
    label_counts = {}
    column_names = None

    hashes = set()
    duplicate_count = 0

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        # Remove whitespace around column names.
        chunk.columns = chunk.columns.str.strip()

        if column_names is None:
            column_names = list(chunk.columns)
            missing_values = pd.Series(
                0,
                index=column_names,
                dtype="int64",
            )

            infinite_values = pd.Series(
                0,
                index=column_names,
                dtype="int64",
            )

        total_rows += len(chunk)

        # Missing values
        missing_values += chunk.isna().sum()

        # Infinite values
        numeric_columns = chunk.select_dtypes(
            include=[np.number]
        ).columns

        if len(numeric_columns) > 0:
            infinite_mask = np.isinf(
                chunk[numeric_columns].to_numpy()
            )

            infinite_counts = pd.Series(
                infinite_mask.sum(axis=0),
                index=numeric_columns,
            )

            infinite_values[infinite_counts.index] += (
                infinite_counts
            )

        # Labels
        if "Label" in chunk.columns:
            labels = chunk["Label"].astype(str).str.strip()

            for label, count in labels.value_counts().items():
                label_counts[label] = (
                    label_counts.get(label, 0) + int(count)
                )

        # Duplicate rows
        for row in chunk.itertuples(
            index=False,
            name=None,
        ):
            current_hash = row_hash(row)

            if current_hash in hashes:
                duplicate_count += 1
            else:
                hashes.add(current_hash)

    print(f"\nRows: {total_rows}")
    print(f"Columns: {len(column_names)}")

    print("\nColumns:")
    for index, column in enumerate(column_names, start=1):
        print(f"{index:02d}. {column}")

    print("\nLabels:")

    if label_counts:
        for label, count in sorted(
            label_counts.items(),
            key=lambda item: item[1],
            reverse=True,
        ):
            percentage = (
                count / total_rows
            ) * 100

            print(
                f"{label:<40}"
                f"{count:>10,}"
                f"  ({percentage:>7.3f}%)"
            )
    else:
        print("Label column not found.")

    print("\nMissing values:")

    missing_found = False

    for column, count in missing_values.items():
        if count > 0:
            missing_found = True
            print(
                f"{column:<40}"
                f"{count:>10,}"
            )

    if not missing_found:
        print("No missing values.")

    print("\nInfinite values:")

    infinite_found = False

    for column, count in infinite_values.items():
        if count > 0:
            infinite_found = True
            print(
                f"{column:<40}"
                f"{count:>10,}"
            )

    if not infinite_found:
        print("No infinite values.")

    print(f"\nDuplicate rows: {duplicate_count:,}")

    return {
        "file": file_path.name,
        "rows": total_rows,
        "columns": len(column_names),
        "labels": label_counts,
        "missing": missing_values,
        "infinite": infinite_values,
        "duplicates": duplicate_count,
        "column_names": column_names,
    }


def inspect_dataset():
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {DATA_DIR}"
        )

    csv_files = sorted(DATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {DATA_DIR}"
        )

    print("=" * 90)
    print("THREATMIND - CIC-IDS2017 DATASET INSPECTION")
    print("=" * 90)

    print(f"\nDataset directory: {DATA_DIR}")
    print(f"CSV files found: {len(csv_files)}")

    results = []

    for file_path in csv_files:
        result = inspect_file(file_path)
        results.append(result)

    print("\n" + "=" * 90)
    print("DATASET SUMMARY")
    print("=" * 90)

    total_rows = sum(
        result["rows"]
        for result in results
    )

    print(f"Files: {len(results)}")
    print(f"Total rows: {total_rows:,}")

    combined_labels = {}

    for result in results:
        for label, count in result["labels"].items():
            combined_labels[label] = (
                combined_labels.get(label, 0) + count
            )

    print("\nCombined labels:")

    for label, count in sorted(
        combined_labels.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        percentage = (
            count / total_rows
        ) * 100

        print(
            f"{label:<40}"
            f"{count:>10,}"
            f"  ({percentage:>7.3f}%)"
        )


if __name__ == "__main__":
    inspect_dataset()
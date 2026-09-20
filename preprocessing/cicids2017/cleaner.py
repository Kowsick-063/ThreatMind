from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw/cicids2017")
PROCESSED_DIR = Path("data/processed/cicids2017")

CHUNK_SIZE = 100_000


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    # Clean column names.
    chunk.columns = (
        chunk.columns
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # Clean labels.
    if "Label" in chunk.columns:
        chunk["Label"] = (
            chunk["Label"]
            .astype(str)
            .str.strip()
        )

    # Convert all feature columns to numeric.
    feature_columns = [
        column
        for column in chunk.columns
        if column != "Label"
    ]

    for column in feature_columns:
        chunk[column] = pd.to_numeric(
            chunk[column],
            errors="coerce",
        )

    # Replace infinite values with NaN.
    chunk.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    # Remove completely empty rows.
    chunk.dropna(
        how="all",
        inplace=True,
    )

    # Remove rows containing invalid feature values.
    chunk.dropna(
        subset=feature_columns,
        inplace=True,
    )

    # Remove duplicate flows inside this chunk.
    chunk.drop_duplicates(
        inplace=True,
    )

    return chunk


def clean_file(
    input_path: Path,
    output_path: Path,
) -> dict:

    print("\n" + "=" * 80)
    print(f"Cleaning: {input_path.name}")
    print("=" * 80)

    total_input = 0
    total_output = 0
    label_counts = {}

    first_chunk = True

    if output_path.exists():
        output_path.unlink()

    for chunk in pd.read_csv(
        input_path,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        input_rows = len(chunk)
        total_input += input_rows

        cleaned = clean_chunk(chunk)

        total_output += len(cleaned)

        if "Label" in cleaned.columns:
            counts = cleaned["Label"].value_counts()

            for label, count in counts.items():
                label_counts[label] = (
                    label_counts.get(label, 0)
                    + int(count)
                )

        cleaned.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )

        first_chunk = False

    removed = total_input - total_output

    print(f"Input rows:   {total_input:,}")
    print(f"Output rows:  {total_output:,}")
    print(f"Removed rows: {removed:,}")

    print("\nLabels after cleaning:")

    for label, count in sorted(
        label_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(
            f"{label:<40}"
            f"{count:>12,}"
        )

    return {
        "file": input_path.name,
        "input_rows": total_input,
        "output_rows": total_output,
        "removed_rows": removed,
        "labels": label_counts,
    }


def clean_dataset():

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset directory not found: {RAW_DIR}"
        )

    csv_files = sorted(
        RAW_DIR.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {RAW_DIR}"
        )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("THREATMIND - CIC-IDS2017 CLEANING")
    print("=" * 80)

    results = []

    for input_path in csv_files:

        output_path = (
            PROCESSED_DIR
            / input_path.name
        )

        result = clean_file(
            input_path,
            output_path,
        )

        results.append(result)

    print("\n" + "=" * 80)
    print("CLEANING SUMMARY")
    print("=" * 80)

    total_input = sum(
        result["input_rows"]
        for result in results
    )

    total_output = sum(
        result["output_rows"]
        for result in results
    )

    print(f"Files processed: {len(results)}")
    print(f"Input rows:      {total_input:,}")
    print(f"Output rows:     {total_output:,}")
    print(f"Removed rows:    {total_input - total_output:,}")

    print(f"\nCleaned files:")
    print(PROCESSED_DIR)


if __name__ == "__main__":
    clean_dataset()
from pathlib import Path
import pandas as pd


INPUT_DIR = Path("data/processed/cicids2017")
OUTPUT_FILE = INPUT_DIR / "cicids2017_combined.csv"

CHUNK_SIZE = 100_000


def build_combined_dataset():
    files = sorted(INPUT_DIR.glob("*.csv"))

    # Do not accidentally read the output file itself
    files = [
        file for file in files
        if file.name != OUTPUT_FILE.name
    ]

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {INPUT_DIR}"
        )

    print("=" * 80)
    print("THREATMIND - BUILDING COMBINED CIC-IDS2017 DATASET")
    print("=" * 80)

    print(f"Input files: {len(files)}")
    print(f"Output: {OUTPUT_FILE}")

    first_write = True
    total_rows = 0
    label_counts = {}

    for file in files:
        print(f"\nProcessing: {file.name}")

        file_rows = 0

        for chunk in pd.read_csv(
            file,
            chunksize=CHUNK_SIZE,
        ):
            chunk.columns = chunk.columns.str.strip()

            # Remove rows with missing labels
            chunk = chunk.dropna(subset=["Label"])

            # Update label statistics
            counts = chunk["Label"].value_counts()

            for label, count in counts.items():
                label_counts[label] = (
                    label_counts.get(label, 0) + int(count)
                )

            chunk.to_csv(
                OUTPUT_FILE,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )

            first_write = False

            rows = len(chunk)
            file_rows += rows
            total_rows += rows

        print(f"Rows added: {file_rows:,}")

    print("\n" + "=" * 80)
    print("COMBINATION COMPLETE")
    print("=" * 80)

    print(f"Total rows: {total_rows:,}")

    print("\nLabel distribution:")

    for label, count in sorted(
        label_counts.items(),
        key=lambda item: item[1],
        reverse=True,
    ):
        percentage = (count / total_rows) * 100

        print(
            f"{label:<30}"
            f"{count:>12,}"
            f" ({percentage:>7.3f}%)"
        )

    print("\nOutput file:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    build_combined_dataset()
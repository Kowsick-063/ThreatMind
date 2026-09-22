from datetime import datetime, timezone

import pandas as pd


STATE_FILE = (
    "data/processed/"
    "threatmind_friday_temporal_states.csv"
)


def main():
    print("=" * 80)
    print("THREATMIND TEMPORAL STATE TIME INSPECTION")
    print("=" * 80)

    df = pd.read_csv(STATE_FILE)

    if "timestamp" not in df.columns:
        raise ValueError(
            "timestamp column not found"
        )

    timestamps = df["timestamp"]

    first_timestamp = float(
        timestamps.iloc[0]
    )

    last_timestamp = float(
        timestamps.iloc[-1]
    )

    print()
    print(f"States: {len(df):,}")
    print()

    print("FIRST STATE")
    print("-" * 80)

    print(
        f"Unix timestamp : "
        f"{first_timestamp}"
    )

    print(
        "UTC            : "
        f"{datetime.fromtimestamp(first_timestamp, timezone.utc)}"
    )

    print()
    print("LAST STATE")
    print("-" * 80)

    print(
        f"Unix timestamp : "
        f"{last_timestamp}"
    )

    print(
        "UTC            : "
        f"{datetime.fromtimestamp(last_timestamp, timezone.utc)}"
    )

    print()
    print("SAMPLE STATES")
    print("-" * 80)

    sample_indices = [
        0,
        1,
        2,
        100,
        500,
        1000,
        2000,
        3000,
        4000,
        5000,
        len(df) - 1,
    ]

    sample_indices = sorted(
        set(
            index
            for index in sample_indices
            if 0 <= index < len(df)
        )
    )

    for index in sample_indices:
        timestamp = float(
            df.iloc[index]["timestamp"]
        )

        utc_time = datetime.fromtimestamp(
            timestamp,
            timezone.utc,
        )

        print(
            f"State {index:>5}: "
            f"{timestamp:.1f} "
            f"→ {utc_time}"
        )

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
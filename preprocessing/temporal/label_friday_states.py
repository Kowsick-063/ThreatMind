from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/"
    "threatmind_friday_temporal_states.csv"
)

OUTPUT_FILE = Path(
    "data/processed/"
    "threatmind_friday_labeled_states.csv"
)


# CIC-IDS2017 was captured at the UNB testbed in
# Atlantic Daylight Time (UTC-3) during July 2017.
LOCAL_OFFSET = timedelta(hours=-3)


# Friday, July 7, 2017.
CAPTURE_DATE = "2017-07-07"


# ------------------------------------------------------------------
# Official CIC-IDS2017 Friday attack intervals.
#
# Times below are the documented CIC/UNB local capture times.
# ------------------------------------------------------------------

BOTNET_INTERVALS = [
    ("10:02", "11:02"),
]


PORTSCAN_INTERVALS = [
    # Firewall rule ON
    ("13:55", "13:57"),
    ("13:58", "14:00"),
    ("14:01", "14:04"),
    ("14:05", "14:07"),
    ("14:08", "14:10"),
    ("14:11", "14:13"),
    ("14:14", "14:16"),
    ("14:17", "14:19"),
    ("14:20", "14:21"),
    ("14:22", "14:24"),
    ("14:33", "14:33"),
    ("14:35", "14:35"),

    # Firewall rule OFF / scan variants
    ("14:51", "14:53"),
    ("14:54", "14:56"),
    ("14:57", "14:59"),
    ("15:00", "15:02"),
    ("15:03", "15:05"),
    ("15:06", "15:07"),
    ("15:08", "15:10"),
    ("15:11", "15:12"),
    ("15:13", "15:15"),
    ("15:16", "15:18"),
    ("15:19", "15:21"),
    ("15:22", "15:24"),
    ("15:25", "15:25"),
    ("15:26", "15:27"),
    ("15:28", "15:29"),
]


DDOS_INTERVALS = [
    ("15:56", "16:16"),
]


def local_to_utc(
    time_string: str,
) -> datetime:
    """
    Convert CIC local capture time (ADT, UTC-3)
    to timezone-aware UTC datetime.
    """

    local_datetime = datetime.strptime(
        f"{CAPTURE_DATE} {time_string}",
        "%Y-%m-%d %H:%M",
    )

    local_datetime = local_datetime.replace(
        tzinfo=timezone(LOCAL_OFFSET)
    )

    return local_datetime.astimezone(
        timezone.utc
    )


def build_intervals(
    intervals: list[tuple[str, str]],
):
    """
    Convert a list of local-time intervals into
    UTC datetime intervals.
    """

    converted = []

    for start, end in intervals:

        start_utc = local_to_utc(start)

        end_utc = local_to_utc(end)

        converted.append(
            (
                start_utc.timestamp(),
                end_utc.timestamp(),
            )
        )

    return converted


def timestamp_in_intervals(
    timestamp: float,
    intervals,
) -> bool:

    for start, end in intervals:

        if start <= timestamp <= end:
            return True

    return False


def classify_timestamp(
    timestamp: float,
    botnet_intervals,
    portscan_intervals,
    ddos_intervals,
) -> str:

    # Check the most specific attack intervals first.

    if timestamp_in_intervals(
        timestamp,
        ddos_intervals,
    ):
        return "DDoS"

    if timestamp_in_intervals(
        timestamp,
        portscan_intervals,
    ):
        return "PortScan"

    if timestamp_in_intervals(
        timestamp,
        botnet_intervals,
    ):
        return "Bot"

    return "BENIGN"


def main():

    print("=" * 80)
    print("THREATMIND FRIDAY TEMPORAL LABEL BUILDER")
    print("=" * 80)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print()
    print(f"Input : {INPUT_FILE}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    # --------------------------------------------------------------
    # Load existing temporal states.
    # --------------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    if "timestamp" not in df.columns:
        raise ValueError(
            "Input dataset does not contain "
            "'timestamp' column."
        )

    print(
        f"States loaded: {len(df):,}"
    )

    # --------------------------------------------------------------
    # Convert documented attack intervals to UTC.
    # --------------------------------------------------------------

    botnet_intervals = build_intervals(
        BOTNET_INTERVALS
    )

    portscan_intervals = build_intervals(
        PORTSCAN_INTERVALS
    )

    ddos_intervals = build_intervals(
        DDOS_INTERVALS
    )

    # --------------------------------------------------------------
    # Display intervals for verification.
    # --------------------------------------------------------------

    print()
    print("Attack intervals in UTC:")
    print("-" * 80)

    print("BOT:")
    for start, end in botnet_intervals:
        print(
            f"  {datetime.fromtimestamp(start, timezone.utc)}"
            f" -> "
            f"{datetime.fromtimestamp(end, timezone.utc)}"
        )

    print()
    print("PORTSCAN:")
    for start, end in portscan_intervals:
        print(
            f"  {datetime.fromtimestamp(start, timezone.utc)}"
            f" -> "
            f"{datetime.fromtimestamp(end, timezone.utc)}"
        )

    print()
    print("DDOS:")
    for start, end in ddos_intervals:
        print(
            f"  {datetime.fromtimestamp(start, timezone.utc)}"
            f" -> "
            f"{datetime.fromtimestamp(end, timezone.utc)}"
        )

    # --------------------------------------------------------------
    # Classify every temporal state.
    # --------------------------------------------------------------

    df["attack_label"] = df["timestamp"].apply(
        lambda timestamp: classify_timestamp(
            float(timestamp),
            botnet_intervals,
            portscan_intervals,
            ddos_intervals,
        )
    )

    df["is_attack"] = (
        df["attack_label"] != "BENIGN"
    ).astype(int)

    # --------------------------------------------------------------
    # Add human-readable UTC time.
    # This is useful for debugging and visualization.
    # --------------------------------------------------------------

    df["datetime_utc"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True,
    )

    # --------------------------------------------------------------
    # Save.
    # --------------------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    # --------------------------------------------------------------
    # Validation / summary.
    # --------------------------------------------------------------

    print()
    print("=" * 80)
    print("LABEL DISTRIBUTION")
    print("=" * 80)

    counts = (
        df["attack_label"]
        .value_counts()
    )

    total = len(df)

    for label, count in counts.items():

        percentage = (
            count / total
        ) * 100

        print(
            f"{label:<15}"
            f"{count:>8,}"
            f" "
            f"({percentage:6.2f}%)"
        )

    print()
    print("=" * 80)
    print("ATTACK TYPE SUMMARY")
    print("=" * 80)

    attack_df = df[
        df["is_attack"] == 1
    ]

    print(
        f"Attack states: "
        f"{len(attack_df):,}"
    )

    print(
        f"Benign states: "
        f"{len(df) - len(attack_df):,}"
    )

    print()

    # --------------------------------------------------------------
    # Print first and last state for each attack.
    # --------------------------------------------------------------

    print("=" * 80)
    print("ATTACK STATE RANGES")
    print("=" * 80)

    for label in [
        "Bot",
        "PortScan",
        "DDoS",
    ]:

        attack_states = df[
            df["attack_label"] == label
        ]

        if attack_states.empty:
            print()
            print(
                f"{label}: NO STATES FOUND"
            )
            continue

        first = attack_states.iloc[0]
        last = attack_states.iloc[-1]

        first_time = datetime.fromtimestamp(
            float(first["timestamp"]),
            timezone.utc,
        )

        last_time = datetime.fromtimestamp(
            float(last["timestamp"]),
            timezone.utc,
        )

        print()
        print(label)

        print(
            f"  First state: "
            f"{first_time}"
        )

        print(
            f"  Last state : "
            f"{last_time}"
        )

        print(
            f"  States     : "
            f"{len(attack_states):,}"
        )

    print()
    print("=" * 80)
    print("COMPLETE")
    print("=" * 80)

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
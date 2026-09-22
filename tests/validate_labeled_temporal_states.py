from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/"
    "threatmind_friday_labeled_states.csv"
)


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
]


LABEL_ORDER = [
    "BENIGN",
    "Bot",
    "PortScan",
    "DDoS",
]


def main():
    print("=" * 100)
    print("THREATMIND LABELED TEMPORAL STATE VALIDATION")
    print("=" * 100)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    print()
    print(f"Dataset: {INPUT_FILE}")
    print(f"Rows:    {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    # --------------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------------

    print()
    print("=" * 100)
    print("1. BASIC VALIDATION")
    print("=" * 100)

    required_columns = [
        "timestamp",
        "attack_label",
        "is_attack",
    ] + FEATURES

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print()
        print("Missing columns:")

        for column in missing_columns:
            print(f"  - {column}")

        raise ValueError(
            "Required columns are missing."
        )

    print()
    print("Required columns: OK")

    missing_values = (
        df[required_columns]
        .isna()
        .sum()
        .sum()
    )

    print(
        f"Missing values:    {missing_values:,}"
    )

    duplicate_rows = df.duplicated().sum()

    print(
        f"Duplicate rows:    {duplicate_rows:,}"
    )

    invalid_attack_values = (
        ~df["is_attack"].isin([0, 1])
    ).sum()

    print(
        f"Invalid is_attack: "
        f"{invalid_attack_values:,}"
    )

    # --------------------------------------------------------------
    # Label consistency
    # --------------------------------------------------------------

    print()
    print("=" * 100)
    print("2. LABEL CONSISTENCY")
    print("=" * 100)

    expected_is_attack = (
        df["attack_label"] != "BENIGN"
    ).astype(int)

    consistency_errors = (
        df["is_attack"]
        != expected_is_attack
    ).sum()

    print(
        "is_attack consistency errors: "
        f"{consistency_errors:,}"
    )

    print()
    print("Label counts:")

    label_counts = (
        df["attack_label"]
        .value_counts()
        .reindex(
            LABEL_ORDER,
            fill_value=0,
        )
    )

    for label, count in label_counts.items():

        percentage = (
            count / len(df)
        ) * 100

        print(
            f"  {label:<12} "
            f"{count:>8,} "
            f"({percentage:6.2f}%)"
        )

    # --------------------------------------------------------------
    # Feature missing / invalid values
    # --------------------------------------------------------------

    print()
    print("=" * 100)
    print("3. FEATURE VALIDATION")
    print("=" * 100)

    print()

    for feature in FEATURES:

        missing = df[feature].isna().sum()

        negative = (
            df[feature] < 0
        ).sum()

        print(
            f"{feature:<28} "
            f"missing={missing:>5,} "
            f"negative={negative:>5,}"
        )

    # --------------------------------------------------------------
    # Feature statistics by attack type
    # --------------------------------------------------------------

    print()
    print("=" * 100)
    print("4. FEATURE STATISTICS BY ATTACK TYPE")
    print("=" * 100)

    grouped = (
        df.groupby("attack_label")[FEATURES]
        .agg(["mean", "median"])
    )

    for feature in FEATURES:

        print()
        print("-" * 100)
        print(feature)
        print("-" * 100)

        for label in LABEL_ORDER:

            if label not in grouped.index:
                continue

            mean_value = grouped.loc[
                label,
                (feature, "mean"),
            ]

            median_value = grouped.loc[
                label,
                (feature, "median"),
            ]

            print(
                f"  {label:<12} "
                f"mean={mean_value:>15.4f} "
                f"median={median_value:>15.4f}"
            )

    # --------------------------------------------------------------
    # Attack vs benign relative changes
    # --------------------------------------------------------------

    print()
    print("=" * 100)
    print("5. ATTACK VS BENIGN RELATIVE CHANGE")
    print("=" * 100)

    benign = df[
        df["attack_label"] == "BENIGN"
    ]

    for label in [
        "Bot",
        "PortScan",
        "DDoS",
    ]:

        attack = df[
            df["attack_label"] == label
        ]

        print()
        print("-" * 100)
        print(label)
        print("-" * 100)

        for feature in FEATURES:

            benign_mean = benign[
                feature
            ].mean()

            attack_mean = attack[
                feature
            ].mean()

            if benign_mean == 0:
                relative_change = float("nan")
            else:
                relative_change = (
                    (
                        attack_mean
                        - benign_mean
                    )
                    / benign_mean
                ) * 100

            print(
                f"  {feature:<28} "
                f"benign={benign_mean:>14.3f} "
                f"attack={attack_mean:>14.3f} "
                f"change={relative_change:>9.2f}%"
            )

    # --------------------------------------------------------------
    # Temporal transitions
    # --------------------------------------------------------------

    print()
    print("=" * 100)
    print("6. TEMPORAL LABEL TRANSITIONS")
    print("=" * 100)

    labels = (
        df["attack_label"]
        .tolist()
    )

    transitions = []

    previous = labels[0]

    for current in labels[1:]:

        if current != previous:

            transitions.append(
                (
                    previous,
                    current,
                )
            )

        previous = current

    print()
    print(
        f"Number of label transitions: "
        f"{len(transitions):,}"
    )

    transition_counts = {}

    for previous, current in transitions:

        key = (
            previous,
            current,
        )

        transition_counts[key] = (
            transition_counts.get(
                key,
                0,
            )
            + 1
        )

    print()

    for (previous, current), count in sorted(
        transition_counts.items()
    ):
        print(
            f"  {previous:<12} "
            f"-> {current:<12} "
            f"{count:>5,}"
        )

    # --------------------------------------------------------------
    # Attack runs
    # --------------------------------------------------------------

    print()
    print("=" * 100)
    print("7. ATTACK RUN ANALYSIS")
    print("=" * 100)

    current_label = labels[0]
    run_start = 0

    runs = []

    for index in range(
        1,
        len(labels),
    ):

        if labels[index] != current_label:

            runs.append(
                (
                    current_label,
                    run_start,
                    index - 1,
                )
            )

            current_label = labels[index]
            run_start = index

    runs.append(
        (
            current_label,
            run_start,
            len(labels) - 1,
        )
    )

    for label in [
        "Bot",
        "PortScan",
        "DDoS",
    ]:

        label_runs = [
            run
            for run in runs
            if run[0] == label
        ]

        print()
        print(
            f"{label}: "
            f"{len(label_runs)} contiguous runs"
        )

        for run_label, start, end in label_runs:

            state_count = (
                end - start + 1
            )

            start_time = df.iloc[
                start
            ]["datetime_utc"]

            end_time = df.iloc[
                end
            ]["datetime_utc"]

            print(
                f"  {start_time} "
                f"-> {end_time} "
                f"({state_count:,} states)"
            )

    # --------------------------------------------------------------
    # Boundary inspection
    # --------------------------------------------------------------

    print()
    print("=" * 100)
    print("8. ATTACK BOUNDARY INSPECTION")
    print("=" * 100)

    transition_indices = []

    previous_label = labels[0]

    for index in range(
        1,
        len(labels),
    ):

        current_label = labels[index]

        if current_label != previous_label:

            transition_indices.append(index)

        previous_label = current_label

    for index in transition_indices:

        start = max(
            0,
            index - 2,
        )

        end = min(
            len(df),
            index + 3,
        )

        print()
        print(
            f"Transition around state "
            f"{index}:"
        )

        print(
            df.iloc[
                start:end
            ][
                [
                    "timestamp",
                    "datetime_utc",
                    "flow_count",
                    "packet_count",
                    "byte_count",
                    "unique_sources",
                    "unique_destinations",
                    "unique_ports",
                    "packets_per_second",
                    "bytes_per_second",
                    "attack_label",
                ]
            ].to_string(
                index=True
            )
        )

    # --------------------------------------------------------------
    # Final verdict
    # --------------------------------------------------------------

    print()
    print("=" * 100)
    print("9. VALIDATION SUMMARY")
    print("=" * 100)

    problems = []

    if missing_values > 0:
        problems.append(
            "missing values"
        )

    if duplicate_rows > 0:
        problems.append(
            "duplicate rows"
        )

    if invalid_attack_values > 0:
        problems.append(
            "invalid attack flags"
        )

    if consistency_errors > 0:
        problems.append(
            "label/is_attack inconsistencies"
        )

    if problems:

        print()
        print(
            "STATUS: REVIEW REQUIRED"
        )

        print()
        print("Problems:")

        for problem in problems:
            print(
                f"  - {problem}"
            )

    else:

        print()
        print(
            "STATUS: STRUCTURALLY VALID"
        )

        print()
        print(
            "The temporal dataset is "
            "structurally ready for "
            "behavioral/model analysis."
        )

    print()
    print("=" * 100)


if __name__ == "__main__":
    main()
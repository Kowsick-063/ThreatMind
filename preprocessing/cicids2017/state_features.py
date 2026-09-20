from pathlib import Path

import pandas as pd


INPUT_FILE = Path(
    "data/processed/cicids2017/cicids2017_combined.csv"
)

OUTPUT_FILE = Path(
    "data/processed/cicids2017/threatmind_state_features.csv"
)


STATE_FEATURES = [
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
]


def main():

    print("=" * 80)
    print("THREATMIND - STATE FEATURE EXTRACTION")
    print("=" * 80)

    print(f"\nInput: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    df.columns = df.columns.str.strip()

    print(f"Original shape: {df.shape}")

    missing_features = [
        feature
        for feature in STATE_FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features: {missing_features}"
        )

    state_df = df[
        STATE_FEATURES + ["Label"]
    ].copy()

    print(
        f"\nSelected state features: "
        f"{len(STATE_FEATURES)}"
    )

    print("\nFeatures:")

    for index, feature in enumerate(
        STATE_FEATURES,
        start=1,
    ):
        print(f"{index:02d}. {feature}")

    print("\nState feature matrix:")
    print(state_df.shape)

    print("\nLabel distribution:")
    print(
        state_df["Label"]
        .value_counts()
        .to_string()
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nSaved to:\n{OUTPUT_FILE}"
    )

    print("\n" + "=" * 80)
    print("STATE FEATURE EXTRACTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
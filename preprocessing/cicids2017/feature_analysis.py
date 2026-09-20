from pathlib import Path

import pandas as pd


DATASET = Path(
    "data/processed/cicids2017/cicids2017_combined.csv"
)


def main():
    print("=" * 80)
    print("THREATMIND - CIC-IDS2017 FEATURE ANALYSIS")
    print("=" * 80)

    df = pd.read_csv(DATASET)

    df.columns = df.columns.str.strip()

    feature_columns = [
        column for column in df.columns
        if column != "Label"
    ]

    print(f"\nTotal samples : {len(df):,}")
    print(f"Total features: {len(feature_columns)}")

    print("\n" + "=" * 80)
    print("FEATURE STATISTICS")
    print("=" * 80)

    stats = []

    for column in feature_columns:
        series = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        stats.append({
            "feature": column,
            "unique_values": series.nunique(),
            "mean": series.mean(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "zero_percentage": (
                (series == 0).mean() * 100
            ),
        })

    stats_df = pd.DataFrame(stats)

    print(
        stats_df.to_string(
            index=False
        )
    )

    output = Path(
        "data/processed/cicids2017/"
        "feature_statistics.csv"
    )

    stats_df.to_csv(
        output,
        index=False
    )

    print(
        f"\nSaved statistics to:\n{output}"
    )

    # ------------------------------------------------------------
    # Candidate ThreatMind state features
    # ------------------------------------------------------------

    candidate_features = [
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

    print("\n" + "=" * 80)
    print("THREATMIND CANDIDATE STATE FEATURES")
    print("=" * 80)

    for feature in candidate_features:
        if feature in df.columns:
            print(f"✓ {feature}")
        else:
            print(f"✗ {feature}")

    print("\n" + "=" * 80)
    print("FEATURE ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
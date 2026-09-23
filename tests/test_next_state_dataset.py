import unittest
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_FILE = Path(
    "data/processed/threatmind_next_state_dataset.csv"
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
    "mean_ttl",
    "std_ttl",
    "mean_tcp_window",
    "std_tcp_window",
    "mean_payload_size",
    "std_payload_size",
    "fragment_count",
    "mean_iat",
    "std_iat",
    "max_iat",
]


class NextStateDatasetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(DATASET_FILE)

    def test_no_duplicate_timestamps(self):
        self.assertFalse(self.data["timestamp"].duplicated().any())

    def test_current_timestamp_before_target(self):
        self.assertTrue(
            (
                self.data["timestamp"]
                < self.data["target_timestamp"]
            ).all()
        )

    def test_target_is_exactly_five_seconds_later(self):
        differences = (
            self.data["target_timestamp"]
            - self.data["timestamp"]
        )
        self.assertTrue((differences == 5).all())

    def test_no_nan_values_in_numerical_features(self):
        numeric_features = self.data[FEATURES]
        self.assertFalse(numeric_features.isna().any().any())

    def test_no_infinite_values(self):
        numeric_data = self.data.select_dtypes(include="number")
        self.assertFalse(
            np.isinf(numeric_data.to_numpy()).any()
        )

    def test_target_labels_are_valid(self):
        self.assertTrue(
            self.data["target_attack_label"].isin(
                ["BENIGN", "Bot", "PortScan", "DDoS"]
            ).all()
        )
        self.assertTrue(
            self.data["target_is_attack"].isin([0, 1]).all()
        )

    def test_dataset_is_chronologically_ordered(self):
        self.assertTrue(
            self.data["timestamp"].is_monotonic_increasing
        )

    def test_target_columns_are_not_current_features(self):
        current_columns = set(self.data.columns) - {
            "timestamp",
            "target_is_attack",
            "target_attack_label",
            "target_timestamp",
        }
        self.assertFalse(
            any(column.startswith("target_") for column in current_columns)
        )
        self.assertNotIn("attack_label", current_columns)
        self.assertNotIn("is_attack", current_columns)


if __name__ == "__main__":
    unittest.main()
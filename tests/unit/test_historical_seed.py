import math
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

import counterfactual.rollout as rollout_module
from counterfactual.interventions import build_intervention
from counterfactual.rollout import (
    get_historical_sequence,
    simulate_counterfactual,
)
from models.world_model.create_sequences import FEATURES
from models.world_model.rollout import load_model


DATASET = Path("data/processed/threatmind_friday_labeled_states.csv")
SCALER = Path("models/world_model/world_model_scaler.joblib")


def _timestamp_with_history(data: pd.DataFrame, contiguous: bool) -> float:
    timestamps = data["timestamp"].astype(float).to_numpy()
    for index in range(59, len(timestamps)):
        is_contiguous = bool(np.all(np.diff(timestamps[index - 59:index + 1]) == 5.0))
        if is_contiguous == contiguous:
            return float(timestamps[index])
    raise AssertionError("The fixture dataset has no matching timestamp.")


class HistoricalSeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv(DATASET)
        cls.valid_timestamp = _timestamp_with_history(cls.data, True)
        cls.gap_timestamp = _timestamp_with_history(cls.data, False)

    def test_valid_history_shape_order_and_endpoint(self):
        sequence = get_historical_sequence(self.valid_timestamp)
        expected = self.data[self.data["timestamp"] <= self.valid_timestamp].tail(60)

        self.assertEqual(sequence.shape, (60, 26))
        np.testing.assert_allclose(
            sequence,
            expected[FEATURES].to_numpy(dtype=np.float32),
        )
        self.assertEqual(float(expected["timestamp"].iloc[-1]), self.valid_timestamp)

    def test_gap_rejection_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "Contiguous historical sequence unavailable"):
            get_historical_sequence(self.gap_timestamp)

    def test_missing_timestamp_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "current_timestamp is required"):
            get_historical_sequence(None)

    def test_history_passes_existing_scaler_and_model(self):
        sequence = get_historical_sequence(self.valid_timestamp)
        scaler = joblib.load(SCALER)
        model = load_model()

        prediction = rollout_module._predict_next(model, sequence, scaler)

        self.assertEqual(prediction.shape, (26,))
        self.assertTrue(np.isfinite(prediction).all())

    def test_counterfactual_uses_real_history_for_both_legs(self):
        state = self.data.loc[
            self.data["timestamp"] == self.valid_timestamp,
            FEATURES,
        ].iloc[0].to_dict()
        history = get_historical_sequence(self.valid_timestamp)
        captured = {}
        original = rollout_module.generate_baseline_trajectory

        def capture(sequence, *args, **kwargs):
            captured["sequence"] = np.asarray(sequence).copy()
            return original(sequence, *args, **kwargs)

        with patch.object(
            rollout_module,
            "generate_baseline_trajectory",
            side_effect=capture,
        ):
            simulation = simulate_counterfactual(
                initial_state=state,
                intervention=build_intervention("NO_ACTION"),
                horizon=12,
                current_timestamp=self.valid_timestamp,
            )

        self.assertEqual(len(simulation["baseline"]), 12)
        self.assertEqual(len(simulation["counterfactual"]), 12)
        np.testing.assert_allclose(captured["sequence"], history)
        for trajectory in (simulation["baseline"], simulation["counterfactual"]):
            for point in trajectory:
                self.assertTrue(math.isfinite(point["attack_probability"]))
                self.assertTrue(math.isfinite(point["risk"]))


if __name__ == "__main__":
    unittest.main()
import math
import unittest

import numpy as np
import pandas as pd

from evaluation.evaluate_real_history_forecast import (
    HORIZON,
    FEATURES,
    state_metrics,
    verify_history,
)


class RealHistoryForecastingTest(unittest.TestCase):
    def test_state_metrics_include_finite_lstm_and_persistence_values(self):
        predicted = np.ones((2, HORIZON, len(FEATURES)), dtype=np.float32)
        actual = np.zeros_like(predicted)
        predicted_scaled = predicted.copy()
        actual_scaled = actual.copy()
        baseline = np.full_like(predicted, 2.0)

        metrics, features = state_metrics(
            predicted,
            actual,
            predicted_scaled,
            actual_scaled,
            baseline,
        )

        self.assertEqual(len(metrics), HORIZON)
        self.assertEqual(len(features), HORIZON * len(FEATURES))
        self.assertTrue(np.isfinite(metrics.select_dtypes("number").to_numpy()).all())

    def test_persistence_baseline_is_current_state_for_every_horizon(self):
        current = np.arange(len(FEATURES), dtype=np.float32)
        baseline = np.repeat(current[None, None, :], HORIZON, axis=1)
        self.assertEqual(baseline.shape, (1, HORIZON, len(FEATURES)))
        for horizon in range(HORIZON):
            np.testing.assert_array_equal(baseline[0, horizon], current)

    def test_history_verification_rejects_missing_temporal_state(self):
        states = pd.DataFrame({"timestamp": [0.0, 5.0, 15.0]})
        test = (
            np.zeros((1, 60, len(FEATURES)), dtype=np.float32),
            np.zeros((1, len(FEATURES)), dtype=np.float32),
            np.asarray([15.0]),
            np.asarray([20.0]),
        )
        with self.assertRaisesRegex(ValueError, "missing timestamp"):
            verify_history(test, states, [0])

    def test_horizon_constant_and_feature_count_are_stable(self):
        self.assertEqual(HORIZON, 12)
        self.assertEqual(len(FEATURES), 26)


if __name__ == "__main__":
    unittest.main()
"""Unit test suite for Step 19 — Attack Trajectory + Multi-Step Forecast Fusion.

Verifies:
  1. Valid historical timestamp produces H1–H12 trajectory.
  2. Every horizon contains calibrated attack probability in [0, 1].
  3. H1–H12 use the correct horizon-specific calibrators.
  4. Category output respects the estimator's supported classes.
  5. DDoS does not get falsely converted into a supported category.
  6. Absolute and delta forecasts retain their source labels.
  7. MITRE confidence is independent of attack probability.
  8. Missing historical sequence produces an explicit failure (no silent fallback).
  9. API endpoint POST /api/forecast/trajectory functions correctly.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
import numpy as np

from backend.api.forecast import ForecastTrajectoryRequest, forecast_trajectory
from counterfactual.rollout import _load_annotation_models, get_historical_sequence
from forecasting.trajectory import generate_attack_trajectory
from forecasting.trajectory_fusion import (
    SUPPORTED_ESTIMATOR_CLASSES,
    detect_ddos_pattern,
    fuse_trajectory_step,
)
from models.world_model.create_sequences import FEATURES


VALID_TIMESTAMP = 1499433815.0
INVALID_TIMESTAMP = 999999999.0


class AttackTrajectoryVerificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.probability_model, cls.category_estimator, cls.calibrators = _load_annotation_models()

    def test_1_valid_historical_timestamp_produces_h1_h12(self):
        """1. Valid historical timestamp produces H1-H12."""
        result = generate_attack_trajectory(current_timestamp=VALID_TIMESTAMP)

        self.assertIn("trajectory", result)
        self.assertIn("trajectory_summary", result)
        self.assertIn("explanation", result)
        self.assertIn("uncertainties", result)
        self.assertIn("mitre_evidence", result)

        trajectory = result["trajectory"]
        self.assertEqual(len(trajectory), 12)

        for step_idx, point in enumerate(trajectory, start=1):
            self.assertEqual(point["horizon"], step_idx)
            self.assertIn("timestamp", point)
            self.assertIn("predicted_attack_probability", point)
            self.assertIn("attack_category", point)
            self.assertIn("category_confidence", point)
            self.assertIn("state_features", point)
            self.assertIn("change_features", point)
            self.assertIn("mitre_techniques", point)
            self.assertIn("uncertainties", point)
            self.assertEqual(len(point["state_features"]), len(FEATURES))

    def test_2_every_horizon_contains_calibrated_attack_probability(self):
        """2. Every horizon contains calibrated attack probability."""
        result = generate_attack_trajectory(current_timestamp=VALID_TIMESTAMP)
        trajectory = result["trajectory"]

        for point in trajectory:
            prob = point["predicted_attack_probability"]
            self.assertIsInstance(prob, float)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)

    def test_3_h1_h12_use_correct_horizon_specific_calibrators(self):
        """3. H1-H12 use the correct horizon-specific calibrators."""
        result = generate_attack_trajectory(current_timestamp=VALID_TIMESTAMP)
        trajectory = result["trajectory"]

        for step in range(1, 13):
            point = trajectory[step - 1]
            source_tag = point["sources"]["predicted_attack_probability"]
            expected_tag = f"PlattCalibrator(horizon_{step:02d})"
            self.assertIn(expected_tag, source_tag)

        # Verify using spy on calibrators
        mock_calibrators = {h: MagicMock() for h in range(1, 13)}
        for h, cal in mock_calibrators.items():
            cal.predict_proba.return_value = np.array([[0.1, 0.42]])

        with patch("forecasting.trajectory._load_annotation_models", return_value=(self.probability_model, self.category_estimator, mock_calibrators)):
            generate_attack_trajectory(current_timestamp=VALID_TIMESTAMP)

        for h in range(1, 13):
            mock_calibrators[h].predict_proba.assert_called_once()

    def test_4_category_output_respects_supported_classes(self):
        """4. Category output respects the estimator's supported classes."""
        result = generate_attack_trajectory(current_timestamp=VALID_TIMESTAMP)
        trajectory = result["trajectory"]

        for point in trajectory:
            cat = point["attack_category"]
            # If not flagged as ddos limitation, category must be in supported classes
            if "category_support_limited" not in str(point["uncertainties"]):
                self.assertIn(cat, SUPPORTED_ESTIMATOR_CLASSES)
            self.assertGreaterEqual(point["category_confidence"], 0.0)
            self.assertLessEqual(point["category_confidence"], 1.0)

    def test_5_ddos_does_not_get_falsely_converted_into_supported_category(self):
        """5. DDoS does not get falsely converted into a supported category."""
        ddos_state = {feat: 100.0 for feat in FEATURES}
        ddos_state["packets_per_second"] = 50000.0  # Volumetric flood
        ddos_state["byte_count"] = 800000000.0
        ddos_state["packet_count"] = 200000.0

        self.assertTrue(detect_ddos_pattern(ddos_state))

        mock_scaled = np.zeros(26, dtype=np.float32)
        fused = fuse_trajectory_step(
            horizon=1,
            timestamp=1499433820.0,
            raw_lstm_state_scaled=mock_scaled,
            production_state_real=ddos_state,
            delta_change_real={},
            delta_reconstructed_real=None,
            probability_model=self.probability_model,
            calibrator=self.calibrators[1],
            category_estimator=self.category_estimator,
        )

        self.assertNotIn(fused["attack_category"], SUPPORTED_ESTIMATOR_CLASSES)
        self.assertEqual(fused["attack_category"], "DDoS")
        self.assertEqual(fused["category_confidence"], 0.0)
        self.assertTrue(
            any("category_support_limited" in u for u in fused["uncertainties"])
        )

    def test_6_absolute_and_delta_forecasts_retain_source_labels(self):
        """6. Absolute and delta forecasts retain their source labels."""
        result = generate_attack_trajectory(current_timestamp=VALID_TIMESTAMP)
        trajectory = result["trajectory"]

        for point in trajectory:
            self.assertEqual(point["absolute_forecast"]["source"], "production_lstm")
            self.assertEqual(point["delta_forecast"]["source"], "delta_lstm")
            self.assertEqual(point["sources"]["absolute_forecast"], "production_lstm")
            self.assertEqual(point["sources"]["delta_forecast"], "delta_lstm")
            self.assertIn("state_features", point["absolute_forecast"])
            self.assertIn("change_features", point["delta_forecast"])

    def test_7_mitre_confidence_is_independent_of_attack_probability(self):
        """7. MITRE confidence is independent of attack probability."""
        normal_state = {feat: 50.0 for feat in FEATURES}
        normal_state["unique_ports"] = 500.0
        normal_state["packets_per_second"] = 50.0
        normal_state["mean_flow_duration"] = 0.01

        mock_scaled = np.zeros(26, dtype=np.float32)

        # Case A: Low probability model output
        mock_prob_model_low = MagicMock()
        mock_prob_model_low.predict_probability.return_value = np.array([0.01])

        # Case B: High probability model output
        mock_prob_model_high = MagicMock()
        mock_prob_model_high.predict_probability.return_value = np.array([0.99])

        fused_low = fuse_trajectory_step(
            horizon=1,
            timestamp=1499433820.0,
            raw_lstm_state_scaled=mock_scaled,
            production_state_real=normal_state,
            delta_change_real={},
            delta_reconstructed_real=None,
            probability_model=mock_prob_model_low,
            calibrator=self.calibrators[1],
            category_estimator=self.category_estimator,
        )

        fused_high = fuse_trajectory_step(
            horizon=1,
            timestamp=1499433820.0,
            raw_lstm_state_scaled=mock_scaled,
            production_state_real=normal_state,
            delta_change_real={},
            delta_reconstructed_real=None,
            probability_model=mock_prob_model_high,
            calibrator=self.calibrators[1],
            category_estimator=self.category_estimator,
        )

        # Probabilities differ significantly
        self.assertNotEqual(
            fused_low["predicted_attack_probability"],
            fused_high["predicted_attack_probability"],
        )

        # MITRE techniques and confidences remain identical because confidence is evidence-grounded
        low_techniques = {t["technique_id"]: t["confidence"] for t in fused_low["mitre_techniques"]}
        high_techniques = {t["technique_id"]: t["confidence"] for t in fused_high["mitre_techniques"]}
        self.assertEqual(low_techniques, high_techniques)

        # Technique confidence is not equal to attack probability
        for t in fused_low["mitre_techniques"]:
            self.assertNotEqual(t["confidence"], fused_low["predicted_attack_probability"])

    def test_8_missing_historical_sequence_produces_explicit_failure(self):
        """8. Missing historical sequence produces an explicit failure."""
        with self.assertRaises(ValueError) as ctx:
            generate_attack_trajectory(current_timestamp=INVALID_TIMESTAMP)
        self.assertIn("No historical state exists", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx2:
            generate_attack_trajectory(current_timestamp=None, historical_sequence=None)
        self.assertIn("Either current_timestamp or historical_sequence must be provided", str(ctx2.exception))

        # Check API endpoint returns HTTP 422 with availability: False
        with self.assertRaises(HTTPException) as http_ctx:
            forecast_trajectory(ForecastTrajectoryRequest(timestamp=INVALID_TIMESTAMP))
        self.assertEqual(http_ctx.exception.status_code, 422)
        self.assertFalse(http_ctx.exception.detail["available"])

    def test_9_api_forecast_trajectory_endpoint(self):
        """9. API endpoint POST /api/forecast/trajectory returns valid payload."""
        req = ForecastTrajectoryRequest(timestamp=VALID_TIMESTAMP)
        data = forecast_trajectory(req)

        self.assertIn("trajectory", data)
        self.assertEqual(len(data["trajectory"]), 12)
        self.assertIn("trajectory_summary", data)
        self.assertIn("peak_attack_probability", data["trajectory_summary"])
        self.assertIn("probability_trend", data["trajectory_summary"])
        self.assertIn("explanation", data)
        self.assertIn("mitre_evidence", data)
        self.assertIn("uncertainties", data)


if __name__ == "__main__":
    unittest.main()

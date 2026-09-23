"""Unit test suite for Step 21 — Real-Time SOC Integration Layer.

Verifies:
  1. Valid timestamp produces complete SOC response.
  2. Invalid timestamp handled correctly.
  3. Current state is present with 26 features.
  4. H1-H12 trajectory is preserved.
  5. Risk analysis is present.
  6. MITRE evidence is preserved.
  7. Graph availability is reported correctly.
  8. All 7 interventions are preserved.
  9. Defense decision is preserved.
  10. Explanation contains OBSERVED, PREDICTED, SIMULATED, RECOMMENDED.
  11. NO_ACTION remains valid when it is the highest-scoring option.
  12. Existing defense score remains unchanged.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi import HTTPException

from backend.api.soc import SocAnalyzeRequest, analyze_soc
from decision.defense_decision import ALL_INTERVENTION_TYPES
from models.world_model.create_sequences import FEATURES
from soc.soc_explanation import generate_soc_explanation
from soc.soc_orchestrator import run_soc_analysis
from soc.soc_response import build_soc_response


VALID_TIMESTAMP = 1499433815.0
INVALID_TIMESTAMP = 999999999.0


class SocOrchestratorTest(unittest.TestCase):
    def test_1_valid_timestamp_produces_complete_soc_response(self):
        """1. Valid timestamp produces complete SOC response."""
        result = run_soc_analysis(VALID_TIMESTAMP)

        expected_top_keys = [
            "timestamp",
            "status",
            "current_state",
            "risk",
            "forecast",
            "mitre",
            "graph",
            "counterfactuals",
            "decision",
            "explanation",
            "uncertainties",
        ]
        for key in expected_top_keys:
            self.assertIn(key, result)

        self.assertEqual(result["status"], "ANALYZED")
        self.assertEqual(result["timestamp"], VALID_TIMESTAMP)

    def test_2_invalid_timestamp_handled_correctly(self):
        """2. Invalid timestamp handled correctly (raises ValueError / HTTPException 422)."""
        with self.assertRaises(ValueError):
            run_soc_analysis(INVALID_TIMESTAMP)

        with self.assertRaises(HTTPException) as ctx:
            analyze_soc(SocAnalyzeRequest(timestamp=INVALID_TIMESTAMP))
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertFalse(ctx.exception.detail["available"])

    def test_3_current_state_is_present(self):
        """3. Current state is present with 26 features."""
        result = run_soc_analysis(VALID_TIMESTAMP)
        current_state = result["current_state"]

        self.assertEqual(len(current_state), len(FEATURES))
        for feature in FEATURES:
            self.assertIn(feature, current_state)
            self.assertIsInstance(current_state[feature], float)

    def test_4_h1_h12_trajectory_is_preserved(self):
        """4. H1-H12 trajectory is preserved."""
        result = run_soc_analysis(VALID_TIMESTAMP)
        forecast = result["forecast"]

        self.assertIn("horizons", forecast)
        horizons = forecast["horizons"]
        self.assertEqual(len(horizons), 12)

        for idx, point in enumerate(horizons, start=1):
            self.assertEqual(point["horizon"], idx)
            self.assertIn("predicted_attack_probability", point)
            self.assertIn("attack_category", point)
            self.assertIn("state_features", point)

    def test_5_risk_analysis_is_present(self):
        """5. Risk analysis is present."""
        result = run_soc_analysis(VALID_TIMESTAMP)
        risk = result["risk"]

        self.assertIn("risk_level", risk)
        self.assertIn(risk["risk_level"], ["LOW", "MODERATE", "HIGH"])
        self.assertIn("peak_probability", risk)
        self.assertIn("peak_horizon", risk)
        self.assertIn("elevated_horizons", risk)
        self.assertIn("trend", risk)

    def test_6_mitre_evidence_is_preserved(self):
        """6. MITRE evidence is preserved."""
        result = run_soc_analysis(VALID_TIMESTAMP)
        mitre = result["mitre"]

        self.assertIsInstance(mitre, list)
        if mitre:
            for item in mitre:
                self.assertIn("technique_id", item)
                self.assertIn("evidence_confidence", item)

    def test_7_graph_availability_reported_correctly(self):
        """7. Graph availability is reported correctly."""
        result = run_soc_analysis(VALID_TIMESTAMP)
        graph = result["graph"]

        self.assertIn("available", graph)
        self.assertTrue(graph["available"])
        self.assertGreater(graph["nodes"], 0)
        self.assertGreater(graph["edges"], 0)

        # Test unavailable graph scenario
        unavailable_response = build_soc_response(
            timestamp=VALID_TIMESTAMP,
            current_state={},
            graph=None,
            trajectory=[],
            risk={"peak_horizon": 1, "peak_probability": 0.0},
            decision={},
        )
        self.assertFalse(unavailable_response["graph"]["available"])

    def test_8_all_seven_interventions_preserved(self):
        """8. All 7 interventions are preserved."""
        result = run_soc_analysis(VALID_TIMESTAMP)
        counterfactuals = result["counterfactuals"]

        self.assertEqual(len(counterfactuals), 7)
        cf_types = [c["intervention"]["type"] for c in counterfactuals]
        self.assertEqual(set(cf_types), set(ALL_INTERVENTION_TYPES))

    def test_9_defense_decision_preserved(self):
        """9. Defense decision is preserved."""
        result = run_soc_analysis(VALID_TIMESTAMP)
        decision = result["decision"]

        self.assertIn("recommended_intervention", decision)
        self.assertIn("decision_basis", decision)
        self.assertIn("reasons", decision)
        self.assertIn("alternatives", decision)

        rec = decision["recommended_intervention"]
        self.assertIn(rec["type"], ALL_INTERVENTION_TYPES)

    def test_10_explanation_contains_observed_predicted_simulated_recommended(self):
        """10. Explanation contains OBSERVED, PREDICTED, SIMULATED, RECOMMENDED."""
        result = run_soc_analysis(VALID_TIMESTAMP)
        explanation = result["explanation"]

        self.assertIn("observed", explanation)
        self.assertIn("predicted", explanation)
        self.assertIn("simulated", explanation)
        self.assertIn("recommended", explanation)
        self.assertIn("summary", explanation)

        self.assertGreater(len(explanation["observed"]), 0)
        self.assertGreater(len(explanation["predicted"]), 0)
        self.assertGreater(len(explanation["simulated"]), 0)
        self.assertGreater(len(explanation["recommended"]), 0)

    def test_11_no_action_remains_valid_when_highest_scoring(self):
        """11. NO_ACTION remains valid when it is the highest-scoring option."""
        # Test directly via explanation and decision response
        rec = {"type": "NO_ACTION", "target": 0}
        decision = {
            "recommended_intervention": rec,
            "decision_basis": {
                "defense_score": 0.0,
                "risk_reduction": 0.0,
                "operational_cost": 0.0,
                "service_disruption_cost": 0.0,
            },
            "reasons": ["Baseline risk is low."],
        }
        risk = {"risk_level": "LOW", "peak_probability": 0.05, "peak_horizon": 1, "trend": "stable"}
        exp = generate_soc_explanation(
            current_state={},
            risk=risk,
            trajectory=[],
            decision=decision,
        )
        self.assertIn("NO_ACTION", exp["summary"])
        self.assertEqual(rec["type"], "NO_ACTION")

    def test_12_existing_defense_score_remains_unchanged(self):
        """12. Existing defense score remains unchanged."""
        result = run_soc_analysis(VALID_TIMESTAMP)
        counterfactuals = result["counterfactuals"]

        for cf in counterfactuals:
            if cf.get("available", True):
                expected = round(
                    cf["risk_reduction"] - cf["operational_cost"] - cf["service_disruption_cost"],
                    6,
                )
                self.assertAlmostEqual(cf["defense_score"], expected, places=5)


if __name__ == "__main__":
    unittest.main()

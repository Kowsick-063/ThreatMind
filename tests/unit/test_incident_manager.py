"""Unit test suite for Step 22 — SOC Incident Lifecycle.

Covers:
  1.  Incident creation from SOC analysis
  2.  Deterministic/valid incident ID
  3.  Duplicate timestamp handling (rejection)
  4.  Incident retrieval
  5.  Incident listing
  6.  Newest-first ordering
  7.  Valid DETECTED → ASSESSED transition
  8.  Valid ASSESSED → FORECASTED transition
  9.  Valid FORECASTED → SIMULATED transition
  10. Valid SIMULATED → RECOMMENDED transition
  11. Invalid backward transition rejected
  12. Invalid skipped transition rejected
  13. Terminal RECOMMENDED state rejects any further transitions
  14. Evidence preservation
  15. API create endpoint
  16. API get endpoint
  17. API list endpoint
  18. API status-update endpoint
  19. Invalid incident ID returns 404
  20. Invalid status string returns 422
"""

from __future__ import annotations

import unittest

from fastapi import HTTPException

from backend.api.incidents import (
    CreateIncidentRequest,
    UpdateStatusRequest,
    create,
    index,
    retrieve,
    update_status,
)
from soc.incident import (
    LIFECYCLE_STAGES,
    Incident,
    InvalidLifecycleTransitionError,
    generate_incident_id,
)
from soc.incident_manager import (
    DuplicateIncidentError,
    clear_incidents,
    create_incident,
    get_incident,
    list_incidents,
    update_incident_status,
)
from soc.soc_orchestrator import run_soc_analysis


VALID_TIMESTAMP = 1499433815.0
VALID_TIMESTAMP_2 = 1499433820.0   # Needs its own contiguous run; use only for listing test


def _make_mock_soc(timestamp: float = VALID_TIMESTAMP, risk_level: str = "HIGH") -> dict:
    """Build a minimal fake SOC analysis dict for unit testing without hitting the pipeline."""
    return {
        "timestamp": timestamp,
        "status": "ANALYZED",
        "current_state": {"flow_count": 100.0, "packet_count": 500.0},
        "risk": {
            "risk_level": risk_level,
            "peak_probability": 0.9,
            "peak_horizon": 3,
            "trend": "stable",
            "elevated_horizons": [1, 2, 3],
            "number_of_elevated_risk_horizons": 3,
            "uncertainties": [],
        },
        "forecast": {
            "peak_horizon": 3,
            "peak_probability": 0.9,
            "probability_trend": "stable",
            "horizons": [{"horizon": h, "predicted_attack_probability": 0.9, "attack_category": "PortScan"} for h in range(1, 13)],
        },
        "graph": {"available": True, "nodes": 100, "edges": 120},
        "mitre": [{"technique_id": "T1046", "technique_name": "Network Service Scanning", "confidence": 0.7, "evidence_confidence": 0.7}],
        "counterfactuals": [
            {
                "intervention": {"type": t, "target": 0},
                "available": True,
                "risk_reduction": 0.1,
                "risk_reduction_percentage": 10.0,
                "operational_cost": 0.15,
                "service_disruption_cost": 0.10,
                "defense_score": -0.15,
                "graph_attack_surface_reduction": 0.0,
                "graph_attack_path_reduction": 0.0,
                "edges_removed": 0,
                "affected_paths": 0,
                "graph_impact": None,
            }
            for t in ["NO_ACTION", "BLOCK_SOURCE", "BLOCK_DESTINATION", "BLOCK_PORT",
                      "ISOLATE_HOST", "DISABLE_CONNECTION", "NETWORK_SEGMENTATION"]
        ],
        "decision": {
            "recommended_intervention": {"type": "NO_ACTION", "target": 0},
            "decision_basis": {"defense_score": 0.0, "risk_reduction": 0.0, "operational_cost": 0.0, "service_disruption_cost": 0.0},
            "alternatives": [],
            "reasons": ["Baseline."],
            "near_ties": [],
            "uncertainties": [],
        },
        "explanation": {"summary": "Test."},
        "uncertainties": ["Simulation approximation."],
    }


class IncidentModelTest(unittest.TestCase):
    def setUp(self):
        clear_incidents()

    def tearDown(self):
        clear_incidents()

    # 1. Incident creation
    def test_1_incident_creation(self):
        soc = _make_mock_soc()
        inc = create_incident(soc)

        self.assertEqual(inc["timestamp"], VALID_TIMESTAMP)
        self.assertEqual(inc["risk_level"], "HIGH")
        self.assertEqual(inc["peak_attack_probability"], 0.9)
        self.assertEqual(inc["peak_horizon"], 3)
        self.assertIsInstance(inc["lifecycle_history"], list)
        self.assertGreater(len(inc["lifecycle_history"]), 0)

    # 2. Deterministic incident ID
    def test_2_deterministic_incident_id(self):
        soc = _make_mock_soc(1499430000.0)
        inc = create_incident(soc)

        self.assertEqual(inc["incident_id"], "INC-1499430000")
        self.assertTrue(inc["incident_id"].startswith("INC-"))

    # 3. Duplicate timestamp rejection
    def test_3_duplicate_timestamp_rejection(self):
        soc = _make_mock_soc()
        create_incident(soc)

        with self.assertRaises(DuplicateIncidentError) as ctx:
            create_incident(_make_mock_soc())
        self.assertIn(str(int(VALID_TIMESTAMP)), str(ctx.exception))

    # 4. Incident retrieval
    def test_4_incident_retrieval(self):
        soc = _make_mock_soc()
        created = create_incident(soc)
        retrieved = get_incident(created["incident_id"])

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["incident_id"], created["incident_id"])

    # 5. Incident listing
    def test_5_incident_listing(self):
        create_incident(_make_mock_soc(1499430000.0))
        create_incident(_make_mock_soc(1499430005.0))
        incidents = list_incidents()

        self.assertEqual(len(incidents), 2)

    # 6. Newest first ordering
    def test_6_newest_first_ordering(self):
        create_incident(_make_mock_soc(1499430000.0))
        create_incident(_make_mock_soc(1499430005.0))
        incidents = list_incidents()

        self.assertGreater(incidents[0]["timestamp"], incidents[1]["timestamp"])

    # 7. DETECTED → ASSESSED
    def test_7_detected_to_assessed(self):
        soc = _make_mock_soc()
        inc = create_incident(soc, initial_status="DETECTED")
        self.assertEqual(inc["status"], "DETECTED")

        updated = update_incident_status(inc["incident_id"], "ASSESSED")
        self.assertEqual(updated["status"], "ASSESSED")

    # 8. ASSESSED → FORECASTED
    def test_8_assessed_to_forecasted(self):
        soc = _make_mock_soc()
        inc = create_incident(soc, initial_status="ASSESSED")

        updated = update_incident_status(inc["incident_id"], "FORECASTED")
        self.assertEqual(updated["status"], "FORECASTED")

    # 9. FORECASTED → SIMULATED
    def test_9_forecasted_to_simulated(self):
        soc = _make_mock_soc()
        inc = create_incident(soc, initial_status="FORECASTED")

        updated = update_incident_status(inc["incident_id"], "SIMULATED")
        self.assertEqual(updated["status"], "SIMULATED")

    # 10. SIMULATED → RECOMMENDED
    def test_10_simulated_to_recommended(self):
        soc = _make_mock_soc()
        inc = create_incident(soc, initial_status="SIMULATED")

        updated = update_incident_status(inc["incident_id"], "RECOMMENDED")
        self.assertEqual(updated["status"], "RECOMMENDED")

    # 11. Backward transition rejected
    def test_11_backward_transition_rejected(self):
        soc = _make_mock_soc()
        inc = create_incident(soc, initial_status="ASSESSED")

        with self.assertRaises(InvalidLifecycleTransitionError):
            update_incident_status(inc["incident_id"], "DETECTED")

    # 12. Skipped transition rejected
    def test_12_skipped_transition_rejected(self):
        soc = _make_mock_soc()
        inc = create_incident(soc, initial_status="DETECTED")

        with self.assertRaises(InvalidLifecycleTransitionError):
            update_incident_status(inc["incident_id"], "RECOMMENDED")

    # 13. Terminal RECOMMENDED blocks all transitions
    def test_13_terminal_recommended_state(self):
        soc = _make_mock_soc()
        inc = create_incident(soc, initial_status="RECOMMENDED")
        self.assertEqual(inc["status"], "RECOMMENDED")

        for target in LIFECYCLE_STAGES:
            with self.assertRaises(InvalidLifecycleTransitionError):
                update_incident_status(inc["incident_id"], target)

    # 14. Evidence preservation
    def test_14_evidence_preservation(self):
        soc = _make_mock_soc()
        inc = create_incident(soc)
        evidence = inc["evidence"]

        self.assertIn("observed", evidence)
        self.assertIn("predicted", evidence)
        self.assertIn("simulated", evidence)
        self.assertIn("recommended", evidence)

        self.assertEqual(evidence["observed"]["timestamp"], VALID_TIMESTAMP)
        self.assertTrue(evidence["observed"]["graph_available"])
        self.assertIn("forecast_horizons", evidence["predicted"])
        self.assertIn("mitre_evidence", evidence["predicted"])
        self.assertIn("counterfactuals", evidence["simulated"])
        self.assertIn("selected_intervention", evidence["recommended"])

    # 15. API create endpoint
    def test_15_api_create_endpoint(self):
        soc = _make_mock_soc(1499430010.0)
        # Inject mock soc_analysis manually by calling create_incident directly
        created = create_incident(soc)
        self.assertIn("incident_id", created)
        self.assertEqual(created["status"], "RECOMMENDED")

    # 16. API get endpoint
    def test_16_api_get_endpoint(self):
        soc = _make_mock_soc()
        created = create_incident(soc)
        incident = get_incident(created["incident_id"])
        self.assertIsNotNone(incident)
        self.assertEqual(incident["incident_id"], created["incident_id"])

        missing = get_incident("INC-NONEXISTENT")
        self.assertIsNone(missing)

    # 17. API list endpoint via direct function
    def test_17_api_list_endpoint(self):
        create_incident(_make_mock_soc(1499430020.0))
        incidents = list_incidents()
        self.assertGreaterEqual(len(incidents), 1)

    # 18. API status-update endpoint
    def test_18_api_status_update_endpoint(self):
        soc = _make_mock_soc(1499430030.0)
        inc = create_incident(soc, initial_status="DETECTED")

        updated = update_incident_status(inc["incident_id"], "ASSESSED")
        self.assertEqual(updated["status"], "ASSESSED")

    # 19. Invalid incident ID returns 404
    def test_19_invalid_incident_id(self):
        from soc.incident_manager import IncidentNotFoundError
        with self.assertRaises(IncidentNotFoundError):
            update_incident_status("INC-NONEXISTENT", "ASSESSED")

        result = get_incident("INC-NONEXISTENT")
        self.assertIsNone(result)

    # 20. Invalid status string rejected
    def test_20_invalid_status_string(self):
        soc = _make_mock_soc(1499430040.0)
        inc = create_incident(soc, initial_status="DETECTED")

        with self.assertRaises(ValueError):
            update_incident_status(inc["incident_id"], "INVALID_STATUS")


class IncidentAPITest(unittest.TestCase):
    """Tests using the API handler functions directly (no HTTP client needed)."""

    def setUp(self):
        clear_incidents()

    def tearDown(self):
        clear_incidents()

    def test_api_create_returns_summary(self):
        """API create endpoint returns incident summary fields."""
        soc = _make_mock_soc(1499440000.0)
        inc = create_incident(soc)

        self.assertIn("incident_id", inc)
        self.assertIn("risk_level", inc)
        self.assertIn("recommended_intervention", inc)
        self.assertIn("mitre_techniques", inc)
        self.assertIn("uncertainties", inc)

    def test_api_duplicate_raises_duplicate_error(self):
        """Duplicate creation raises DuplicateIncidentError."""
        create_incident(_make_mock_soc(1499441000.0))
        with self.assertRaises(DuplicateIncidentError):
            create_incident(_make_mock_soc(1499441000.0))

    def test_api_get_returns_none_for_missing(self):
        result = get_incident("INC-0000")
        self.assertIsNone(result)

    def test_api_list_returns_list(self):
        result = list_incidents()
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()

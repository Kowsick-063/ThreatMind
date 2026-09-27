from __future__ import annotations

import sys
from pathlib import Path

# ============================================================
# PROJECT IMPORT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

import pytest

from soc.incident import (
    LIFECYCLE_STAGES,
    InvalidLifecycleTransitionError,
)

from soc.incident_manager import (
    DuplicateIncidentError,
    IncidentNotFoundError,
    clear_incidents,
    create_incident,
    get_incident,
    list_incidents,
    update_incident_status,
)


# ============================================================
# TEST DATA
# ============================================================

VALID_TIMESTAMP = 1499433815.0
VALID_TIMESTAMP_2 = 1499433820.0


def make_soc(
    timestamp: float = VALID_TIMESTAMP,
    risk_level: str = "HIGH",
) -> dict:
    """
    Create a representative SOC analysis result
    for incident-manager testing.
    """

    return {
        "timestamp": timestamp,

        "status": "ANALYZED",

        "current_state": {
            "flow_count": 100.0,
            "packet_count": 500.0,
        },

        "risk": {
            "risk_level": risk_level,
            "peak_probability": 0.9,
            "peak_horizon": 3,
            "trend": "stable",
            "elevated_horizons": [
                1,
                2,
                3,
            ],
            "number_of_elevated_risk_horizons": 3,
            "uncertainties": [],
        },

        "forecast": {
            "peak_horizon": 3,
            "peak_probability": 0.9,
            "probability_trend": "stable",

            "horizons": [
                {
                    "horizon": horizon,
                    "predicted_attack_probability": 0.9,
                    "attack_category": "PortScan",
                }
                for horizon in range(1, 13)
            ],
        },

        "graph": {
            "available": True,
            "nodes": 100,
            "edges": 120,
        },

        "mitre": [
            {
                "technique_id": "T1046",
                "technique_name": "Network Service Scanning",
                "confidence": 0.7,
                "evidence_confidence": 0.7,
            }
        ],

        "counterfactuals": [
            {
                "intervention": {
                    "type": intervention_type,
                    "target": 0,
                },

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

            for intervention_type in [
                "NO_ACTION",
                "BLOCK_SOURCE",
                "BLOCK_DESTINATION",
                "BLOCK_PORT",
                "ISOLATE_HOST",
                "DISABLE_CONNECTION",
                "NETWORK_SEGMENTATION",
            ]
        ],

        "decision": {
            "recommended_intervention": {
                "type": "NO_ACTION",
                "target": 0,
            },

            "decision_basis": {
                "defense_score": 0.0,
                "risk_reduction": 0.0,
                "operational_cost": 0.0,
                "service_disruption_cost": 0.0,
            },

            "alternatives": [],

            "reasons": [
                "Baseline.",
            ],

            "near_ties": [],

            "uncertainties": [],
        },

        "explanation": {
            "summary": "Test incident.",
        },

        "uncertainties": [
            "Simulation approximation.",
        ],
    }


# ============================================================
# FIXTURE
# ============================================================

@pytest.fixture(autouse=True)
def reset_incidents():
    """
    Ensure every test starts with an empty
    incident store.
    """

    clear_incidents()

    yield

    clear_incidents()


# ============================================================
# 1. INCIDENT CREATION
# ============================================================

def test_incident_creation():
    incident = create_incident(
        make_soc()
    )

    assert incident["incident_id"] == "INC-1499433815"

    assert incident["status"] == "RECOMMENDED"

    assert incident["risk_level"] == "HIGH"

    assert incident["peak_attack_probability"] == 0.9

    assert incident["peak_horizon"] == 3

    assert incident["attack_category"] == "PortScan"


def test_deterministic_incident_id():
    incident = create_incident(
        make_soc(
            VALID_TIMESTAMP
        )
    )

    assert (
        incident["incident_id"]
        == "INC-1499433815"
    )


def test_duplicate_timestamp_rejected():
    create_incident(
        make_soc(
            VALID_TIMESTAMP
        )
    )

    with pytest.raises(
        DuplicateIncidentError
    ):
        create_incident(
            make_soc(
                VALID_TIMESTAMP
            )
        )


# ============================================================
# 2. INCIDENT RETRIEVAL
# ============================================================

def test_incident_retrieval():
    created = create_incident(
        make_soc()
    )

    retrieved = get_incident(
        created["incident_id"]
    )

    assert retrieved is not None

    assert (
        retrieved["incident_id"]
        == created["incident_id"]
    )


def test_missing_incident_returns_none():
    result = get_incident(
        "INC-DOES-NOT-EXIST"
    )

    assert result is None


# ============================================================
# 3. INCIDENT LISTING
# ============================================================

def test_list_incidents():
    create_incident(
        make_soc(
            VALID_TIMESTAMP
        )
    )

    incidents = list_incidents()

    assert len(incidents) == 1

    assert (
        incidents[0]["incident_id"]
        == "INC-1499433815"
    )


def test_list_incidents_newest_first():
    create_incident(
        make_soc(
            VALID_TIMESTAMP
        )
    )

    create_incident(
        make_soc(
            VALID_TIMESTAMP_2
        )
    )

    incidents = list_incidents()

    assert len(incidents) == 2

    assert (
        incidents[0]["incident_id"]
        == "INC-1499433820"
    )

    assert (
        incidents[1]["incident_id"]
        == "INC-1499433815"
    )


# ============================================================
# 4. DETECTED → ASSESSED
# ============================================================

def test_detected_to_assessed():
    incident = create_incident(
        make_soc(),
        initial_status="DETECTED",
    )

    updated = update_incident_status(
        incident["incident_id"],
        "ASSESSED",
    )

    assert updated["status"] == "ASSESSED"


# ============================================================
# 5. ASSESSED → FORECASTED
# ============================================================

def test_assessed_to_forecasted():
    incident = create_incident(
        make_soc(),
        initial_status="ASSESSED",
    )

    updated = update_incident_status(
        incident["incident_id"],
        "FORECASTED",
    )

    assert updated["status"] == "FORECASTED"


# ============================================================
# 6. FORECASTED → SIMULATED
# ============================================================

def test_forecasted_to_simulated():
    incident = create_incident(
        make_soc(),
        initial_status="FORECASTED",
    )

    updated = update_incident_status(
        incident["incident_id"],
        "SIMULATED",
    )

    assert updated["status"] == "SIMULATED"


# ============================================================
# 7. SIMULATED → RECOMMENDED
# ============================================================

def test_simulated_to_recommended():
    incident = create_incident(
        make_soc(),
        initial_status="SIMULATED",
    )

    updated = update_incident_status(
        incident["incident_id"],
        "RECOMMENDED",
    )

    assert (
        updated["status"]
        == "RECOMMENDED"
    )


# ============================================================
# 8. BACKWARD TRANSITION REJECTED
# ============================================================

def test_backward_transition_rejected():
    incident = create_incident(
        make_soc(),
        initial_status="ASSESSED",
    )

    with pytest.raises(
        InvalidLifecycleTransitionError
    ):
        update_incident_status(
            incident["incident_id"],
            "DETECTED",
        )


# ============================================================
# 9. SKIPPED TRANSITION REJECTED
# ============================================================

def test_skipped_transition_rejected():
    incident = create_incident(
        make_soc(),
        initial_status="DETECTED",
    )

    with pytest.raises(
        InvalidLifecycleTransitionError
    ):
        update_incident_status(
            incident["incident_id"],
            "FORECASTED",
        )


# ============================================================
# 10. RECOMMENDED IS TERMINAL
# ============================================================

def test_recommended_is_terminal():
    incident = create_incident(
        make_soc(),
        initial_status="RECOMMENDED",
    )

    with pytest.raises(
        InvalidLifecycleTransitionError
    ):
        update_incident_status(
            incident["incident_id"],
            "DETECTED",
        )


# ============================================================
# 11. INVALID STATUS
# ============================================================

def test_invalid_status_rejected():
    incident = create_incident(
        make_soc(),
        initial_status="DETECTED",
    )

    with pytest.raises(
        ValueError
    ):
        update_incident_status(
            incident["incident_id"],
            "INVALID_STATUS",
        )


# ============================================================
# 12. MISSING INCIDENT UPDATE
# ============================================================

def test_missing_incident_update_rejected():
    with pytest.raises(
        IncidentNotFoundError
    ):
        update_incident_status(
            "INC-DOES-NOT-EXIST",
            "ASSESSED",
        )


# ============================================================
# 13. EVIDENCE STRUCTURE
# ============================================================

def test_evidence_structure():
    incident = create_incident(
        make_soc()
    )

    evidence = incident["evidence"]

    assert "observed" in evidence

    assert "predicted" in evidence

    assert "simulated" in evidence

    assert "recommended" in evidence


# ============================================================
# 14. OBSERVED EVIDENCE
# ============================================================

def test_observed_evidence():
    incident = create_incident(
        make_soc()
    )

    observed = (
        incident["evidence"]["observed"]
    )

    assert (
        observed["timestamp"]
        == VALID_TIMESTAMP
    )

    assert (
        observed["graph_available"]
        is True
    )


# ============================================================
# 15. PREDICTED EVIDENCE
# ============================================================

def test_predicted_evidence():
    incident = create_incident(
        make_soc()
    )

    predicted = (
        incident["evidence"]["predicted"]
    )

    assert (
        "forecast_horizons"
        in predicted
    )

    assert (
        len(
            predicted["forecast_horizons"]
        )
        == 12
    )

    assert (
        "mitre_evidence"
        in predicted
    )

    assert (
        len(
            predicted["mitre_evidence"]
        )
        == 1
    )


# ============================================================
# 16. SIMULATED EVIDENCE
# ============================================================

def test_simulated_evidence():
    incident = create_incident(
        make_soc()
    )

    simulated = (
        incident["evidence"]["simulated"]
    )

    assert (
        "counterfactuals"
        in simulated
    )

    assert (
        len(
            simulated["counterfactuals"]
        )
        == 7
    )


# ============================================================
# 17. RECOMMENDED EVIDENCE
# ============================================================

def test_recommended_evidence():
    incident = create_incident(
        make_soc()
    )

    recommended = (
        incident["evidence"]["recommended"]
    )

    assert (
        "selected_intervention"
        in recommended
    )

    assert (
        recommended[
            "selected_intervention"
        ]["type"]
        == "NO_ACTION"
    )


# ============================================================
# 18. LIFECYCLE HISTORY
# ============================================================

def test_lifecycle_history_created():
    incident = create_incident(
        make_soc(),
        initial_status="DETECTED",
    )

    assert (
        "lifecycle_history"
        in incident
    )

    assert isinstance(
        incident["lifecycle_history"],
        list,
    )

    assert (
        len(
            incident["lifecycle_history"]
        )
        == 1
    )

    assert (
        incident[
            "lifecycle_history"
        ][0]["status"]
        == "DETECTED"
    )


# ============================================================
# 19. LIFECYCLE HISTORY UPDATE
# ============================================================

def test_lifecycle_history_records_transition():
    incident = create_incident(
        make_soc(),
        initial_status="DETECTED",
    )

    incident_id = incident[
        "incident_id"
    ]

    update_incident_status(
        incident_id,
        "ASSESSED",
    )

    updated = get_incident(
        incident_id
    )

    assert updated is not None

    assert (
        len(
            updated["lifecycle_history"]
        )
        == 2
    )

    assert (
        updated[
            "lifecycle_history"
        ][0]["status"]
        == "DETECTED"
    )

    assert (
        updated[
            "lifecycle_history"
        ][1]["status"]
        == "ASSESSED"
    )


# ============================================================
# 20. COMPLETE LIFECYCLE
# ============================================================

def test_complete_incident_lifecycle():
    incident = create_incident(
        make_soc(),
        initial_status="DETECTED",
    )

    incident_id = incident[
        "incident_id"
    ]

    update_incident_status(
        incident_id,
        "ASSESSED",
    )

    update_incident_status(
        incident_id,
        "FORECASTED",
    )

    update_incident_status(
        incident_id,
        "SIMULATED",
    )

    update_incident_status(
        incident_id,
        "RECOMMENDED",
    )

    final = get_incident(
        incident_id
    )

    assert final is not None

    assert (
        final["status"]
        == "RECOMMENDED"
    )

    assert [
        item["status"]
        for item in final[
            "lifecycle_history"
        ]
    ] == [
        "DETECTED",
        "ASSESSED",
        "FORECASTED",
        "SIMULATED",
        "RECOMMENDED",
    ]


# ============================================================
# 21. LIFECYCLE CONSTANT
# ============================================================

def test_lifecycle_stages():
    assert LIFECYCLE_STAGES == (
        "DETECTED",
        "ASSESSED",
        "FORECASTED",
        "SIMULATED",
        "RECOMMENDED",
    )
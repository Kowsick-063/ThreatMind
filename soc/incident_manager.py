"""In-memory incident manager for ThreatMind SOC workflows.

Handles incident creation, retrieval, newest-first listing, and strictly
forward lifecycle transitions.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from soc.incident import (
    LIFECYCLE_STAGES,
    Incident,
    InvalidLifecycleTransitionError,
    generate_incident_id,
)


class DuplicateIncidentError(ValueError):
    """Raised when an incident for a given timestamp already exists."""
    pass


class IncidentNotFoundError(KeyError):
    """Raised when the requested incident ID does not exist."""
    pass


# Thread-safe in-memory stores
_incidents_by_id: dict[str, Incident] = {}
_incidents_by_timestamp: dict[float, str] = {}


def create_incident(
    soc_analysis: dict[str, Any],
    initial_status: str = "RECOMMENDED",
) -> dict[str, Any]:
    """Create a structured incident object from a completed SOC analysis.

    Parameters
    ----------
    soc_analysis:
        Output dict from run_soc_analysis or build_soc_response.
    initial_status:
        Starting status in the lifecycle (defaults to RECOMMENDED for completed analyses).

    Returns
    -------
    dict representing the stored incident.
    """
    timestamp = float(soc_analysis["timestamp"])

    # Duplicate check
    if timestamp in _incidents_by_timestamp:
        existing_id = _incidents_by_timestamp[timestamp]
        raise DuplicateIncidentError(
            f"An incident already exists for timestamp {timestamp}: '{existing_id}'."
        )

    incident_id = generate_incident_id(timestamp)
    now = datetime.now(timezone.utc).isoformat()

    current_state = soc_analysis.get("current_state", {})
    risk = soc_analysis.get("risk", {})
    forecast = soc_analysis.get("forecast", {})
    horizons = forecast.get("horizons", [])
    graph = soc_analysis.get("graph", {})
    mitre = soc_analysis.get("mitre", [])
    counterfactuals = soc_analysis.get("counterfactuals", [])
    decision = soc_analysis.get("decision", {})
    uncertainties = soc_analysis.get("uncertainties", [])

    # Extract primary attack category from forecast
    attack_category = "UNKNOWN"
    if horizons:
        attack_category = str(horizons[0].get("attack_category", "UNKNOWN"))

    recommended_intervention = decision.get("recommended_intervention", {})
    decision_basis = decision.get("decision_basis", {})

    # Tiered evidence preservation
    evidence = {
        "observed": {
            "timestamp": timestamp,
            "current_state": current_state,
            "graph_available": bool(graph.get("available", False)),
            "graph_summary": graph if graph.get("available") else None,
        },
        "predicted": {
            "forecast_horizons": horizons,
            "peak_horizon": forecast.get("peak_horizon", 1),
            "peak_probability": forecast.get("peak_probability", 0.0),
            "probability_trend": forecast.get("probability_trend", "stable"),
            "attack_category": attack_category,
            "mitre_evidence": mitre,
        },
        "simulated": {
            "counterfactuals": counterfactuals,
        },
        "recommended": {
            "selected_intervention": recommended_intervention,
            "decision_basis": decision_basis,
            "alternatives": decision.get("alternatives", []),
            "near_ties": decision.get("near_ties", []),
            "reasons": decision.get("reasons", []),
        },
    }

    # Record lifecycle history
    status_clean = initial_status.upper().strip()
    if status_clean not in LIFECYCLE_STAGES:
        raise ValueError(
            f"Invalid initial status: '{initial_status}'. Must be one of {LIFECYCLE_STAGES}."
        )

    if status_clean == "RECOMMENDED":
        # Full analysis completed all stages
        lifecycle_history = [
            {"status": stage, "transitioned_at": now}
            for stage in LIFECYCLE_STAGES
        ]
    else:
        stage_idx = LIFECYCLE_STAGES.index(status_clean)
        lifecycle_history = [
            {"status": stage, "transitioned_at": now}
            for stage in LIFECYCLE_STAGES[: stage_idx + 1]
        ]

    incident = Incident(
        incident_id=incident_id,
        timestamp=timestamp,
        status=status_clean,
        created_at=now,
        risk_level=risk.get("risk_level", "LOW"),
        peak_attack_probability=float(forecast.get("peak_probability", risk.get("peak_probability", 0.0))),
        peak_horizon=int(forecast.get("peak_horizon", risk.get("peak_horizon", 1))),
        attack_category=attack_category,
        mitre_techniques=mitre,
        recommended_intervention=recommended_intervention,
        decision_basis=decision_basis,
        uncertainties=uncertainties,
        evidence=evidence,
        lifecycle_history=lifecycle_history,
    )

    _incidents_by_id[incident_id] = incident
    _incidents_by_timestamp[timestamp] = incident_id

    return incident.to_dict()


def get_incident(incident_id: str) -> dict[str, Any] | None:
    """Retrieve an incident by its incident_id."""
    incident = _incidents_by_id.get(incident_id)
    return incident.to_dict() if incident is not None else None


def list_incidents() -> list[dict[str, Any]]:
    """Return all stored incidents ordered newest first by timestamp."""
    sorted_incidents = sorted(
        _incidents_by_id.values(),
        key=lambda inc: inc.timestamp,
        reverse=True,
    )
    return [inc.to_dict() for inc in sorted_incidents]


def update_incident_status(incident_id: str, status: str) -> dict[str, Any]:
    """Transition an incident to a new lifecycle status.

    Raises
    ------
    IncidentNotFoundError
        If incident_id is not found.
    InvalidLifecycleTransitionError
        If the status transition violates forward-only rules.
    ValueError
        If status is unrecognized.
    """
    incident = _incidents_by_id.get(incident_id)
    if incident is None:
        raise IncidentNotFoundError(f"Incident '{incident_id}' not found.")

    incident.transition_to(status)
    return incident.to_dict()


def clear_incidents() -> None:
    """Reset the in-memory incident store. Used for test isolation."""
    _incidents_by_id.clear()
    _incidents_by_timestamp.clear()


__all__ = [
    "DuplicateIncidentError",
    "IncidentNotFoundError",
    "clear_incidents",
    "create_incident",
    "get_incident",
    "list_incidents",
    "update_incident_status",
]

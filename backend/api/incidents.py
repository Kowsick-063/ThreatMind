"""Incident management API endpoints for ThreatMind SOC."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from soc.incident import LIFECYCLE_STAGES, InvalidLifecycleTransitionError
from soc.incident_manager import (
    DuplicateIncidentError,
    IncidentNotFoundError,
    create_incident,
    get_incident,
    list_incidents,
    update_incident_status,
)
from soc.soc_orchestrator import run_soc_analysis

router = APIRouter(prefix="/api/incidents", tags=["Incidents"])


class CreateIncidentRequest(BaseModel):
    timestamp: float = Field(..., description="Unix timestamp to analyze and create an incident for.")


class UpdateStatusRequest(BaseModel):
    status: str = Field(..., description=f"Target lifecycle status. One of: {', '.join(LIFECYCLE_STAGES)}")


@router.post("")
def create(request: CreateIncidentRequest) -> dict[str, Any]:
    """Run SOC analysis and create a structured incident."""
    try:
        soc_analysis = run_soc_analysis(timestamp=request.timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": str(exc), "available": False, "timestamp": request.timestamp}) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail={"error": str(exc), "available": False}) from exc

    try:
        incident = create_incident(
    soc_analysis,
    initial_status="DETECTED",
)
    except DuplicateIncidentError as exc:
        raise HTTPException(status_code=409, detail={"error": str(exc)}) from exc

    return {
        "incident_id": incident["incident_id"],
        "status": incident["status"],
        "risk_level": incident["risk_level"],
        "peak_attack_probability": incident["peak_attack_probability"],
        "peak_horizon": incident["peak_horizon"],
        "attack_category": incident["attack_category"],
        "recommended_intervention": incident["recommended_intervention"],
        "mitre_techniques": incident["mitre_techniques"],
        "uncertainties": incident["uncertainties"],
    }


@router.get("")
def index() -> list[dict[str, Any]]:
    """List all incidents, newest first."""
    return list_incidents()


@router.get("/{incident_id}")
def retrieve(incident_id: str) -> dict[str, Any]:
    """Retrieve a complete incident by ID."""
    incident = get_incident(incident_id)
    if incident is None:
        raise HTTPException(status_code=404, detail={"error": f"Incident '{incident_id}' not found."})
    return incident


@router.patch("/{incident_id}/status")
def update_status(incident_id: str, request: UpdateStatusRequest) -> dict[str, Any]:
    """Advance an incident's lifecycle status."""
    target = request.status.upper().strip()
    if target not in LIFECYCLE_STAGES:
        raise HTTPException(
            status_code=422,
            detail={"error": f"Unknown status: '{request.status}'. Allowed: {', '.join(LIFECYCLE_STAGES)}."},
        )
    try:
        return update_incident_status(incident_id, target)
    except IncidentNotFoundError as exc:
        raise HTTPException(status_code=404, detail={"error": str(exc)}) from exc
    except InvalidLifecycleTransitionError as exc:
        raise HTTPException(status_code=409, detail={"error": str(exc)}) from exc


__all__ = ["router"]

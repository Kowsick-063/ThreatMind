"""POST /api/forecast/trajectory — Multi-step attack trajectory generation endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from forecasting.trajectory import generate_attack_trajectory

router = APIRouter(prefix="/api/forecast", tags=["Forecasting"])


class ForecastTrajectoryRequest(BaseModel):
    """Input payload for trajectory forecasting."""

    timestamp: float = Field(
        ...,
        description=(
            "Unix timestamp (seconds) of the current state. "
            "Requires a contiguous 60-state (5-minute) historical sequence in the dataset."
        ),
    )


@router.post("/trajectory")
def forecast_trajectory(request: ForecastTrajectoryRequest) -> dict[str, Any]:
    """Generate a structured H1-H12 attack trajectory fusing ThreatMind models.

    Integrates:
      - Production LSTM world model (absolute state rollout)
      - Delta-LSTM world model (change signal rollout)
      - AttackProbabilityModel + 12 horizon-specific Platt calibrators
      - AttackCategoryEstimator
      - MITRE ATT&CK technique evidence mapper

    Requires a contiguous 60-state history ending at the requested timestamp.
    If unavailable or if temporal gaps exist, returns HTTP 422 with explicit error details.
    """
    try:
        result = generate_attack_trajectory(current_timestamp=request.timestamp)
        return result
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": str(exc),
                "message": "Contiguous historical sequence unavailable for the requested timestamp.",
                "available": False,
                "timestamp": request.timestamp,
            },
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "error": str(exc),
                "available": False,
            },
        ) from exc


__all__ = ["router"]

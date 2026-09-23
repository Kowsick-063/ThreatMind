"""POST /api/decision/recommend — Real-time SOC Defense Decision Engine endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from decision.defense_decision import generate_defense_decision

router = APIRouter(prefix="/api/decision", tags=["Decision"])


class DecisionRecommendRequest(BaseModel):
    """Input payload for defense decision recommendation."""

    timestamp: float = Field(
        ...,
        description="Unix timestamp (seconds) of the network state to analyze and recommend defense for.",
    )


@router.post("/recommend")
def recommend_defense(request: DecisionRecommendRequest) -> dict[str, Any]:
    """Generate a deterministic SOC defense recommendation from trajectory and counterfactuals.

    Integrates:
      - Trajectory forecasting (H1-H12)
      - Horizon risk analysis
      - Target selection from real network graph topology
      - Counterfactual simulation of 7 interventions
      - Defense score ranking and explainability
    """
    try:
        return generate_defense_decision(timestamp=request.timestamp)
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": str(exc),
                "message": "Contiguous historical sequence or snapshot unavailable for timestamp.",
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

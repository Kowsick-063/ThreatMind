"""POST /api/soc/analyze — Real-time SOC end-to-end orchestration endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from soc.soc_orchestrator import run_soc_analysis

router = APIRouter(prefix="/api/soc", tags=["SOC"])


class SocAnalyzeRequest(BaseModel):
    """Input payload for end-to-end SOC analysis."""

    timestamp: float = Field(
        ...,
        description="Unix timestamp (seconds) of the network state to execute end-to-end SOC analysis for.",
    )


@router.post("/analyze")
def analyze_soc(request: SocAnalyzeRequest) -> dict[str, Any]:
    """Execute complete ThreatMind pipeline and return structured SOC response.

    Orchestrates:
      1. Historical 60-state sequence retrieval
      2. 26-feature network state evaluation
      3. Graph snapshot inspection
      4. H1-H12 attack trajectory forecasting
      5. Trajectory risk analysis
      6. MITRE ATT&CK evidence mapping
      7. Counterfactual simulation of 7 interventions
      8. Defense score ranking and recommendation
      9. Structured tiered SOC response
    """
    try:
        return run_soc_analysis(timestamp=request.timestamp)
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

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class InterventionInput(BaseModel):
    type: str = Field(..., description="Simulation-only intervention type")
    target: int | str = Field(default=0, description="Intervention target")


class CounterfactualRequest(BaseModel):
    sequence_id: str = Field(default="default-sequence")
    intervention: InterventionInput
    horizon: int = Field(default=12, ge=1, le=12)
    state: dict[str, Any] | None = None
    current_timestamp: float | None = Field(
        default=None,
        description="Unix timestamp identifying the newest historical state.",
    )


class TrajectoryPoint(BaseModel):
    horizon: int
    state: dict[str, Any]
    attack_probability: float | None = None
    attack_category: str | None = None
    category_confidence: float | None = None


class SimulationPayload(BaseModel):
    baseline: dict[str, Any]
    counterfactual: dict[str, Any]
    comparison: dict[str, Any]
    explanation: str


class DefenseComparisonEntry(BaseModel):
    intervention: str
    target: int | str
    baseline_risk: float
    counterfactual_risk: float
    risk_reduction: float
    risk_reduction_percentage: float
    operational_cost: float
    service_disruption_cost: float
    defense_score: float
    peak_risk: float
    final_risk: float

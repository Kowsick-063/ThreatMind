from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


LIFECYCLE_STAGES = (
    "DETECTED",
    "ASSESSED",
    "FORECASTED",
    "SIMULATED",
    "RECOMMENDED",
)

STATUS_ORDER = {
    status: index
    for index, status in enumerate(LIFECYCLE_STAGES)
}

VALID_TRANSITIONS = {
    "DETECTED": "ASSESSED",
    "ASSESSED": "FORECASTED",
    "FORECASTED": "SIMULATED",
    "SIMULATED": "RECOMMENDED",
    "RECOMMENDED": None,
}


class InvalidLifecycleTransitionError(ValueError):
    """Raised when an incident lifecycle transition is invalid."""


def generate_incident_id(timestamp: float) -> str:
    """Generate a deterministic incident ID from a Unix timestamp."""
    return f"INC-{int(timestamp)}"


@dataclass
class Incident:
    incident_id: str
    timestamp: float
    status: str
    created_at: str

    risk_level: str
    peak_attack_probability: float
    peak_horizon: Optional[int]
    attack_category: Optional[str]

    recommended_intervention: Optional[Dict[str, Any]] = None
    decision_basis: Dict[str, Any] = field(default_factory=dict)

    mitre_techniques: List[Dict[str, Any]] = field(
        default_factory=list
    )

    uncertainties: List[Any] = field(
        default_factory=list
    )

    evidence: Dict[str, Any] = field(
        default_factory=dict
    )

    lifecycle_history: List[Dict[str, Any]] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_soc_analysis(
        cls,
        analysis: Dict[str, Any],
        initial_status: str = "DETECTED",
    ) -> "Incident":

        timestamp = float(analysis["timestamp"])

        if initial_status not in LIFECYCLE_STAGES:
            raise ValueError(
                f"Invalid initial status: {initial_status}"
            )

        risk = analysis.get("risk", {})
        forecast = analysis.get("forecast", {})
        graph = analysis.get("graph", {})
        mitre = analysis.get("mitre", [])
        counterfactuals = analysis.get(
            "counterfactuals",
            [],
        )
        decision = analysis.get(
            "decision",
            {},
        )

        peak_probability = float(
            risk.get(
                "peak_probability",
                forecast.get(
                    "peak_probability",
                    0.0,
                ),
            )
        )

        peak_horizon = risk.get(
            "peak_horizon",
            forecast.get(
                "peak_horizon",
            ),
        )

        attack_category = None

        horizons = forecast.get(
            "horizons",
            [],
        )

        if horizons:
            first_horizon = horizons[0]

            if isinstance(first_horizon, dict):
                attack_category = first_horizon.get(
                    "attack_category"
                )

        recommended_intervention = None

        if isinstance(decision, dict):
            recommended_intervention = decision.get(
                "recommended_intervention"
            )

        decision_basis = (
            decision.get(
                "decision_basis",
                {},
            )
            if isinstance(decision, dict)
            else {}
        )

        now = datetime.now(
            timezone.utc
        ).isoformat()

        evidence = {
            "observed": {
                "timestamp": timestamp,
                "current_state": analysis.get(
                    "current_state",
                    {},
                ),
                "graph_available": bool(
                    graph.get(
                        "available",
                        False,
                    )
                ),
                "graph": graph,
            },

            "predicted": {
                "forecast_horizons": horizons,
                "peak_probability": peak_probability,
                "peak_horizon": peak_horizon,
                "mitre_evidence": mitre,
            },

            "simulated": {
                "counterfactuals": counterfactuals,
            },

            "recommended": {
                "selected_intervention":
                    recommended_intervention,
                "decision_basis":
                    decision_basis,
            },
        }

        lifecycle_history = [
            {
                "status": initial_status,
                "timestamp": now,
            }
        ]

        return cls(
            incident_id=generate_incident_id(timestamp),
            timestamp=timestamp,
            status=initial_status,
            created_at=now,

            risk_level=risk.get(
                "risk_level",
                "UNKNOWN",
            ),

            peak_attack_probability=peak_probability,

            peak_horizon=peak_horizon,

            attack_category=attack_category,

            recommended_intervention=
                recommended_intervention,

            decision_basis=decision_basis,

            mitre_techniques=(
                mitre
                if isinstance(mitre, list)
                else []
            ),

            uncertainties=(
                analysis.get(
                    "uncertainties",
                    risk.get(
                        "uncertainties",
                        [],
                    ),
                )
            ),

            evidence=evidence,

            lifecycle_history=lifecycle_history,
        )


__all__ = [
    "Incident",
    "LIFECYCLE_STAGES",
    "STATUS_ORDER",
    "VALID_TRANSITIONS",
    "InvalidLifecycleTransitionError",
    "generate_incident_id",
]
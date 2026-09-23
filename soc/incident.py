"""SOC Incident model and lifecycle transitions for ThreatMind.

Lifecycle stages:
  DETECTED → ASSESSED → FORECASTED → SIMULATED → RECOMMENDED (terminal)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


LIFECYCLE_STAGES: list[str] = [
    "DETECTED",
    "ASSESSED",
    "FORECASTED",
    "SIMULATED",
    "RECOMMENDED",
]

# Strict single-step forward transitions map
VALID_TRANSITIONS: dict[str, str | None] = {
    "DETECTED": "ASSESSED",
    "ASSESSED": "FORECASTED",
    "FORECASTED": "SIMULATED",
    "SIMULATED": "RECOMMENDED",
    "RECOMMENDED": None,  # Terminal
}


class InvalidLifecycleTransitionError(ValueError):
    """Raised when an invalid lifecycle status transition is attempted."""

    def __init__(self, current_status: str, requested_status: str):
        self.current_status = current_status
        self.requested_status = requested_status
        expected_next = VALID_TRANSITIONS.get(current_status)
        if expected_next is None:
            msg = (
                f"Cannot transition incident from terminal status '{current_status}' "
                f"to '{requested_status}'."
            )
        else:
            msg = (
                f"Invalid lifecycle transition from '{current_status}' to '{requested_status}'. "
                f"Valid next transition is '{expected_next}'."
            )
        super().__init__(msg)


@dataclass
class Incident:
    """In-memory SOC incident representation."""

    incident_id: str
    timestamp: float
    status: str
    created_at: str
    risk_level: str
    peak_attack_probability: float
    peak_horizon: int
    attack_category: str
    mitre_techniques: list[dict[str, Any]]
    recommended_intervention: dict[str, Any]
    decision_basis: dict[str, Any]
    uncertainties: list[str]
    evidence: dict[str, Any] = field(default_factory=dict)
    lifecycle_history: list[dict[str, Any]] = field(default_factory=list)

    def transition_to(self, new_status: str) -> None:
        """Transition incident to a new lifecycle status following strict forward rules."""
        target_status = new_status.upper().strip()
        if target_status not in LIFECYCLE_STAGES:
            raise ValueError(
                f"Unknown status: '{new_status}'. Allowed statuses: {', '.join(LIFECYCLE_STAGES)}."
            )

        valid_next = VALID_TRANSITIONS.get(self.status)
        if valid_next is None:
            raise InvalidLifecycleTransitionError(self.status, target_status)

        if target_status != valid_next:
            raise InvalidLifecycleTransitionError(self.status, target_status)

        now = datetime.now(timezone.utc).isoformat()
        self.status = target_status
        self.lifecycle_history.append({
            "status": target_status,
            "transitioned_at": now,
        })

    def to_dict(self) -> dict[str, Any]:
        """Convert incident to a JSON-serializable dictionary."""
        return asdict(self)


def generate_incident_id(timestamp: float | int) -> str:
    """Generate a deterministic incident identifier from the timestamp."""
    return f"INC-{int(timestamp)}"


__all__ = [
    "LIFECYCLE_STAGES",
    "VALID_TRANSITIONS",
    "Incident",
    "InvalidLifecycleTransitionError",
    "generate_incident_id",
]

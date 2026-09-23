from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InterventionSpec:
    """Simulation-only hypothetical intervention definition."""

    intervention_id: str
    type: str
    target: int | str
    operational_cost: float = 0.0
    service_disruption_cost: float = 0.0
    description: str = ""


SUPPORTED_INTERVENTIONS = {
    "NO_ACTION": InterventionSpec(
        intervention_id="no_action",
        type="NO_ACTION",
        target=0,
        operational_cost=0.0,
        service_disruption_cost=0.0,
        description="No intervention applied; baseline future trajectory is preserved.",
    ),
    "BLOCK_SOURCE": InterventionSpec(
        intervention_id="block_source",
        type="BLOCK_SOURCE",
        target=0,
        operational_cost=0.15,
        service_disruption_cost=0.1,
        description="Simulate reducing traffic originating from the selected source.",
    ),
    "BLOCK_DESTINATION": InterventionSpec(
        intervention_id="block_destination",
        type="BLOCK_DESTINATION",
        target=0,
        operational_cost=0.18,
        service_disruption_cost=0.12,
        description="Simulate reducing traffic directed toward the selected destination.",
    ),
    "BLOCK_PORT": InterventionSpec(
        intervention_id="block_port",
        type="BLOCK_PORT",
        target=0,
        operational_cost=0.2,
        service_disruption_cost=0.3,
        description="Simulate blocking traffic associated with the selected destination port.",
    ),
    "ISOLATE_HOST": InterventionSpec(
        intervention_id="isolate_host",
        type="ISOLATE_HOST",
        target=0,
        operational_cost=0.35,
        service_disruption_cost=0.5,
        description="Simulate substantially reducing communication involving the selected host.",
    ),
    "DISABLE_CONNECTION": InterventionSpec(
        intervention_id="disable_connection",
        type="DISABLE_CONNECTION",
        target=0,
        operational_cost=0.22,
        service_disruption_cost=0.25,
        description="Simulate suppression of the selected communication relationship.",
    ),
    "NETWORK_SEGMENTATION": InterventionSpec(
        intervention_id="network_segmentation",
        type="NETWORK_SEGMENTATION",
        target=0,
        operational_cost=0.45,
        service_disruption_cost=0.6,
        description="Simulate restricting cross-segment communication between network regions.",
    ),
}


NO_ACTION = SUPPORTED_INTERVENTIONS["NO_ACTION"]


def build_intervention(
    intervention_type: str,
    target: int | str = 0,
    operational_cost: float | None = None,
    service_disruption_cost: float | None = None,
    description: str | None = None,
) -> InterventionSpec:
    """Return a deterministic intervention specification for simulation use only."""

    normalized = intervention_type.upper()
    if normalized not in SUPPORTED_INTERVENTIONS:
        allowed = ", ".join(sorted(SUPPORTED_INTERVENTIONS))
        raise ValueError(f"Unsupported intervention type: {intervention_type}. Allowed: {allowed}")

    template = SUPPORTED_INTERVENTIONS[normalized]
    effective_description = description or template.description
    effective_operational_cost = (
        template.operational_cost
        if operational_cost is None
        else float(operational_cost)
    )
    effective_service_disruption_cost = (
        template.service_disruption_cost
        if service_disruption_cost is None
        else float(service_disruption_cost)
    )

    return InterventionSpec(
        intervention_id=f"{template.intervention_id}_{target}".strip("_")
        if str(target) not in {"", "0"} or normalized == "NO_ACTION"
        else template.intervention_id,
        type=normalized,
        target=target,
        operational_cost=effective_operational_cost,
        service_disruption_cost=effective_service_disruption_cost,
        description=effective_description,
    )


__all__ = [
    "InterventionSpec",
    "SUPPORTED_INTERVENTIONS",
    "NO_ACTION",
    "build_intervention",
]

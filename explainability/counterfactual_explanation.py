from __future__ import annotations

from typing import Any


def explain_counterfactual(baseline: list[dict[str, Any]], counterfactual: list[dict[str, Any]], intervention: dict[str, Any]) -> str:
    if not baseline or not counterfactual:
        return "No trajectory data was available for explanation."

    baseline_risk = sum(point.get("risk", 0.0) for point in baseline)
    counterfactual_risk = sum(point.get("risk", 0.0) for point in counterfactual)
    max_delta_horizon = max(
        range(len(counterfactual)),
        key=lambda i: abs(counterfactual[i].get("risk", 0.0) - baseline[i].get("risk", 0.0)),
        default=0,
    ) + 1

    return (
        f"The simulated {intervention.get('type', 'INTERVENTION')} intervention reduced cumulative risk "
        f"from {baseline_risk:.4f} to {counterfactual_risk:.4f}. The largest trajectory difference occurred at horizon "
        f"t+{max_delta_horizon}. The intervention incurred an operational cost of "
        f"{intervention.get('operational_cost', 0.0):.3f} and service disruption cost of "
        f"{intervention.get('service_disruption_cost', 0.0):.3f}."
    )


__all__ = ["explain_counterfactual"]

from __future__ import annotations

from typing import Any


def rank_interventions(simulation_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        simulation_results,
        key=lambda item: item.get("defense_score", float("-inf")),
        reverse=True,
    )
    return [
        {
            "intervention": item["intervention"],
            "target": item["target"],
            "risk_reduction": item["risk_reduction"],
            "operational_cost": item["operational_cost"],
            "service_disruption_cost": item["service_disruption_cost"],
            "defense_score": item["defense_score"],
        }
        for item in ranked
    ]


__all__ = ["rank_interventions"]

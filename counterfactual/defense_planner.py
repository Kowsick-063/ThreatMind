from __future__ import annotations

from typing import Any


def rank_interventions(simulation_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(
        simulation_results,
        key=lambda item: item["summary"]["defense_score"],
        reverse=True,
    )
    return [
        {
            "intervention": item["intervention"],
            "target": item["intervention"]["target"],
            "risk_reduction": item["summary"]["risk_reduction"],
            "operational_cost": item["summary"]["operational_cost"],
            "service_disruption_cost": item["summary"]["service_disruption_cost"],
            "defense_score": item["summary"]["defense_score"],
        }
        for item in ranked
    ]


__all__ = ["rank_interventions"]


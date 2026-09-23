from __future__ import annotations

from typing import Any


def compare_trajectories(baseline_trajectory: list[dict[str, Any]], counterfactual_trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    max_length = max(len(baseline_trajectory), len(counterfactual_trajectory))
    for horizon in range(1, max_length + 1):
        baseline_point = baseline_trajectory[horizon - 1] if horizon - 1 < len(baseline_trajectory) else {}
        counterfactual_point = counterfactual_trajectory[horizon - 1] if horizon - 1 < len(counterfactual_trajectory) else {}
        rows.append(
            {
                "horizon": horizon,
                "baseline_attack_probability": baseline_point.get("attack_probability"),
                "counterfactual_attack_probability": counterfactual_point.get("attack_probability"),
                "baseline_risk": baseline_point.get("risk"),
                "counterfactual_risk": counterfactual_point.get("risk"),
                "baseline_category": baseline_point.get("attack_category"),
                "counterfactual_category": counterfactual_point.get("attack_category"),
            }
        )
    return rows


__all__ = ["compare_trajectories"]

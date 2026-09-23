from __future__ import annotations

from typing import Any

from counterfactual.interventions import SUPPORTED_INTERVENTIONS, build_intervention
from counterfactual.rollout import simulate_counterfactual


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _snapshot_intervention(intervention: Any) -> dict[str, Any]:
    if hasattr(intervention, "type"):
        return {
            "id": getattr(intervention, "intervention_id", getattr(intervention, "type", "unknown")),
            "type": getattr(intervention, "type", "UNKNOWN"),
            "target": getattr(intervention, "target", 0),
            "operational_cost": _safe_float(getattr(intervention, "operational_cost", 0.0)),
            "service_disruption_cost": _safe_float(getattr(intervention, "service_disruption_cost", 0.0)),
        }
    if isinstance(intervention, dict):
        return {
            "id": intervention.get("id") or intervention.get("intervention_id") or intervention.get("type") or "unknown",
            "type": intervention.get("type") or "UNKNOWN",
            "target": intervention.get("target", 0),
            "operational_cost": _safe_float(intervention.get("operational_cost", 0.0)),
            "service_disruption_cost": _safe_float(intervention.get("service_disruption_cost", 0.0)),
        }
    return {
        "id": "unknown",
        "type": str(intervention),
        "target": 0,
        "operational_cost": 0.0,
        "service_disruption_cost": 0.0,
    }


def _as_trajectory(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, list):
        if not value:
            return []
        if isinstance(value[0], dict):
            return value
        if isinstance(value[0], list):
            return value[0]
    return []


def _trajectory_risks(trajectory: list[dict[str, Any]]) -> list[float]:
    return [_safe_float(point.get("risk", 0.0)) for point in trajectory if isinstance(point, dict)]


def _risk_reduction(baseline_risk: float, counterfactual_risk: float) -> float:
    return baseline_risk - counterfactual_risk


def _risk_reduction_percentage(baseline_risk: float, counterfactual_risk: float) -> float:
    reduction = _risk_reduction(baseline_risk, counterfactual_risk)
    if baseline_risk == 0:
        return 0.0
    return (reduction / baseline_risk) * 100.0


def _trajectory_summary(trajectory: list[dict[str, Any]]) -> dict[str, float]:
    risks = _trajectory_risks(trajectory)
    if not risks:
        return {
            "cumulative_risk": 0.0,
            "average_risk": 0.0,
            "peak_risk": 0.0,
            "final_risk": 0.0,
        }
    return {
        "cumulative_risk": sum(risks),
        "average_risk": sum(risks) / len(risks),
        "peak_risk": max(risks),
        "final_risk": risks[-1],
    }


def _compare_horizon_points(baseline_point: dict[str, Any], counterfactual_point: dict[str, Any]) -> dict[str, Any]:
    baseline_risk = _safe_float(baseline_point.get("risk", 0.0))
    counterfactual_risk = _safe_float(counterfactual_point.get("risk", 0.0))
    risk_reduction = _risk_reduction(baseline_risk, counterfactual_risk)
    return {
        "baseline_risk": baseline_risk,
        "counterfactual_risk": counterfactual_risk,
        "risk_reduction": risk_reduction,
        "baseline_attack_probability": _safe_float(baseline_point.get("attack_probability"), 0.0),
        "counterfactual_attack_probability": _safe_float(counterfactual_point.get("attack_probability"), 0.0),
        "baseline_category": baseline_point.get("attack_category"),
        "counterfactual_category": counterfactual_point.get("attack_category"),
        "baseline_category_confidence": _safe_float(baseline_point.get("category_confidence"), 0.0),
        "counterfactual_category_confidence": _safe_float(counterfactual_point.get("category_confidence"), 0.0),
    }


def _compare_single_trajectory(
    baseline_trajectory: list[dict[str, Any]],
    counterfactual_trajectory: list[dict[str, Any]],
    intervention: Any,
) -> dict[str, Any]:
    baseline = _as_trajectory(baseline_trajectory)
    counterfactual = _as_trajectory(counterfactual_trajectory)

    intervention_data = _snapshot_intervention(intervention)
    baseline_summary = _trajectory_summary(baseline)
    counterfactual_summary = _trajectory_summary(counterfactual)
    baseline_total = baseline_summary["cumulative_risk"]
    counterfactual_total = counterfactual_summary["cumulative_risk"]
    risk_reduction = _risk_reduction(baseline_total, counterfactual_total)
    risk_reduction_percentage = _risk_reduction_percentage(baseline_total, counterfactual_total)
    defense_score = risk_reduction - intervention_data["operational_cost"] - intervention_data["service_disruption_cost"]

    trajectory_rows = []
    max_length = max(len(baseline), len(counterfactual))
    for index in range(max_length):
        baseline_point = baseline[index] if index < len(baseline) else {}
        counterfactual_point = counterfactual[index] if index < len(counterfactual) else {}
        trajectory_rows.append({
            "horizon": index + 1,
            **_compare_horizon_points(baseline_point, counterfactual_point),
        })

    return {
        "intervention": {
            "id": intervention_data["id"],
            "type": intervention_data["type"],
            "target": intervention_data["target"],
        },
        "summary": {
            "baseline_cumulative_risk": round(baseline_total, 6),
            "counterfactual_cumulative_risk": round(counterfactual_total, 6),
            "risk_reduction": round(risk_reduction, 6),
            "risk_reduction_percentage": round(risk_reduction_percentage, 10),
            "baseline_average_risk": round(baseline_summary["average_risk"], 6),
            "counterfactual_average_risk": round(counterfactual_summary["average_risk"], 6),
            "baseline_peak_risk": round(baseline_summary["peak_risk"], 6),
            "counterfactual_peak_risk": round(counterfactual_summary["peak_risk"], 6),
            "baseline_final_risk": round(baseline_summary["final_risk"], 6),
            "counterfactual_final_risk": round(counterfactual_summary["final_risk"], 6),
            "operational_cost": round(intervention_data["operational_cost"], 6),
            "service_disruption_cost": round(intervention_data["service_disruption_cost"], 6),
            "defense_score": round(defense_score, 6),
        },
        "trajectory": trajectory_rows,
    }


def compare_interventions(
    baseline_trajectory: Any = None,
    counterfactual_trajectories: Any = None,
    interventions: Any = None,
    *,
    initial_state: Any = None,
    intervention_types: list[str] | None = None,
    forecasting_model=None,
    scaler=None,
    horizon: int = 12,
) -> list[dict[str, Any]] | dict[str, Any]:
    """Direct trajectory comparison or legacy simulation-backed comparison."""
    if initial_state is not None and intervention_types is not None:
        results = []
        for name in intervention_types:
            intervention = build_intervention(name)
            simulation = simulate_counterfactual(
                initial_state=initial_state,
                intervention=intervention,
                forecasting_model=forecasting_model,
                scaler=scaler,
                horizon=horizon,
            )
            results.append(
                _compare_single_trajectory(simulation["baseline"], simulation["counterfactual"], intervention)
            )
        return results

    if baseline_trajectory is None:
        return []

    if not isinstance(baseline_trajectory, list):
        return []

    if intervention_types is not None and interventions is None:
        interventions = list(intervention_types)

    if counterfactual_trajectories is None:
        counterfactual_trajectories = []

    if isinstance(counterfactual_trajectories, list) and counterfactual_trajectories and isinstance(counterfactual_trajectories[0], list):
        return _compare_single_trajectory(baseline_trajectory, counterfactual_trajectories[0], interventions or {"type": "NO_ACTION", "target": 0})

    return _compare_single_trajectory(baseline_trajectory, counterfactual_trajectories, interventions or {"type": "NO_ACTION", "target": 0})


def compare_all_interventions(
    baseline_trajectory: list[dict[str, Any]],
    counterfactual_trajectories: list[list[dict[str, Any]]] | list[dict[str, Any]],
    interventions: list[Any],
) -> list[dict[str, Any]]:
    """Compare one baseline against multiple intervention trajectories."""
    if counterfactual_trajectories is None:
        return []
    if not isinstance(counterfactual_trajectories, list):
        counterfactual_trajectories = [counterfactual_trajectories]
    if not isinstance(interventions, list):
        interventions = [interventions]
    if len(counterfactual_trajectories) != len(interventions):
        raise ValueError("counterfactual_trajectories and interventions must have the same length.")

    results = []
    for index, counterfactual_trajectory in enumerate(counterfactual_trajectories):
        results.append(_compare_single_trajectory(baseline_trajectory, counterfactual_trajectory, interventions[index]))
    return results


__all__ = ["compare_interventions", "compare_all_interventions"]

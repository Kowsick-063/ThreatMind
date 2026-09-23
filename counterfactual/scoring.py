from __future__ import annotations

from typing import Any

RISK_WEIGHTS = {
    "attack_probability": 0.55,
    "traffic_intensity": 0.25,
    "anomaly_component": 0.15,
    "category_confidence": 0.05,
}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _traffic_intensity(state: dict[str, Any] | None) -> float:
    if not state:
        return 0.0
    traffic = [
        state.get("flow_count", 0),
        state.get("total_packet_count", 0),
        state.get("total_byte_count", 0),
        state.get("mean_flow_packets_per_sec", 0),
        state.get("mean_flow_bytes_per_sec", 0),
    ]
    normalized = sum(_safe_float(value) for value in traffic)
    return min(normalized / 10000.0, 1.0)


def _anomaly_component(state: dict[str, Any] | None, probability: float) -> float:
    if not state:
        return max(probability - 0.5, 0.0)
    anomaly = _safe_float(state.get("anomaly_score", 0.0), 0.0)
    return min(max(anomaly, probability * 0.5), 1.0)


def score_risk(
    state: dict[str, Any] | None = None,
    attack_probability: float | None = None,
    category_confidence: float | None = None,
    attack_flow_ratio: float | None = None,
) -> float:
    """Compute a deterministic risk score from available forecast attributes."""
    probability = min(max(_safe_float(attack_probability, 0.0), 0.0), 1.0)
    traffic = _traffic_intensity(state)
    anomaly = _anomaly_component(state, probability)
    confidence = min(max(_safe_float(category_confidence, 0.0), 0.0), 1.0)
    attack_ratio = min(max(_safe_float(attack_flow_ratio, 0.0), 0.0), 1.0)

    weighted = (
        RISK_WEIGHTS["attack_probability"] * probability
        + RISK_WEIGHTS["traffic_intensity"] * traffic
        + RISK_WEIGHTS["anomaly_component"] * anomaly
        + RISK_WEIGHTS["category_confidence"] * max(confidence, attack_ratio)
    )

    return round(min(max(weighted, 0.0), 1.0), 6)


def compute_trajectory_risk(trajectory: list[dict[str, Any]]) -> dict[str, Any]:
    risks = []
    for point in trajectory:
        state = point.get("state") if isinstance(point.get("state"), dict) else {}
        risk = score_risk(
            state=state,
            attack_probability=point.get("attack_probability"),
            category_confidence=point.get("category_confidence"),
        )
        risks.append(risk)
        point["risk"] = risk

    if not risks:
        return {
            "cumulative_risk": 0.0,
            "peak_risk": 0.0,
            "average_risk": 0.0,
            "final_risk": 0.0,
        }

    cumulative_risk = sum(risks)
    return {
        "cumulative_risk": round(cumulative_risk, 6),
        "peak_risk": round(max(risks), 6),
        "average_risk": round(cumulative_risk / len(risks), 6),
        "final_risk": round(risks[-1], 6),
    }


def compare_risk(baseline_risk: float, counterfactual_risk: float) -> dict[str, float]:
    risk_reduction = baseline_risk - counterfactual_risk
    risk_reduction_percentage = 0.0
    if baseline_risk > 0:
        risk_reduction_percentage = (risk_reduction / baseline_risk) * 100.0
    return {
        "risk_reduction": round(risk_reduction, 6),
        "risk_reduction_percentage": round(risk_reduction_percentage, 6),
    }


__all__ = ["RISK_WEIGHTS", "score_risk", "compute_trajectory_risk", "compare_risk"]

"""Risk analyzer for ThreatMind attack trajectories.

Consumes the calibrated H1-H12 attack trajectory and produces neutral,
quantifiable risk assessments using an explicit elevated-risk threshold.
"""

from __future__ import annotations

from typing import Any


# Explicit elevated-risk threshold (configurable)
ELEVATED_RISK_THRESHOLD: float = 0.50
MODERATE_RISK_THRESHOLD: float = 0.30


def analyze_trajectory_risk(
    trajectory_points: list[dict[str, Any]],
    elevated_threshold: float = ELEVATED_RISK_THRESHOLD,
) -> dict[str, Any]:
    """Analyze risk across an H1-H12 forecast trajectory.

    Parameters
    ----------
    trajectory_points:
        List of 12 horizon point dictionaries containing 'predicted_attack_probability'
        and 'horizon'.
    elevated_threshold:
        Threshold above which a horizon is considered elevated risk.

    Returns
    -------
    dict with risk_level, peak_probability, peak_horizon, elevated_horizons,
    trend, initial_probability, final_probability, and uncertainties.
    """
    if not trajectory_points:
        return {
            "risk_level": "LOW",
            "peak_probability": 0.0,
            "peak_horizon": 1,
            "elevated_horizons": [],
            "number_of_elevated_risk_horizons": 0,
            "trend": "stable",
            "initial_attack_probability": 0.0,
            "final_attack_probability": 0.0,
            "peak_attack_probability": 0.0,
            "uncertainties": ["Trajectory contains no points."],
        }

    probs: list[float] = [
        float(p.get("predicted_attack_probability", 0.0)) for p in trajectory_points
    ]

    initial_prob = probs[0]
    final_prob = probs[-1]
    peak_prob = max(probs)
    peak_horizon = probs.index(peak_prob) + 1

    # Trend calculation
    diff = final_prob - initial_prob
    if diff > 0.05:
        trend = "increasing"
    elif diff < -0.05:
        trend = "decreasing"
    else:
        trend = "stable"

    # Identify horizons exceeding elevated risk threshold
    elevated_horizons = [
        int(p.get("horizon", idx + 1))
        for idx, p in enumerate(trajectory_points)
        if float(p.get("predicted_attack_probability", 0.0)) >= elevated_threshold
    ]

    # Neutral risk classification
    if peak_prob >= elevated_threshold or len(elevated_horizons) >= 3:
        risk_level = "HIGH"
    elif peak_prob >= MODERATE_RISK_THRESHOLD or len(elevated_horizons) > 0:
        risk_level = "MODERATE"
    else:
        risk_level = "LOW"

    uncertainties: list[str] = [
        f"Elevated-risk classification uses explicit threshold of {elevated_threshold:.2f}.",
        "Probabilities represent statistical risk forecasts and do not assert guaranteed attack occurrence.",
    ]

    return {
        "risk_level": risk_level,
        "peak_probability": round(peak_prob, 6),
        "peak_attack_probability": round(peak_prob, 6),
        "peak_horizon": peak_horizon,
        "initial_attack_probability": round(initial_prob, 6),
        "final_attack_probability": round(final_prob, 6),
        "probability_trend": trend,
        "trend": trend,
        "elevated_horizons": elevated_horizons,
        "number_of_elevated_risk_horizons": len(elevated_horizons),
        "elevated_risk_threshold": elevated_threshold,
        "uncertainties": uncertainties,
    }


__all__ = [
    "ELEVATED_RISK_THRESHOLD",
    "MODERATE_RISK_THRESHOLD",
    "analyze_trajectory_risk",
]

"""Structured SOC response builder for ThreatMind.

Assembles the unified SOC response schema incorporating:
  - Current network state
  - Risk assessment
  - Multi-step forecast (H1-H12)
  - MITRE ATT&CK evidence
  - Real graph topology metrics
  - Counterfactual intervention evaluations
  - Defense decision and explanation
"""

from __future__ import annotations

from typing import Any


def build_soc_response(
    timestamp: float | int,
    current_state: dict[str, float],
    graph: dict[str, Any] | None,
    trajectory: list[dict[str, Any]],
    risk: dict[str, Any],
    decision: dict[str, Any],
    mitre: list[dict[str, Any]] | None = None,
    counterfactuals: list[dict[str, Any]] | None = None,
    explanation: dict[str, Any] | None = None,
    uncertainties: list[str] | None = None,
) -> dict[str, Any]:
    """Build the canonical ThreatMind SOC response object.

    Parameters
    ----------
    timestamp:
        Unix timestamp of analyzed network state.
    current_state:
        26-feature dictionary of current network flow aggregates.
    graph:
        Graph snapshot summary dict or None if unavailable.
    trajectory:
        List of 12 horizon forecast points (H1..H12).
    risk:
        Risk analysis output from analyze_trajectory_risk.
    decision:
        Decision output from generate_defense_decision (contains recommended_intervention,
        decision_basis, alternatives, reasons, near_ties).
    mitre:
        Optional list of MITRE techniques observed across the trajectory.
    counterfactuals:
        Optional list of 7 evaluated counterfactual interventions.
    explanation:
        Optional structured SOC explanation dict.
    uncertainties:
        Optional aggregated uncertainty statements.

    Returns
    -------
    dict matching the exact SOC response schema.
    """
    # Peak metrics for forecast block
    peak_h = risk.get("peak_horizon", 1)
    peak_p = risk.get("peak_probability", 0.0)

    forecast_block = {
        "horizons": trajectory,
        "peak_horizon": peak_h,
        "peak_probability": peak_p,
        "probability_trend": risk.get("trend", "stable"),
    }

    # Graph block with explicit availability reporting
    if graph is not None and graph.get("available", True):
        graph_block = {
            "available": True,
            "timestamp": graph.get("timestamp", timestamp),
            "nodes": graph.get("nodes", 0),
            "edges": graph.get("edges", 0),
            "attack_surface_score": graph.get("attack_surface_score", 0.0),
            "unique_sources": graph.get("unique_sources", 0),
            "unique_destinations": graph.get("unique_destinations", 0),
            "unique_services": graph.get("unique_services", 0),
        }
    else:
        graph_block = {
            "available": False,
        }

    # Combine uncertainties without duplicates
    combined_uncertainties: list[str] = []
    for source in [uncertainties, risk.get("uncertainties"), decision.get("uncertainties")]:
        if source:
            for u in source:
                if u not in combined_uncertainties:
                    combined_uncertainties.append(u)

    if not graph_block["available"]:
        missing_graph_note = (
            "Graph topology data was unavailable at this timestamp; "
            "evaluations are grounded in 26-feature state forecasting."
        )
        if missing_graph_note not in combined_uncertainties:
            combined_uncertainties.append(missing_graph_note)

    return {
        "timestamp": timestamp,
        "status": "ANALYZED",
        "current_state": current_state,
        "risk": risk,
        "forecast": forecast_block,
        "mitre": mitre or [],
        "graph": graph_block,
        "counterfactuals": counterfactuals or [],
        "decision": decision,
        "explanation": explanation or {},
        "uncertainties": combined_uncertainties,
    }


__all__ = ["build_soc_response"]

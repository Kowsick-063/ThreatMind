"""Real-Time SOC Orchestrator for ThreatMind.

Orchestrates the complete pipeline:
  timestamp
  → historical network state
  → real graph snapshot
  → H1-H12 trajectory forecast
  → risk analysis
  → MITRE ATT&CK evidence
  → counterfactual simulation
  → defense decision
  → structured SOC response
"""

from __future__ import annotations

from typing import Any

from counterfactual.graph_counterfactual import compute_attack_surface
from counterfactual.graph_simulator import get_graph_snapshot_by_timestamp
from counterfactual.rollout import get_historical_sequence
from decision.defense_decision import generate_defense_decision
from decision.risk_analyzer import analyze_trajectory_risk
from forecasting.trajectory import generate_attack_trajectory
from soc.soc_explanation import generate_soc_explanation
from soc.soc_response import build_soc_response


def run_soc_analysis(timestamp: float | int) -> dict[str, Any]:
    """Execute the end-to-end SOC analysis and defense recommendation pipeline.

    Parameters
    ----------
    timestamp:
        Unix timestamp to evaluate. Must exist in the historical dataset with
        a contiguous 60-state history.

    Returns
    -------
    dict conforming to the ThreatMind SOC response schema.
    """
    numeric_ts = float(timestamp)

    # 1. Retrieve historical 60-state sequence (validates continuity)
    # Raises ValueError explicitly if unavailable or if temporal gaps exist.
    _ = get_historical_sequence(numeric_ts)

    # 2. Graph snapshot retrieval
    snapshot = get_graph_snapshot_by_timestamp(numeric_ts)
    if snapshot is not None and len(snapshot.edge_index) > 0:
        attack_surface = compute_attack_surface(snapshot)
        graph_data: dict[str, Any] = {
            "available": True,
            "timestamp": snapshot.timestamp,
            "nodes": len(snapshot.node_ids),
            "edges": len(snapshot.edge_index),
            "attack_surface_score": attack_surface.get("attack_surface_score", 0.0),
            "unique_sources": len(set(snapshot.edge_index[:, 0].tolist())),
            "unique_destinations": len(set(snapshot.edge_index[:, 1].tolist())),
            "unique_services": attack_surface.get("unique_services", 0),
        }
    else:
        graph_data = {"available": False}

    # 3. Trajectory forecasting (H1-H12)
    trajectory_result = generate_attack_trajectory(current_timestamp=numeric_ts)
    current_state = trajectory_result.get("current_state", {})
    trajectory = trajectory_result.get("trajectory", [])
    mitre_evidence = trajectory_result.get("mitre_evidence", [])

    # 4. Risk analysis
    risk_analysis = analyze_trajectory_risk(trajectory)

    # 5. Defense decision & counterfactual evaluations
    decision_result = generate_defense_decision(
        timestamp=numeric_ts,
        trajectory_result=trajectory_result,
    )
    evaluated_interventions = decision_result.get("interventions", [])
    decision = decision_result.get("decision", {})

    # 6. SOC-tiered explanation
    soc_explanation = generate_soc_explanation(
        current_state=current_state,
        risk=risk_analysis,
        trajectory=trajectory,
        decision=decision,
        graph=graph_data if graph_data.get("available") else None,
        mitre=mitre_evidence,
    )

    # 7. Assemble structured SOC response
    return build_soc_response(
        timestamp=numeric_ts,
        current_state=current_state,
        graph=graph_data,
        trajectory=trajectory,
        risk=risk_analysis,
        decision=decision,
        mitre=mitre_evidence,
        counterfactuals=evaluated_interventions,
        explanation=soc_explanation,
        uncertainties=soc_explanation.get("uncertainties", []),
    )


__all__ = ["run_soc_analysis"]

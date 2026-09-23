"""Defense decision explanation and rationale synthesis for ThreatMind.

Explicitly distinguishes:
  - Observed (current network features, graph topology, active endpoints)
  - Predicted (calibrated attack probability, category trajectory H1-H12)
  - Simulated (counterfactual traffic reductions, graph edge and path removals)
  - Recommended (highest defense score intervention with trade-off analysis)
"""

from __future__ import annotations

from typing import Any


def generate_defense_explanation(
    recommended: dict[str, Any],
    decision_basis: dict[str, Any],
    risk_analysis: dict[str, Any],
    evaluated_interventions: list[dict[str, Any]],
    graph_impact: dict[str, Any] | None = None,
    graph_available: bool = False,
    near_ties: list[dict[str, Any]] | None = None,
    current_timestamp: float | int | str | None = None,
) -> dict[str, Any]:
    """Generate structured SOC decision explanation distinguishing evidence tiers.

    Parameters
    ----------
    recommended:
        Dict with 'type' and 'target' of the recommended intervention.
    decision_basis:
        Dict with defense_score, risk_reduction, graph_attack_surface_reduction, etc.
    risk_analysis:
        Output from analyze_trajectory_risk.
    evaluated_interventions:
        List of all evaluated intervention summary dicts.
    graph_impact:
        Optional graph impact dictionary from simulate_graph_counterfactual.
    graph_available:
        Whether real graph snapshot was available for evaluation.
    near_ties:
        Optional list of interventions near-tied with the top choice.
    current_timestamp:
        Optional initial timestamp.

    Returns
    -------
    dict with summary, risk_evidence, intervention_evidence, graph_evidence,
    tradeoffs, and uncertainties.
    """
    rec_type = recommended.get("type", "NO_ACTION")
    rec_target = recommended.get("target", 0)
    score = decision_basis.get("defense_score", 0.0)
    risk_red = decision_basis.get("risk_reduction", 0.0)
    risk_level = risk_analysis.get("risk_level", "LOW")
    peak_prob = risk_analysis.get("peak_probability", 0.0)
    peak_h = risk_analysis.get("peak_horizon", 1)
    trend = risk_analysis.get("trend", "stable")
    elevated_count = risk_analysis.get("number_of_elevated_risk_horizons", 0)

    # ------------------------------------------------------------------
    # 1. Risk Evidence (Observed & Predicted)
    # ------------------------------------------------------------------
    risk_evidence: list[str] = [
        f"Observed: Evaluated network state at timestamp {current_timestamp or 't=0'}."
        if current_timestamp
        else "Observed: Evaluated initial network state.",
        f"Predicted: Multi-step forecast indicates {risk_level} risk level with a {trend} trend.",
        f"Predicted: Peak attack probability reaches {peak_prob:.4f} at horizon H{peak_h}.",
        f"Predicted: {elevated_count} of 12 forecast horizons exceed the elevated risk threshold.",
    ]

    # ------------------------------------------------------------------
    # 2. Intervention Evidence (Simulated & Recommended)
    # ------------------------------------------------------------------
    intervention_evidence: list[str] = [
        f"Recommended: Intervention {rec_type} (target: {rec_target}) achieved the highest defense score ({score:.4f}).",
        f"Simulated: Counterfactual rollout projects a cumulative risk reduction of {risk_red:.4f}.",
    ]
    if rec_type == "NO_ACTION":
        intervention_evidence.append(
            "Recommended: NO_ACTION selected because baseline risk is low or active "
            "interventions impose operational/disruption costs exceeding risk reduction."
        )

    # ------------------------------------------------------------------
    # 3. Graph Evidence (Observed & Simulated)
    # ------------------------------------------------------------------
    graph_evidence: list[str] = []
    if graph_available and graph_impact is not None:
        edges_rem = graph_impact.get("edges_removed", 0)
        paths_rem = graph_impact.get("affected_attack_paths", 0)
        surf_pct = decision_basis.get("graph_attack_surface_reduction", 0.0)

        graph_evidence.append(
            f"Observed: Real network graph snapshot analyzed at timestamp {current_timestamp}."
        )
        graph_evidence.append(
            f"Simulated: {rec_type} removes {edges_rem} communication edge(s) and affects {paths_rem} path(s)."
        )
        graph_evidence.append(
            f"Simulated: Attack surface score reduced by {surf_pct:.2f}%."
        )
    else:
        graph_evidence.append(
            "Observed: Graph snapshot unavailable for this timestamp; recommendation based on 26-feature state counterfactual rollout."
        )

    # ------------------------------------------------------------------
    # 4. Trade-offs
    # ------------------------------------------------------------------
    op_cost = decision_basis.get("operational_cost", 0.0)
    disrup_cost = decision_basis.get("service_disruption_cost", 0.0)
    tradeoffs: list[str] = [
        f"Trade-off: Risk reduction ({risk_red:.4f}) vs operational cost ({op_cost:.2f}) and service disruption cost ({disrup_cost:.2f}).",
        f"Net defense score: {score:.4f} (defense_score = risk_reduction - operational_cost - disruption_cost).",
    ]
    if near_ties:
        for tie in near_ties:
            tie_type = tie.get("intervention", {}).get("type", "UNKNOWN")
            tie_score = tie.get("defense_score", 0.0)
            tradeoffs.append(
                f"Near-tie notice: {tie_type} has a defense score of {tie_score:.4f} "
                f"(difference < 0.01 from recommended {rec_type})."
            )

    # ------------------------------------------------------------------
    # 5. Uncertainties
    # ------------------------------------------------------------------
    uncertainties: list[str] = [
        "Counterfactual rollout models estimated defensive traffic shifts; real-world adversary adaptation may vary.",
        "Probabilities represent statistical risk forecasts and do not assert guaranteed attack occurrence.",
    ]
    if not graph_available:
        uncertainties.append(
            "Graph topology evidence was unavailable; graph attack surface metrics were not evaluated."
        )

    # ------------------------------------------------------------------
    # 6. Summary Narrative
    # ------------------------------------------------------------------
    if rec_type == "NO_ACTION":
        summary = (
            f"Risk analysis projects {risk_level} risk (peak probability: {peak_prob:.4f} at H{peak_h}). "
            f"NO_ACTION is recommended (defense score: {score:.4f}) as active interventions would incur "
            f"operational and disruption overhead without commensurate risk reduction."
        )
    else:
        summary = (
            f"Risk analysis projects {risk_level} risk (peak probability: {peak_prob:.4f} at H{peak_h}). "
            f"Recommended intervention is {rec_type} (target: {rec_target}) with defense score {score:.4f}, "
            f"providing {risk_red:.4f} risk reduction."
        )
        if graph_available and graph_impact is not None:
            summary += f" Graph simulation indicates removal of {graph_impact.get('edges_removed', 0)} edge(s)."

    return {
        "summary": summary,
        "risk_evidence": risk_evidence,
        "intervention_evidence": intervention_evidence,
        "graph_evidence": graph_evidence,
        "tradeoffs": tradeoffs,
        "uncertainties": uncertainties,
    }


__all__ = ["generate_defense_explanation"]

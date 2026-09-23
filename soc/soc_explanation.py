"""SOC explanation generator distinguishing evidence tiers.

Strictly separates:
  - OBSERVED: Current network state, observed graph topology, observed traffic rates
  - PREDICTED: H1-H12 attack probabilities, peak horizon, category trajectory, MITRE evidence
  - SIMULATED: Counterfactual intervention effects, risk reduction, graph shifts, operational costs
  - RECOMMENDED: Selected defensive action, defense score, rationale, alternatives
"""

from __future__ import annotations

from typing import Any


def generate_soc_explanation(
    current_state: dict[str, float],
    risk: dict[str, Any],
    trajectory: list[dict[str, Any]],
    decision: dict[str, Any],
    graph: dict[str, Any] | None = None,
    mitre: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate structured SOC explanation separating evidence tiers.

    Parameters
    ----------
    current_state:
        Observed 26 network flow aggregate features.
    risk:
        Risk analysis dict (risk_level, peak_probability, elevated_horizons, trend).
    trajectory:
        H1-H12 forecast trajectory list.
    decision:
        Decision dictionary (recommended_intervention, decision_basis, reasons, alternatives).
    graph:
        Optional graph snapshot summary.
    mitre:
        Optional list of MITRE techniques.

    Returns
    -------
    dict with observed, predicted, simulated, recommended, summary, and uncertainties.
    """
    rec = decision.get("recommended_intervention", {})
    rec_type = rec.get("type", "NO_ACTION")
    rec_target = rec.get("target", 0)
    basis = decision.get("decision_basis", {})
    score = basis.get("defense_score", 0.0)
    risk_red = basis.get("risk_reduction", 0.0)

    risk_level = risk.get("risk_level", "LOW")
    peak_p = risk.get("peak_probability", 0.0)
    peak_h = risk.get("peak_horizon", 1)
    trend = risk.get("trend", "stable")
    elevated_count = risk.get("number_of_elevated_risk_horizons", 0)

    # ------------------------------------------------------------------
    # 1. OBSERVED: Current state, graph topology, traffic rates
    # ------------------------------------------------------------------
    fc = current_state.get("flow_count", 0.0)
    pc = current_state.get("packet_count", 0.0)
    bc = current_state.get("byte_count", 0.0)
    up = current_state.get("unique_ports", 0.0)
    pps = current_state.get("packets_per_second", 0.0)
    bps = current_state.get("bytes_per_second", 0.0)

    observed: list[str] = [
        f"Flow and packet volume: {fc:.0f} flows, {pc:.0f} packets, {bc:.0f} bytes across {up:.0f} destination ports.",
        f"Throughput metrics: {pps:.1f} packets/second, {bps:.1f} bytes/second.",
    ]
    if graph and graph.get("available", True):
        nodes = graph.get("nodes", 0)
        edges = graph.get("edges", 0)
        surf = graph.get("attack_surface_score", 0.0)
        observed.append(
            f"Active graph topology: {nodes} communicating nodes, {edges} active edges, attack surface score: {surf:.2f}."
        )
    else:
        observed.append("Graph topology snapshot was unavailable for this timestamp.")

    # ------------------------------------------------------------------
    # 2. PREDICTED: H1-H12 probabilities, peak horizon, categories, MITRE
    # ------------------------------------------------------------------
    predicted: list[str] = [
        f"Forecast trajectory indicates {risk_level} risk level with {trend} probability trend.",
        f"Peak predicted attack probability reaches {peak_p:.4f} at horizon H{peak_h} (calibrated forecast, not confirmed event).",
        f"{elevated_count} of 12 forecast horizons exceed the elevated-risk threshold.",
    ]
    if trajectory:
        cats = [p.get("attack_category", "UNKNOWN") for p in trajectory]
        distinct_cats = []
        for c in cats:
            if not distinct_cats or distinct_cats[-1] != c:
                distinct_cats.append(c)
        predicted.append(f"Predicted attack category progression: {' -> '.join(distinct_cats)}.")

    if mitre:
        tech_summaries = [f"{t.get('technique_id')} ({t.get('technique_name', '')})" for t in mitre[:3]]
        predicted.append(f"MITRE ATT&CK techniques grounded in predicted state: {', '.join(tech_summaries)}.")

    # ------------------------------------------------------------------
    # 3. SIMULATED: Counterfactual intervention effects, costs, graph shifts
    # ------------------------------------------------------------------
    op_cost = basis.get("operational_cost", 0.0)
    disrup_cost = basis.get("service_disruption_cost", 0.0)
    edges_rem = basis.get("edges_removed", 0)
    surf_red = basis.get("graph_attack_surface_reduction", 0.0)

    simulated: list[str] = [
        f"Simulating {rec_type} (target: {rec_target}) yields cumulative risk reduction of {risk_red:.4f}.",
        f"Associated intervention costs: operational cost = {op_cost:.2f}, service disruption cost = {disrup_cost:.2f}.",
    ]
    if edges_rem > 0 or surf_red > 0.0:
        simulated.append(
            f"Graph counterfactual simulation removes {edges_rem} communication edge(s) and reduces attack surface by {surf_red:.2f}%."
        )

    # ------------------------------------------------------------------
    # 4. RECOMMENDED: Selected action, defense score, rationale, alternatives
    # ------------------------------------------------------------------
    recommended: list[str] = [
        f"Selected action: {rec_type} (target: {rec_target}) with highest defense score ({score:.4f}).",
        f"Defense score formula: defense_score = risk_reduction ({risk_red:.4f}) - operational_cost ({op_cost:.2f}) - disruption_cost ({disrup_cost:.2f}).",
    ]
    for reason in decision.get("reasons", []):
        if reason not in recommended:
            recommended.append(f"Rationale: {reason}")

    near_ties = decision.get("near_ties", [])
    if near_ties:
        tie_names = ", ".join(f"{t.get('type')} ({t.get('defense_score', 0.0):.4f})" for t in near_ties)
        recommended.append(f"Near-tie alternatives within 0.01 defense score: {tie_names}.")

    # Summary synthesis
    summary = (
        f"ThreatMind SOC evaluation identified {risk_level} risk (peak probability: {peak_p:.4f} at H{peak_h}). "
        f"Recommended defensive action is {rec_type} (target: {rec_target}) with defense score {score:.4f}, "
        f"providing {risk_red:.4f} projected risk reduction."
    )

    uncertainties: list[str] = [
        "Forecast probabilities represent statistical model estimates, not guaranteed real-world attack occurrences.",
        "Counterfactual simulations reflect modeled structural changes; adversary adaptation may alter real outcomes.",
    ]

    return {
        "summary": summary,
        "observed": observed,
        "predicted": predicted,
        "simulated": simulated,
        "recommended": recommended,
        "uncertainties": uncertainties,
    }


__all__ = ["generate_soc_explanation"]

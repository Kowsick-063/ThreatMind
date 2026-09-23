"""SOC defense decision engine for ThreatMind.

Evaluates:
  - H1-H12 attack trajectory risk
  - Target selection from real network graph observations
  - 7 defensive interventions via counterfactual rollout and graph simulation
  - Defense score ranking: defense_score = risk_reduction - operational_cost - disruption_cost
  - Near-tie identification and honest NO_ACTION recommendation
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from counterfactual.comparator import _compare_single_trajectory
from counterfactual.graph_simulator import get_graph_snapshot_by_timestamp, simulate_graph_counterfactual
from counterfactual.interventions import SUPPORTED_INTERVENTIONS, build_intervention
from counterfactual.rollout import simulate_counterfactual
from decision.decision_explanation import generate_defense_explanation
from decision.risk_analyzer import analyze_trajectory_risk
from forecasting.trajectory import generate_attack_trajectory


ALL_INTERVENTION_TYPES = [
    "NO_ACTION",
    "BLOCK_SOURCE",
    "BLOCK_DESTINATION",
    "BLOCK_PORT",
    "ISOLATE_HOST",
    "DISABLE_CONNECTION",
    "NETWORK_SEGMENTATION",
]


def select_real_targets(timestamp: float | int | None) -> dict[str, Any]:
    """Select real observed entities from the historical graph snapshot.

    Never fabricates targets. If the graph snapshot is missing or contains
    zero edges, returns None for graph-dependent interventions.
    """
    targets: dict[str, Any] = {
        "NO_ACTION": 0,
        "BLOCK_SOURCE": None,
        "BLOCK_DESTINATION": None,
        "BLOCK_PORT": None,
        "ISOLATE_HOST": None,
        "DISABLE_CONNECTION": None,
        "NETWORK_SEGMENTATION": None,
    }

    snapshot = get_graph_snapshot_by_timestamp(timestamp)
    if snapshot is None or len(snapshot.edge_index) == 0:
        return targets

    node_ids = snapshot.node_ids
    edge_index = snapshot.edge_index
    edge_features = snapshot.edge_features

    src_indices = edge_index[:, 0]
    dst_indices = edge_index[:, 1]

    src_ips = [node_ids[idx] for idx in src_indices if idx < len(node_ids)]
    dst_ips = [node_ids[idx] for idx in dst_indices if idx < len(node_ids)]

    # 1. Most active source IP
    if src_ips:
        targets["BLOCK_SOURCE"] = Counter(src_ips).most_common(1)[0][0]

    # 2. Most targeted destination IP
    if dst_ips:
        targets["BLOCK_DESTINATION"] = Counter(dst_ips).most_common(1)[0][0]

    # 3. Most frequent destination port (edge_features col 2)
    if edge_features is not None and edge_features.shape[1] > 2:
        dst_ports = [int(p) for p in edge_features[:, 2] if int(p) > 0]
        if dst_ports:
            targets["BLOCK_PORT"] = Counter(dst_ports).most_common(1)[0][0]

    # 4. Host with highest degree (in + out)
    all_active_ips = src_ips + dst_ips
    if all_active_ips:
        targets["ISOLATE_HOST"] = Counter(all_active_ips).most_common(1)[0][0]

    # 5. Most frequent source -> destination connection
    if src_ips and dst_ips and len(src_ips) == len(dst_ips):
        pairs = [f"{s}->{d}" for s, d in zip(src_ips, dst_ips)]
        if pairs:
            targets["DISABLE_CONNECTION"] = Counter(pairs).most_common(1)[0][0]

    # 6. Network segmentation target subnet prefix
    if src_ips:
        # Check for 192.168.10. internal subnet prefix common in CIC-IDS2017
        for ip in src_ips:
            if ip.startswith("192.168.10."):
                targets["NETWORK_SEGMENTATION"] = "192.168.10."
                break
        if targets["NETWORK_SEGMENTATION"] is None and len(src_ips) > 0:
            parts = src_ips[0].split(".")
            if len(parts) >= 3:
                targets["NETWORK_SEGMENTATION"] = f"{parts[0]}.{parts[1]}.{parts[2]}."

    return targets


def evaluate_defense_options(
    initial_state: dict[str, float] | None,
    timestamp: float | int | None,
    horizon: int = 12,
    custom_targets: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate all 7 interventions using existing simulation and scoring pipelines.

    Preserves the existing defense score:
      defense_score = risk_reduction - operational_cost - service_disruption_cost
    """
    state = initial_state or {}
    observed_targets = custom_targets or select_real_targets(timestamp)

    evaluated_options: list[dict[str, Any]] = []

    for itype in ALL_INTERVENTION_TYPES:
        target = observed_targets.get(itype)

        # Check if intervention requires an observed target that is absent
        if itype != "NO_ACTION" and (target is None or target == ""):
            evaluated_options.append({
                "intervention": {
                    "type": itype,
                    "target": None,
                },
                "available": False,
                "reason": "No valid target observed in current graph",
            })
            continue

        try:
            intervention = build_intervention(itype, target=target if target is not None else 0)
            simulation = simulate_counterfactual(
                initial_state=state,
                intervention=intervention,
                horizon=horizon,
                current_timestamp=timestamp,
            )
            graph_sim = simulate_graph_counterfactual(
                intervention=intervention,
                timestamp=timestamp,
            )
            graph_impact = graph_sim.get("graph_impact") if graph_sim.get("graph_available") else None

            comparison = _compare_single_trajectory(
                simulation["baseline"],
                simulation["counterfactual"],
                intervention,
                graph_impact=graph_impact,
            )
            summary = comparison["summary"]

            # Extract graph metrics
            if graph_impact is not None:
                surf_red = float(graph_impact.get("attack_surface_reduction_percentage", 0.0))
                path_red = float(graph_impact.get("attack_path_reduction_percentage", 0.0))
                edges_rem = int(graph_impact.get("edges_removed", 0))
                affected_paths = int(graph_impact.get("affected_attack_paths", 0))
            else:
                surf_red = 0.0
                path_red = 0.0
                edges_rem = 0
                affected_paths = 0

            evaluated_options.append({
                "intervention": {
                    "type": itype,
                    "target": target if target is not None else 0,
                    "description": intervention.description,
                },
                "available": True,
                "risk_reduction": round(float(summary["risk_reduction"]), 6),
                "risk_reduction_percentage": round(float(summary["risk_reduction_percentage"]), 6),
                "operational_cost": round(float(summary["operational_cost"]), 6),
                "service_disruption_cost": round(float(summary["service_disruption_cost"]), 6),
                "defense_score": round(float(summary["defense_score"]), 6),
                "graph_attack_surface_reduction": round(surf_red, 6),
                "graph_attack_path_reduction": round(path_red, 6),
                "edges_removed": edges_rem,
                "affected_paths": affected_paths,
                "graph_impact": graph_impact,
            })
        except Exception as exc:
            evaluated_options.append({
                "intervention": {
                    "type": itype,
                    "target": target,
                },
                "available": False,
                "reason": f"Simulation failure: {exc}",
            })

    return evaluated_options


def generate_defense_decision(
    timestamp: float | int | str,
    trajectory_result: dict[str, Any] | None = None,
    custom_targets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate a transparent SOC defense decision from trajectory and counterfactuals.

    Parameters
    ----------
    timestamp:
        Historical timestamp to analyze.
    trajectory_result:
        Optional precomputed trajectory from generate_attack_trajectory.
    custom_targets:
        Optional target dictionary overriding graph detection.

    Returns
    -------
    dict with timestamp, risk, trajectory_summary, interventions, decision,
    explanation, and uncertainties.
    """
    # 1. Obtain trajectory if not supplied
    if trajectory_result is None:
        trajectory_result = generate_attack_trajectory(current_timestamp=timestamp)

    trajectory = trajectory_result.get("trajectory", [])
    initial_state = trajectory_result.get("current_state", {})
    trajectory_summary = trajectory_result.get("trajectory_summary", {})

    # 2. Risk analysis
    risk_analysis = analyze_trajectory_risk(trajectory)

    # 3. Evaluate all 7 interventions
    evaluated = evaluate_defense_options(
        initial_state=initial_state,
        timestamp=timestamp,
        horizon=len(trajectory),
        custom_targets=custom_targets,
    )

    # 4. Rank available interventions
    available = [opt for opt in evaluated if opt.get("available", True)]
    if not available:
        # Fallback to NO_ACTION if everything errored
        no_action = build_intervention("NO_ACTION")
        available = [{
            "intervention": {"type": "NO_ACTION", "target": 0, "description": no_action.description},
            "available": True,
            "risk_reduction": 0.0,
            "risk_reduction_percentage": 0.0,
            "operational_cost": 0.0,
            "service_disruption_cost": 0.0,
            "defense_score": 0.0,
            "graph_attack_surface_reduction": 0.0,
            "graph_attack_path_reduction": 0.0,
            "edges_removed": 0,
            "affected_paths": 0,
            "graph_impact": None,
        }]

    ranked = sorted(available, key=lambda x: x["defense_score"], reverse=True)
    top_option = ranked[0]
    alternatives = ranked[1:]

    # 5. Near-tie detection (score delta < 0.01)
    near_ties: list[dict[str, Any]] = []
    top_score = top_option["defense_score"]
    for alt in alternatives:
        if abs(top_score - alt["defense_score"]) < 0.01:
            near_ties.append(alt)

    # 6. Formulate reasons
    reasons: list[str] = [
        f"Highest defense score ({top_option['defense_score']:.4f}) among all evaluated options.",
        f"Achieves risk reduction of {top_option['risk_reduction']:.4f} ({top_option['risk_reduction_percentage']:.1f}% reduction).",
        f"Operational cost ({top_option['operational_cost']:.2f}) and disruption cost ({top_option['service_disruption_cost']:.2f}) are justified by risk mitigation.",
    ]
    if top_option["intervention"]["type"] == "NO_ACTION":
        reasons.append(
            "Baseline risk trajectory is low or active countermeasures impose net disruption costs exceeding risk benefits."
        )
    if top_option.get("edges_removed", 0) > 0:
        reasons.append(
            f"Removes {top_option['edges_removed']} active graph edge(s) and reduces attack surface by {top_option['graph_attack_surface_reduction']:.2f}%."
        )
    if near_ties:
        tie_types = ", ".join(t["intervention"]["type"] for t in near_ties)
        reasons.append(
            f"Near-tie notice: Options [{tie_types}] are within 0.01 defense score of the recommended action."
        )

    decision_basis = {
        "defense_score": top_option["defense_score"],
        "risk_reduction": top_option["risk_reduction"],
        "operational_cost": top_option["operational_cost"],
        "service_disruption_cost": top_option["service_disruption_cost"],
        "graph_attack_surface_reduction": top_option.get("graph_attack_surface_reduction", 0.0),
        "graph_attack_path_reduction": top_option.get("graph_attack_path_reduction", 0.0),
        "edges_removed": top_option.get("edges_removed", 0),
        "affected_paths": top_option.get("affected_paths", 0),
    }

    # 7. Synthesize explanation
    graph_snapshot = get_graph_snapshot_by_timestamp(timestamp)
    explanation = generate_defense_explanation(
        recommended=top_option["intervention"],
        decision_basis=decision_basis,
        risk_analysis=risk_analysis,
        evaluated_interventions=evaluated,
        graph_impact=top_option.get("graph_impact"),
        graph_available=graph_snapshot is not None,
        near_ties=near_ties,
        current_timestamp=timestamp,
    )

    decision_output = {
        "recommended_intervention": top_option["intervention"],
        "decision_basis": decision_basis,
        "alternatives": alternatives,
        "reasons": reasons,
        "near_ties": [
            {
                "type": t["intervention"]["type"],
                "target": t["intervention"]["target"],
                "defense_score": t["defense_score"],
            }
            for t in near_ties
        ],
        "uncertainties": explanation["uncertainties"],
    }

    return {
        "timestamp": timestamp,
        "risk": risk_analysis,
        "trajectory_summary": trajectory_summary,
        "interventions": evaluated,
        "decision": decision_output,
        "explanation": explanation,
        "uncertainties": explanation["uncertainties"],
    }


__all__ = [
    "ALL_INTERVENTION_TYPES",
    "evaluate_defense_options",
    "generate_defense_decision",
    "select_real_targets",
]

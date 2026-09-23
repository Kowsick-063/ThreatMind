from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from counterfactual.comparator import compare_interventions
from counterfactual.defense_planner import rank_interventions
from counterfactual.graph_simulator import simulate_graph_counterfactual
from counterfactual.interventions import SUPPORTED_INTERVENTIONS, build_intervention
from counterfactual.rollout import simulate_counterfactual
from counterfactual.schemas import CounterfactualRequest
from counterfactual.scoring import compare_risk, compute_trajectory_risk
from counterfactual.trajectory import compare_trajectories
from explainability.counterfactual_explanation import explain_counterfactual

router = APIRouter(prefix="/api/counterfactual", tags=["Counterfactual"])


@router.get("/interventions")
def get_interventions():
    return {
        "interventions": [
            {
                "type": key,
                "target": value.target,
                "operational_cost": value.operational_cost,
                "service_disruption_cost": value.service_disruption_cost,
                "description": value.description,
            }
            for key, value in SUPPORTED_INTERVENTIONS.items()
        ]
    }


@router.post("/simulate")
def simulate(request: CounterfactualRequest):
    intervention = build_intervention(
        request.intervention.type,
        request.intervention.target,
    )
    state = request.state or {
        "flow_count": 120,
        "packet_count": 520,
        "total_packet_count": 520,
        "total_byte_count": 12000,
        "mean_flow_packets_per_sec": 2.1,
        "mean_flow_bytes_per_sec": 120.0,
        "syn_rate": 0.2,
        "ack_rate": 0.4,
        "rst_rate": 0.1,
    }

    simulation = simulate_counterfactual(
        initial_state=state,
        intervention=intervention,
        horizon=request.horizon,
        current_timestamp=request.current_timestamp,
    )
    baseline_summary = compute_trajectory_risk(simulation["baseline"])
    counterfactual_summary = compute_trajectory_risk(simulation["counterfactual"])
    delta = compare_risk(
        baseline_summary["cumulative_risk"],
        counterfactual_summary["cumulative_risk"],
    )

    comparison = {
        "baseline_risk": baseline_summary["cumulative_risk"],
        "counterfactual_risk": counterfactual_summary["cumulative_risk"],
        "risk_reduction": delta["risk_reduction"],
        "risk_reduction_percentage": delta["risk_reduction_percentage"],
        "peak_risk": counterfactual_summary["peak_risk"],
        "final_risk": counterfactual_summary["final_risk"],
    }
    explanation = explain_counterfactual(
        simulation["baseline"],
        simulation["counterfactual"],
        simulation["intervention"],
    )

    graph_simulation = simulate_graph_counterfactual(
        intervention=intervention,
        timestamp=request.current_timestamp,
    )

    response_payload = {
        "baseline": baseline_summary,
        "counterfactual": counterfactual_summary,
        "comparison": comparison,
        "explanation": explanation,
    }
    if graph_simulation.get("graph_available"):
        response_payload["baseline_graph"] = graph_simulation.get("baseline_graph_summary")
        response_payload["counterfactual_graph"] = graph_simulation.get("counterfactual_graph_summary")
        response_payload["graph_impact"] = graph_simulation.get("graph_impact")

    return response_payload


@router.post("/graph-simulate")
def graph_simulate(request: CounterfactualRequest):
    intervention = build_intervention(
        request.intervention.type,
        request.intervention.target,
    )
    return simulate_graph_counterfactual(
        intervention=intervention,
        timestamp=request.current_timestamp,
    )


@router.post("/compare")
def compare_all(request: CounterfactualRequest):
    state = request.state or {
        "flow_count": 120,
        "packet_count": 520,
        "total_packet_count": 520,
        "total_byte_count": 12000,
        "mean_flow_packets_per_sec": 2.1,
        "mean_flow_bytes_per_sec": 120.0,
        "syn_rate": 0.2,
        "ack_rate": 0.4,
        "rst_rate": 0.1,
    }
    results = compare_interventions(
        initial_state=state,
        intervention_types=list(SUPPORTED_INTERVENTIONS),
        horizon=request.horizon,
        current_timestamp=request.current_timestamp,
    )
    ranked = rank_interventions(results)
    return {
        "interventions": results,
        "ranked": ranked,
    }


@router.get("/trajectory/{trajectory_id}")
def get_trajectory(trajectory_id: str):
    return {
        "trajectory_id": trajectory_id,
        "baseline": [],
        "counterfactual": [],
        "comparison": [],
    }


__all__ = ["router"]

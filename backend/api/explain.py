"""POST /api/explain/attack — MITRE ATT&CK mapping and attack explainability endpoint."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from mitre.explanation import generate_attack_explanation
from counterfactual.graph_simulator import get_graph_snapshot_by_timestamp, simulate_graph_counterfactual
from counterfactual.graph_counterfactual import compute_attack_surface
from counterfactual.interventions import build_intervention

router = APIRouter(prefix="/api/explain", tags=["Explainability"])


class ExplainRequest(BaseModel):
    """Input schema for the attack explanation endpoint."""

    timestamp: float | None = Field(
        default=None,
        description=(
            "Unix timestamp (seconds) of the network state to explain. "
            "If provided, the real CIC-IDS2017 graph snapshot nearest to this "
            "timestamp (within 30 seconds) is used for graph evidence."
        ),
    )
    attack_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Calibrated attack probability from the ML pipeline. "
            "Reported in forecast_evidence only — not used as ATT&CK confidence."
        ),
    )
    attack_category: str | None = Field(
        default=None,
        description="Predicted attack category (e.g. 'PortScan', 'DDoS', 'Bot', 'BENIGN').",
    )
    state: dict[str, float] | None = Field(
        default=None,
        description=(
            "Network flow aggregate features keyed by feature name. "
            "Keys must correspond to the 26 LSTM feature names "
            "(flow_count, packet_count, byte_count, unique_sources, "
            "unique_destinations, unique_ports, syn_count, ack_count, "
            "rst_count, fin_count, psh_count, mean_flow_duration, "
            "mean_packets_per_flow, mean_bytes_per_flow, packets_per_second, "
            "bytes_per_second, mean_ttl, std_ttl, mean_tcp_window, "
            "std_tcp_window, mean_payload_size, std_payload_size, "
            "fragment_count, mean_iat, std_iat, max_iat)."
        ),
    )
    graph: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional caller-supplied graph summary. If timestamp is provided, "
            "the real graph snapshot is used instead and this field is ignored "
            "unless the timestamp lookup fails."
        ),
    )
    trajectory: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Optional 12-step forecast trajectory from the LSTM rollout. "
            "Each element should contain 'horizon', 'attack_probability', "
            "'attack_category', 'category_confidence', and 'risk'."
        ),
    )
    ports: list[int] | None = Field(
        default=None,
        description="Optional explicit list of destination ports observed.",
    )
    protocols: list[str] | None = Field(
        default=None,
        description="Optional list of protocols observed (e.g. ['TCP', 'UDP']).",
    )


@router.post("/attack")
def explain_attack(request: ExplainRequest) -> dict[str, Any]:
    """
    Generate a structured MITRE ATT&CK mapping and explainable attack narrative.

    The explanation integrates:
    - Evidence from observed network state features
    - Mapped MITRE ATT&CK techniques (with evidence confidence, not model probability)
    - Real graph topology evidence (if a graph snapshot exists at the timestamp)
    - Forecast trajectory evidence from the LSTM ML pipeline

    Returns a structured JSON object with:
      observations     — human-readable feature observations
      techniques       — ATT&CK technique matches with evidence confidence
      graph_evidence   — topology evidence per technique (if available)
      forecast_evidence— trajectory and probability from ML (separate from ATT&CK)
      uncertainties    — explicit limitations
      summary          — single-paragraph narrative
    """
    graph_features: dict[str, Any] | None = None
    graph_impact: dict[str, Any] | None = None
    graph_edges: list[Any] | None = None

    # --- Real graph lookup ---
    if request.timestamp is not None:
        snapshot = get_graph_snapshot_by_timestamp(request.timestamp)
        if snapshot is not None:
            attack_surface = compute_attack_surface(snapshot)
            graph_features = {
                "timestamp": snapshot.timestamp,
                "is_attack": snapshot.is_attack,
                "nodes": len(snapshot.node_ids),
                "edges": len(snapshot.edge_index),
                "unique_destinations": len(set(snapshot.edge_index[:, 1].tolist())) if len(snapshot.edge_index) > 0 else 0,
                "unique_sources": len(set(snapshot.edge_index[:, 0].tolist())) if len(snapshot.edge_index) > 0 else 0,
                "unique_services": attack_surface.get("unique_services", 0.0),
                "attack_surface_score": attack_surface.get("attack_surface_score", 0.0),
                "external_bridge_channels": attack_surface.get("external_bridge_channels", 0.0),
            }
            # Use NO_ACTION to get full graph impact baseline (zero removals = baseline summary)
            no_action = build_intervention("NO_ACTION")
            graph_sim = simulate_graph_counterfactual(intervention=no_action, timestamp=request.timestamp)
            if graph_sim.get("graph_available"):
                graph_impact = graph_sim.get("graph_impact")

            if len(snapshot.edge_index) > 0:
                graph_edges = snapshot.edge_index.tolist()
        else:
            # Timestamp provided but no matching snapshot — use caller-supplied graph if given
            if request.graph:
                graph_features = request.graph
    elif request.graph:
        graph_features = request.graph

    explanation = generate_attack_explanation(
        state=request.state,
        attack_category=request.attack_category,
        attack_probability=request.attack_probability,
        trajectory=request.trajectory,
        graph_features=graph_features,
        graph_impact=graph_impact,
        graph_edges=graph_edges,
        ports=request.ports,
        protocols=request.protocols,
        timestamp=request.timestamp,
    )

    # Attach graph availability metadata
    explanation["graph_available"] = graph_features is not None
    if request.timestamp is not None:
        explanation["timestamp"] = request.timestamp

    return explanation


@router.get("/techniques")
def list_techniques() -> dict[str, Any]:
    """Return the full MITRE ATT&CK technique registry used by ThreatMind."""
    from mitre.attack_techniques import list_techniques as _list
    return {"techniques": _list()}


__all__ = ["router"]

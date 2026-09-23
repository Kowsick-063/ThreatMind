"""Structured explainability layer for ThreatMind attack analysis.

generate_attack_explanation() assembles a structured explanation from:
  - raw network state features
  - mapped MITRE ATT&CK techniques
  - graph topology evidence (from Step 17 graph counterfactual)
  - forecast trajectory evidence (from LSTM rollout ML annotation)

Constraints
-----------
- No new ML inference is performed here.
- Communication paths are labelled as 'communication paths', not attack chains.
- Probability is reported separately from ATT&CK technique confidence.
- Limitations are always included — the explanations are honest about
  the boundaries of network-telemetry-only attribution.
"""

from __future__ import annotations

from typing import Any

from mitre.technique_mapper import map_network_behavior_to_techniques


# ---------------------------------------------------------------------------
# Observation extractors
# ---------------------------------------------------------------------------

def _extract_observations(
    state: dict[str, float],
    attack_category: str | None,
    attack_probability: float | None,
) -> list[str]:
    """Turn raw feature values into human-readable observation strings."""
    obs: list[str] = []

    def _v(key: str) -> float:
        return float(state.get(key, 0.0))

    flow_count = _v("flow_count")
    if flow_count > 0:
        obs.append(f"Observed {flow_count:.0f} network flows in the current monitoring window.")

    unique_ports = _v("unique_ports")
    if unique_ports > 50:
        obs.append(f"Unusually high number of unique destination ports: {unique_ports:.0f}. "
                   "This may indicate network service scanning activity.")
    elif unique_ports > 10:
        obs.append(f"Elevated unique destination ports: {unique_ports:.0f}.")

    unique_dsts = _v("unique_destinations")
    if unique_dsts > 30:
        obs.append(f"High destination diversity: {unique_dsts:.0f} unique destination hosts. "
                   "Consistent with broad-scan behaviour.")
    elif unique_dsts > 10:
        obs.append(f"Elevated destination diversity: {unique_dsts:.0f} unique destination hosts.")

    syn_count = _v("syn_count")
    pkt_total = _v("packet_count")
    if pkt_total > 0 and syn_count / pkt_total > 0.5:
        obs.append(f"High SYN ratio: {syn_count:.0f} SYNs out of {pkt_total:.0f} total packets "
                   f"({syn_count / pkt_total * 100:.1f}%). "
                   "Consistent with connection initiation at scale or half-open scanning.")

    rst_count = _v("rst_count")
    if rst_count > 0 and pkt_total > 0 and rst_count / pkt_total > 0.3:
        obs.append(f"High RST ratio: {rst_count:.0f} RSTs. "
                   "May indicate connection refusals (closed ports during scanning) "
                   "or session terminations under stress.")

    pps = _v("packets_per_second")
    if pps > 1000:
        obs.append(f"Very high packet rate: {pps:.0f} packets/second. "
                   "Consistent with volumetric DoS or high-throughput flood traffic.")
    elif pps > 200:
        obs.append(f"Elevated packet rate: {pps:.0f} packets/second.")

    bps = _v("bytes_per_second")
    if bps > 100000:
        obs.append(f"Very high byte rate: {bps:.0f} bytes/second. "
                   "Consistent with volumetric flooding.")

    mean_dur = _v("mean_flow_duration")
    if 0 < mean_dur < 0.5:
        obs.append(f"Very short mean flow duration: {mean_dur:.4f}s. "
                   "Consistent with rapid probe or SYN-only flows.")

    mean_iat = _v("mean_iat")
    std_iat = _v("std_iat")
    if mean_iat > 0 and std_iat / max(mean_iat, 1e-6) < 0.3:
        obs.append(f"Low inter-arrival time variance (mean IAT={mean_iat:.2f}s, std={std_iat:.2f}s). "
                   "Periodic traffic may indicate automated or scripted activity.")

    if attack_category and attack_category != "BENIGN":
        prob_str = f" (ML attack probability: {attack_probability:.3f})" if attack_probability is not None else ""
        obs.append(f"ML category estimator predicted attack category: '{attack_category}'{prob_str}.")

    if not obs:
        obs.append("No anomalous network behaviour was detected in the current monitoring window.")

    return obs


def _extract_graph_evidence(
    graph_features: dict[str, Any] | None,
    graph_impact: dict[str, Any] | None,
    techniques: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assemble graph topology evidence per technique."""
    if not graph_features and not graph_impact:
        return []

    gf = graph_features or {}
    gi = graph_impact or {}

    def _gv(d: dict, key: str, default: float = 0.0) -> float:
        v = d.get(key, default)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    graph_evidence: list[dict[str, Any]] = []

    # One block per matched technique that has graph signals.
    from mitre.attack_techniques import TECHNIQUE_REGISTRY
    for t_result in techniques:
        tech_id = t_result.get("technique_id", "")
        technique = TECHNIQUE_REGISTRY.get(tech_id)
        if technique is None or not technique.graph_signals:
            continue

        source_nodes: list[str] = list(gi.get("affected_sources", []))
        dest_nodes: list[str] = list(gi.get("affected_destinations", []))
        ports: list[int] = list(gi.get("affected_ports", []))
        affected_paths = int(_gv(gi, "affected_attack_paths"))
        remaining_paths = int(_gv(gi, "remaining_attack_paths"))
        edges = int(_gv(gf, "edges"))
        nodes = int(_gv(gf, "nodes"))
        attack_surface = _gv(gf, "attack_surface_score")

        if edges == 0 and nodes == 0 and not source_nodes:
            continue

        graph_evidence.append({
            "technique_id": tech_id,
            "technique_name": technique.technique_name,
            "source_nodes": source_nodes[:10],   # cap for readability
            "destination_nodes": dest_nodes[:10],
            "ports": ports[:20],
            "active_edges": edges,
            "active_nodes": nodes,
            "attack_surface_score": attack_surface,
            "affected_communication_paths": affected_paths,
            "remaining_communication_paths": remaining_paths,
            "note": (
                "Source/destination nodes and communication paths are derived "
                "from the real CIC-IDS2017 network graph at the nearest matching "
                "timestamp. Communication paths are directed 1-hop and 2-hop "
                "graph paths, not confirmed attacker kill chains."
            ),
        })

    return graph_evidence


def _extract_forecast_evidence(
    attack_probability: float | None,
    attack_category: str | None,
    trajectory: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Assemble forecast trajectory evidence. Kept completely separate from ATT&CK confidence."""
    evidence: dict[str, Any] = {}

    if attack_probability is not None:
        evidence["current_attack_probability"] = round(float(attack_probability), 4)

    if attack_category:
        evidence["category"] = attack_category

    if trajectory:
        final_prob = None
        for point in reversed(trajectory):
            p = point.get("attack_probability")
            if p is not None:
                try:
                    final_prob = round(float(p), 4)
                    break
                except (TypeError, ValueError):
                    pass

        if final_prob is not None:
            evidence["future_attack_probability_h12"] = final_prob

        risks = [float(p.get("risk", 0.0)) for p in trajectory if p.get("risk") is not None]
        if risks:
            evidence["trajectory_peak_risk"] = round(max(risks), 4)
            evidence["trajectory_mean_risk"] = round(sum(risks) / len(risks), 4)

        evidence["note"] = (
            "Forecast probability is produced by the trained AttackProbabilityModel + "
            "horizon-specific Platt calibrators. It is NOT equivalent to ATT&CK technique "
            "confidence, which is derived independently from evidence feature strength."
        )

    return evidence


def _build_uncertainties(
    state: dict[str, float],
    techniques: list[dict[str, Any]],
    graph_available: bool,
    attack_probability: float | None,
) -> list[str]:
    """Always include explicit limitations."""
    uncertainties: list[str] = [
        "Network telemetry alone cannot confirm attacker identity, intent, or specific "
        "MITRE ATT&CK procedure (sub-technique) without additional host-level or "
        "endpoint forensics.",
        "All ATT&CK technique confidence values represent the fraction of expected "
        "network-observable evidence conditions that were satisfied, not a probability "
        "of technique execution.",
        "The CIC-IDS2017 dataset contains network flows captured in a controlled lab "
        "environment. Applicability to production networks may differ.",
    ]

    if not graph_available:
        uncertainties.append(
            "No real graph snapshot was available at the requested timestamp "
            "(tolerance: ±30 seconds). Graph evidence is absent from this explanation."
        )

    if attack_probability is not None and attack_probability < 0.5:
        uncertainties.append(
            f"Attack probability is low ({attack_probability:.3f}). "
            "Technique mappings are speculative and should not be acted upon without "
            "corroborating evidence."
        )

    if not techniques:
        uncertainties.append(
            "No MITRE ATT&CK techniques met the minimum evidence threshold. "
            "The observed network state does not produce enough feature-level evidence "
            "to support a confident technique attribution."
        )

    if techniques and all(t["confidence"] < 0.3 for t in techniques):
        uncertainties.append(
            "All matched techniques have low evidence confidence (< 0.30). "
            "These mappings should be treated as weak indicators requiring "
            "additional corroboration."
        )

    return uncertainties


def _build_summary(
    observations: list[str],
    techniques: list[dict[str, Any]],
    attack_category: str | None,
    attack_probability: float | None,
) -> str:
    """Produce a one-paragraph human-readable summary."""
    if not techniques and not observations:
        return (
            "No significant network anomalies or MITRE ATT&CK technique indicators "
            "were detected in the current monitoring window."
        )

    top_tech = techniques[0] if techniques else None

    category_part = ""
    if attack_category and attack_category != "BENIGN":
        category_part = f" The ML category estimator identified '{attack_category}' behaviour"
        if attack_probability is not None:
            category_part += f" with an attack probability of {attack_probability:.3f}"
        category_part += "."

    tech_part = ""
    if top_tech:
        tech_part = (
            f" The strongest evidence-based technique match is "
            f"{top_tech['technique_id']} ({top_tech['technique_name']}) "
            f"with an evidence confidence of {top_tech['confidence']:.2f}."
        )
        if len(techniques) > 1:
            others = ", ".join(
                f"{t['technique_id']} ({t['technique_name']})"
                for t in techniques[1:3]
            )
            tech_part += f" Additional weaker indicators include: {others}."

    limit_part = (
        " Note: ATT&CK technique confidence reflects evidence strength from network "
        "telemetry only and should not be equated with model attack probability."
    )

    obs_part = ""
    if observations:
        obs_part = f" Key observation: {observations[0]}"

    return (
        f"ThreatMind detected anomalous network activity in the current monitoring window.{obs_part}"
        f"{category_part}{tech_part}{limit_part}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_attack_explanation(
    state: dict[str, float] | None = None,
    attack_category: str | None = None,
    attack_probability: float | None = None,
    trajectory: list[dict[str, Any]] | None = None,
    graph_features: dict[str, Any] | None = None,
    graph_impact: dict[str, Any] | None = None,
    graph_edges: list[Any] | None = None,
    ports: list[int] | None = None,
    protocols: list[str] | None = None,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """
    Generate a structured, evidence-linked attack explanation.

    Parameters
    ----------
    state:
        26-dim network state feature dict (FEATURES keys).
    attack_category:
        Predicted category from attack_category_estimation model.
    attack_probability:
        Calibrated probability from attack_probability_model + calibrators.
        Reported as-is under forecast_evidence. NOT used as ATT&CK confidence.
    trajectory:
        Optional 12-step forecast trajectory from rollout.py.
    graph_features:
        Optional summary dict from a real graph snapshot.
    graph_impact:
        Optional impact dict from graph_counterfactual.calculate_graph_impact().
    graph_edges:
        Optional raw edge list (not scored, passed through).
    ports:
        Optional explicit port list.
    protocols:
        Optional protocol list.
    timestamp:
        The timestamp used for graph snapshot lookup (informational only).

    Returns
    -------
    {
        "summary": str,
        "observations": [str, ...],
        "techniques": [
            {
                "technique_id": str,
                "technique_name": str,
                "tactic": str,
                "confidence": float,
                "evidence": [str, ...]
            },
            ...
        ],
        "graph_evidence": [...],
        "forecast_evidence": {...},
        "uncertainties": [str, ...]
    }
    """
    if state is None:
        state = {}

    # --- MITRE technique mapping ---
    mapping_result = map_network_behavior_to_techniques(
        state=state,
        attack_category=attack_category,
        attack_probability=attack_probability,
        graph_features=graph_features,
        graph_edges=graph_edges,
        ports=ports,
        protocols=protocols,
    )
    techniques = mapping_result["techniques"]

    # --- Observations ---
    observations = _extract_observations(state, attack_category, attack_probability)

    # --- Graph evidence ---
    graph_available = graph_features is not None or graph_impact is not None
    graph_evidence = _extract_graph_evidence(graph_features, graph_impact, techniques)

    # --- Forecast evidence ---
    forecast_evidence = _extract_forecast_evidence(
        attack_probability, attack_category, trajectory
    )

    # --- Uncertainties ---
    uncertainties = _build_uncertainties(state, techniques, graph_available, attack_probability)

    # --- Summary ---
    summary = _build_summary(observations, techniques, attack_category, attack_probability)

    return {
        "summary": summary,
        "observations": observations,
        "techniques": techniques,
        "graph_evidence": graph_evidence,
        "forecast_evidence": forecast_evidence,
        "uncertainties": uncertainties,
        "graph_available": graph_available,
    }


__all__ = ["generate_attack_explanation"]

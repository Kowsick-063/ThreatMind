"""Evidence-based MITRE ATT&CK technique mapper.

Design
------
Confidence values represent evidence strength — the fraction of required
conditions and supporting signals satisfied by the observed network state.

Confidence is NOT the model's attack_probability.  The two are kept entirely
separate and must not be conflated.

Confidence formula
------------------
  base_score     = conditions_met / total_conditions        (0.0–1.0)
  feature_bonus  = features_observed / total_features * 0.2 (0.0–0.2)
  raw_confidence = min(base_score + feature_bonus, 1.0)
  confidence     = raw_confidence * technique.max_evidence_confidence

This guarantees:
  - A technique with zero conditions met produces confidence = 0.
  - A technique with all conditions met and all features present produces
    confidence <= max_evidence_confidence, not 1.0.
  - model attack_probability is never used as a confidence input.
"""

from __future__ import annotations

import operator as op_mod
from typing import Any

from mitre.attack_techniques import TECHNIQUE_REGISTRY, AttackTechnique


# ---------------------------------------------------------------------------
# Operator helpers
# ---------------------------------------------------------------------------

_OPS: dict[str, Any] = {
    ">":  op_mod.gt,
    ">=": op_mod.ge,
    "<":  op_mod.lt,
    "<=": op_mod.le,
    "==": op_mod.eq,
    "!=": op_mod.ne,
}


def _eval_condition(state: dict[str, float], feature: str, operator: str, threshold: float) -> bool:
    """Evaluate one required condition against the observed network state."""
    if feature not in state:
        return False
    fn = _OPS.get(operator)
    if fn is None:
        return False
    try:
        return bool(fn(float(state[feature]), float(threshold)))
    except (TypeError, ValueError):
        return False


def _score_technique(
    technique: AttackTechnique,
    state: dict[str, float],
    graph_features: dict[str, Any] | None,
    attack_category: str | None,
) -> tuple[float, list[str]]:
    """
    Compute evidence confidence and gather supporting evidence strings.

    Returns
    -------
    (confidence, evidence_strings)
    """
    evidence: list[str] = []

    # --- Required conditions ---
    conditions = technique.required_conditions
    n_conditions = len(conditions)
    if n_conditions == 0:
        conditions_score = 0.0
    else:
        met = 0
        for feat, oper, thr in conditions:
            if _eval_condition(state, feat, oper, thr):
                met += 1
                val = state.get(feat, "?")
                evidence.append(
                    f"{feat} = {val:.4g} (condition: {feat} {oper} {thr} ✓)"
                )
            else:
                val = state.get(feat, "N/A")
                val_str = f"{val:.4g}" if isinstance(val, (int, float)) else str(val)
                evidence.append(
                    f"{feat} = {val_str} (condition: {feat} {oper} {thr} ✗)"
                )
        conditions_score = met / n_conditions

    # If no conditions are met at all, skip the technique.
    if conditions_score == 0.0 and n_conditions > 0:
        return 0.0, []

    # --- Feature presence bonus (which evidence features are non-zero) ---
    total_feats = len(technique.evidence_features)
    feats_observed = 0
    for feat in technique.evidence_features:
        val = state.get(feat)
        if val is not None and float(val) != 0.0:
            feats_observed += 1
    feature_bonus = (feats_observed / max(total_feats, 1)) * 0.2

    # --- Category consistency bonus ---
    category_bonus = 0.0
    if attack_category and attack_category in technique.consistent_categories:
        category_bonus = 0.1
        evidence.append(
            f"Observed attack category '{attack_category}' is consistent with {technique.technique_id}."
        )

    # --- Graph signal bonus ---
    graph_bonus = 0.0
    if graph_features:
        graph_signal_hits = _check_graph_signals(technique, graph_features, evidence)
        graph_bonus = min(graph_signal_hits * 0.05, 0.15)

    raw_confidence = min(conditions_score + feature_bonus + category_bonus + graph_bonus, 1.0)
    confidence = round(raw_confidence * technique.max_evidence_confidence, 4)

    return confidence, evidence


def _check_graph_signals(
    technique: AttackTechnique,
    graph_features: dict[str, Any],
    evidence: list[str],
) -> int:
    """
    Check graph topology signals and return count of hits.

    Graph signals are high-level observations derived from graph snapshot
    metrics (edge count, port diversity, node count, byte volume, etc.).
    """
    hits = 0
    gf = graph_features

    def _gv(key: str, default: float = 0.0) -> float:
        val = gf.get(key, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    for signal in technique.graph_signals:
        matched = False

        if signal == "high_destination_diversity":
            unique_dsts = _gv("unique_destinations") or _gv("destination_count")
            if unique_dsts > 20:
                matched = True
                evidence.append(f"Graph: {unique_dsts:.0f} unique destination nodes observed.")

        elif signal == "many_unique_ports_in_edges":
            up = _gv("unique_services") or _gv("unique_ports")
            if up > 30:
                matched = True
                evidence.append(f"Graph: {up:.0f} unique destination ports observed in edges.")

        elif signal == "low_edge_to_node_ratio":
            edges = _gv("edges") or _gv("edge_count")
            nodes = _gv("nodes") or _gv("node_count")
            if nodes > 0 and (edges / nodes) < 2.0:
                matched = True
                evidence.append(
                    f"Graph: edge-to-node ratio {edges / nodes:.2f} is low, consistent with scanning."
                )

        elif signal == "high_edge_count":
            edges = _gv("edges") or _gv("edge_count")
            if edges > 100:
                matched = True
                evidence.append(f"Graph: {edges:.0f} active communication edges observed.")

        elif signal == "high_byte_volume":
            bv = _gv("byte_volume_removed") or _gv("byte_count")
            if bv > 50000:
                matched = True
                evidence.append(f"Graph: high byte volume {bv:.0f} bytes on graph edges.")

        elif signal == "many_unique_sources":
            us = _gv("unique_sources") or _gv("source_count")
            if us > 10:
                matched = True
                evidence.append(f"Graph: {us:.0f} unique source nodes observed.")

        elif signal == "concentrated_destination_ports":
            up = _gv("unique_services") or _gv("unique_ports")
            edges = _gv("edges") or _gv("edge_count")
            if edges > 0 and up < 5:
                matched = True
                evidence.append(
                    f"Graph: flows concentrated on {up:.0f} destination port(s) — consistent with brute force."
                )

        elif signal == "low_port_diversity":
            up = _gv("unique_services") or _gv("unique_ports")
            if up < 5:
                matched = True
                evidence.append(f"Graph: only {up:.0f} unique port(s) observed.")

        elif signal == "repeated_source_destination_pairs":
            paths = _gv("affected_attack_paths") or _gv("remaining_attack_paths")
            if paths > 5:
                matched = True
                evidence.append(
                    f"Graph: {paths:.0f} communication paths from a small source set."
                )

        elif signal == "high_node_count":
            nodes = _gv("nodes") or _gv("node_count")
            if nodes > 50:
                matched = True
                evidence.append(f"Graph: {nodes:.0f} network nodes active in this snapshot.")

        elif signal == "low_mean_edge_weight":
            edges = _gv("edges") or _gv("edge_count")
            bv = _gv("byte_volume_removed") or _gv("byte_count")
            if edges > 0 and (bv / edges) < 100:
                matched = True
                evidence.append(
                    f"Graph: mean bytes per edge {bv / edges:.1f} — low weight consistent with probing."
                )

        if matched:
            hits += 1

    return hits


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def map_network_behavior_to_techniques(
    state: dict[str, float] | None = None,
    attack_category: str | None = None,
    attack_probability: float | None = None,
    graph_features: dict[str, Any] | None = None,
    graph_edges: list[Any] | None = None,
    ports: list[int] | None = None,
    protocols: list[str] | None = None,
    min_confidence: float = 0.05,
) -> dict[str, Any]:
    """
    Map observed network behavior to MITRE ATT&CK techniques.

    Parameters
    ----------
    state:
        Dictionary of network flow aggregates keyed by feature name.
        Keys must correspond to the FEATURES list in create_sequences.py.
    attack_category:
        Category string predicted by the attack category estimator
        (e.g. "PortScan", "DDoS", "Bot", "BENIGN").
    attack_probability:
        Calibrated attack probability from the ML pipeline.
        Used for evidence framing only — NOT converted to ATT&CK confidence.
    graph_features:
        Optional graph-level metrics from a real graph snapshot.
        Keys may include: nodes, edges, unique_services, unique_destinations,
        unique_sources, byte_count, attack_surface_score, etc.
    graph_edges:
        Optional edge list (not used for confidence scoring, kept for
        downstream graph evidence assembly).
    ports:
        Optional explicit port list observed in the current snapshot.
    protocols:
        Optional protocol list.
    min_confidence:
        Minimum confidence threshold below which a technique is not returned.

    Returns
    -------
    {
        "techniques": [
            {
                "technique_id": str,
                "technique_name": str,
                "tactic": str,
                "confidence": float,
                "evidence": [str, ...]
            },
            ...
        ]
    }

    Confidence represents the proportion of required network conditions and
    supporting evidence met by the observed state, scaled to the technique's
    maximum achievable evidence confidence.

    Confidence is NOT the model's attack_probability.
    """
    if state is None:
        state = {}

    # Augment state with explicit ports list if graph_features provides them.
    effective_state = dict(state)
    if ports and "unique_ports" not in effective_state:
        effective_state["unique_ports"] = float(len(set(ports)))

    results: list[dict[str, Any]] = []

    for tech_id, technique in TECHNIQUE_REGISTRY.items():
        confidence, evidence = _score_technique(
            technique, effective_state, graph_features, attack_category
        )
        if confidence >= min_confidence:
            results.append({
                "technique_id": tech_id,
                "technique_name": technique.technique_name,
                "tactic": technique.tactic,
                "confidence": confidence,
                "evidence": evidence,
            })

    # Sort by confidence descending.
    results.sort(key=lambda x: x["confidence"], reverse=True)

    return {"techniques": results}


__all__ = ["map_network_behavior_to_techniques"]

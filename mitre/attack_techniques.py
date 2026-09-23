"""Static MITRE ATT&CK technique registry grounded in CIC-IDS2017 network evidence.

Design constraints
------------------
- Only techniques with clear, observable, feature-level evidence in CIC-IDS2017
  aggregated flow telemetry are included.
- Each technique specifies the exact network features that support it and the
  minimum conditions that must be met for the technique to be considered.
- Confidence scores are derived from evidence strength counts, not from model
  probability values.
- Techniques that require host-level telemetry (e.g. process trees, registry
  keys, memory forensics) are NOT included because the dataset provides only
  network flow aggregates.

Evidence feature names correspond to the FEATURES list in
models/world_model/create_sequences.py plus graph topology signals from
counterfactual/graph_counterfactual.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AttackTechnique:
    """One MITRE ATT&CK technique entry with evidence grounding."""

    technique_id: str
    technique_name: str
    tactic: str
    description: str
    # Network flow features that provide direct evidence for this technique.
    evidence_features: list[str]
    # Threshold conditions that must evaluate True against observed network state.
    # Each condition is a (feature_name, operator, threshold) triple, e.g.:
    #   ("unique_ports", ">", 50)
    required_conditions: list[tuple[str, str, float]]
    # Graph-level signals that additionally support this technique (optional).
    graph_signals: list[str] = field(default_factory=list)
    # Observable CIC-IDS2017 categories that are consistent with this technique.
    consistent_categories: list[str] = field(default_factory=list)
    # Maximum confidence achievable purely from network evidence (0.0–1.0).
    # This is a design parameter, not a model probability.
    max_evidence_confidence: float = 0.75

    def to_dict(self) -> dict[str, Any]:
        return {
            "technique_id": self.technique_id,
            "technique_name": self.technique_name,
            "tactic": self.tactic,
            "description": self.description,
            "evidence_features": self.evidence_features,
            "required_conditions": [
                {"feature": f, "operator": op, "threshold": thr}
                for f, op, thr in self.required_conditions
            ],
            "graph_signals": self.graph_signals,
            "consistent_categories": self.consistent_categories,
            "max_evidence_confidence": self.max_evidence_confidence,
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TECHNIQUE_REGISTRY: dict[str, AttackTechnique] = {

    # ------------------------------------------------------------------
    # T1046 — Network Service Discovery / Port Scanning
    # Grounded in: high unique_ports, high unique_destinations,
    #              elevated syn_count, short mean_flow_duration,
    #              low mean_packets_per_flow.
    # ------------------------------------------------------------------
    "T1046": AttackTechnique(
        technique_id="T1046",
        technique_name="Network Service Discovery",
        tactic="Discovery",
        description=(
            "Adversaries may attempt to get a listing of services running on "
            "remote hosts by scanning the network. Evidence: rapid connection "
            "attempts across many destination ports from a small source set, "
            "characterized by high SYN rates and short-lived flows."
        ),
        evidence_features=[
            "unique_ports",
            "unique_destinations",
            "syn_count",
            "mean_flow_duration",
            "mean_packets_per_flow",
        ],
        required_conditions=[
            ("unique_ports", ">", 50),
        ],
        graph_signals=[
            "high_destination_diversity",
            "many_unique_ports_in_edges",
            "low_edge_to_node_ratio",
        ],
        consistent_categories=["PortScan"],
        max_evidence_confidence=0.80,
    ),

    # ------------------------------------------------------------------
    # T1498 — Network Denial of Service
    # Grounded in: extreme packet/byte/flow volumes, high SYN or RST,
    #              high packets_per_second, high bytes_per_second,
    #              low mean_flow_duration (UDP floods) or high syn_count.
    # ------------------------------------------------------------------
    "T1498": AttackTechnique(
        technique_id="T1498",
        technique_name="Network Denial of Service",
        tactic="Impact",
        description=(
            "Adversaries may perform Network Denial of Service (DoS) attacks "
            "to degrade or block the availability of targeted resources. "
            "Evidence: abnormally high packet/byte rates, elevated SYN or RST "
            "counts, and high unique source counts consistent with amplification "
            "or reflection attacks."
        ),
        evidence_features=[
            "flow_count",
            "packet_count",
            "byte_count",
            "packets_per_second",
            "bytes_per_second",
            "syn_count",
            "rst_count",
        ],
        required_conditions=[
            ("packets_per_second", ">", 1000),
        ],
        graph_signals=[
            "high_edge_count",
            "high_byte_volume",
            "many_unique_sources",
        ],
        consistent_categories=["DDoS", "DoS"],
        max_evidence_confidence=0.78,
    ),

    # ------------------------------------------------------------------
    # T1110 — Brute Force
    # Grounded in: many flows to the same destination port (22, 3389, 443),
    #              repeated small payloads, many SYN-ACK sequences to the
    #              same destination, elevated unique_sources to one target.
    # ------------------------------------------------------------------
    "T1110": AttackTechnique(
        technique_id="T1110",
        technique_name="Brute Force",
        tactic="Credential Access",
        description=(
            "Adversaries may use brute force techniques to gain access to "
            "accounts when passwords are unknown or hash cracking is infeasible. "
            "Evidence: repeated authentication-protocol connections (SSH/RDP/HTTP) "
            "from one or a few sources to the same destination, with low payload "
            "variance and elevated SYN-ACK round trips."
        ),
        evidence_features=[
            "flow_count",
            "unique_destinations",
            "unique_ports",
            "syn_count",
            "ack_count",
            "mean_packets_per_flow",
            "mean_payload_size",
        ],
        required_conditions=[
            ("syn_count", ">", 200),
            ("unique_ports", "<", 5),
        ],
        graph_signals=[
            "concentrated_destination_ports",
            "low_port_diversity",
            "repeated_source_destination_pairs",
        ],
        consistent_categories=["FTP-Patator", "SSH-Patator", "Brute Force"],
        max_evidence_confidence=0.70,
    ),

    # ------------------------------------------------------------------
    # T1595.001 — Active Scanning: Scanning IP Blocks
    # Grounded in: high unique_sources or unique_destinations,
    #              very low mean_flow_duration (probe-style), high syn_count,
    #              low mean_packets_per_flow.
    # ------------------------------------------------------------------
    "T1595.001": AttackTechnique(
        technique_id="T1595.001",
        technique_name="Active Scanning: Scanning IP Blocks",
        tactic="Reconnaissance",
        description=(
            "Adversaries may scan victim IP blocks to gather information that "
            "can be used during targeting. Evidence: rapid probe flows across "
            "many unique destinations with very short durations and minimal "
            "payload exchange."
        ),
        evidence_features=[
            "unique_destinations",
            "unique_sources",
            "syn_count",
            "mean_flow_duration",
            "mean_packets_per_flow",
        ],
        required_conditions=[
            ("unique_destinations", ">", 30),
            ("mean_flow_duration", "<", 1.0),
        ],
        graph_signals=[
            "high_destination_diversity",
            "high_node_count",
            "low_mean_edge_weight",
        ],
        consistent_categories=["PortScan"],
        max_evidence_confidence=0.72,
    ),

    # ------------------------------------------------------------------
    # T1071.001 — Application Layer Protocol: Web Protocols
    # Grounded in: elevated mean_payload_size, large byte_count,
    #              moderate flow_count, low syn_count (established sessions),
    #              mean_ttl consistent with long-haul routes.
    # ------------------------------------------------------------------
    "T1071.001": AttackTechnique(
        technique_id="T1071.001",
        technique_name="Application Layer Protocol: Web Protocols",
        tactic="Command and Control",
        description=(
            "Adversaries may communicate using application layer protocols "
            "associated with web traffic to avoid detection. Evidence: "
            "elevated mean payload sizes with established TCP sessions "
            "(low SYN ratio) and inter-arrival times consistent with "
            "periodic callback behaviour."
        ),
        evidence_features=[
            "mean_payload_size",
            "byte_count",
            "flow_count",
            "syn_count",
            "mean_iat",
            "std_iat",
        ],
        required_conditions=[
            ("mean_payload_size", ">", 500),
            ("syn_count", "<", 100),
        ],
        graph_signals=[
            "established_tcp_edges",
            "moderate_payload_volume",
        ],
        consistent_categories=["Bot"],
        max_evidence_confidence=0.60,
    ),

    # ------------------------------------------------------------------
    # T1059 — Command and Scripting Interpreter
    # Grounded in: bot-like periodic flow patterns, low flow duration
    #              variance, periodic IAT, small but consistent payloads.
    #              This is a weaker mapping — confidence capped low.
    # ------------------------------------------------------------------
    "T1059": AttackTechnique(
        technique_id="T1059",
        technique_name="Command and Scripting Interpreter",
        tactic="Execution",
        description=(
            "Adversaries may abuse command and script interpreters to execute "
            "commands, scripts, or binaries. Network-observable evidence is "
            "indirect: periodic, low-variance flows with consistent small "
            "payloads may indicate automated scripted behaviour. This mapping "
            "has low specificity from network telemetry alone."
        ),
        evidence_features=[
            "mean_iat",
            "std_iat",
            "mean_payload_size",
            "std_payload_size",
            "flow_count",
        ],
        required_conditions=[
            ("std_iat", "<", 5.0),
            ("flow_count", ">", 10),
        ],
        graph_signals=[
            "periodic_edge_pattern",
        ],
        consistent_categories=["Bot"],
        max_evidence_confidence=0.45,
    ),
}


def get_technique(technique_id: str) -> AttackTechnique | None:
    """Return a technique by ID, or None if not registered."""
    return TECHNIQUE_REGISTRY.get(technique_id)


def list_techniques() -> list[dict[str, Any]]:
    """Return all registered techniques as serialisable dicts."""
    return [t.to_dict() for t in TECHNIQUE_REGISTRY.values()]


__all__ = [
    "AttackTechnique",
    "TECHNIQUE_REGISTRY",
    "get_technique",
    "list_techniques",
]

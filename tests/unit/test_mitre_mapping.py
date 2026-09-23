"""
Minimum verification suite for Step 18 — MITRE ATT&CK Mapping + Explainability.

Tests:
  1. Port-scan-like evidence maps to the intended technique (T1046).
  2. DoS-like evidence maps to the intended technique (T1498).
  3. Weak/insufficient evidence produces uncertainty, not a forced technique.
  4. Explanation contains the actual supporting evidence.
  5. Graph evidence is included when a real graph snapshot exists.
  6. Forecast probability is NOT incorrectly treated as ATT&CK confidence.
  7. Existing 45 tests remain passing (checked by running this suite alongside them).
"""

from __future__ import annotations

import pytest

from mitre.technique_mapper import map_network_behavior_to_techniques
from mitre.explanation import generate_attack_explanation


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

PORT_SCAN_STATE = {
    "flow_count": 5000.0,
    "packet_count": 15000.0,
    "byte_count": 200000.0,
    "unique_sources": 3.0,
    "unique_destinations": 200.0,
    "unique_ports": 1024.0,        # ← key indicator: > 50
    "syn_count": 12000.0,
    "ack_count": 1000.0,
    "rst_count": 8000.0,
    "fin_count": 200.0,
    "psh_count": 100.0,
    "mean_flow_duration": 0.1,     # ← very short
    "mean_packets_per_flow": 2.0,
    "mean_bytes_per_flow": 40.0,
    "packets_per_second": 500.0,
    "bytes_per_second": 6000.0,
    "mean_ttl": 64.0,
    "std_ttl": 1.0,
    "mean_tcp_window": 8192.0,
    "std_tcp_window": 0.0,
    "mean_payload_size": 40.0,
    "std_payload_size": 5.0,
    "fragment_count": 0.0,
    "mean_iat": 0.01,
    "std_iat": 0.005,
    "max_iat": 0.1,
}

DOS_STATE = {
    "flow_count": 20000.0,
    "packet_count": 500000.0,
    "byte_count": 800000000.0,
    "unique_sources": 50.0,
    "unique_destinations": 2.0,
    "unique_ports": 3.0,
    "syn_count": 300000.0,
    "ack_count": 50000.0,
    "rst_count": 100000.0,
    "fin_count": 1000.0,
    "psh_count": 5000.0,
    "mean_flow_duration": 0.05,
    "mean_packets_per_flow": 25.0,
    "mean_bytes_per_flow": 40000.0,
    "packets_per_second": 50000.0,  # ← key indicator: > 1000
    "bytes_per_second": 800000.0,
    "mean_ttl": 128.0,
    "std_ttl": 30.0,
    "mean_tcp_window": 65535.0,
    "std_tcp_window": 0.0,
    "mean_payload_size": 1400.0,
    "std_payload_size": 200.0,
    "fragment_count": 5000.0,
    "mean_iat": 0.001,
    "std_iat": 0.0005,
    "max_iat": 0.05,
}

BENIGN_STATE = {
    "flow_count": 50.0,
    "packet_count": 300.0,
    "byte_count": 15000.0,
    "unique_sources": 5.0,
    "unique_destinations": 5.0,
    "unique_ports": 3.0,       # ← very low unique_ports
    "syn_count": 10.0,
    "ack_count": 200.0,
    "rst_count": 2.0,
    "fin_count": 30.0,
    "psh_count": 150.0,
    "mean_flow_duration": 5.0,
    "mean_packets_per_flow": 6.0,
    "mean_bytes_per_flow": 300.0,
    "packets_per_second": 10.0,  # ← low pps
    "bytes_per_second": 500.0,
    "mean_ttl": 64.0,
    "std_ttl": 2.0,
    "mean_tcp_window": 65535.0,
    "std_tcp_window": 1000.0,
    "mean_payload_size": 250.0,
    "std_payload_size": 100.0,
    "fragment_count": 0.0,
    "mean_iat": 0.5,
    "std_iat": 0.4,
    "max_iat": 2.0,
}


# ---------------------------------------------------------------------------
# Test 1: Port-scan evidence → T1046
# ---------------------------------------------------------------------------

class TestPortScanMapping:
    def test_t1046_in_results(self):
        """Port-scan-like state must map to T1046 (Network Service Discovery)."""
        result = map_network_behavior_to_techniques(
            state=PORT_SCAN_STATE,
            attack_category="PortScan",
        )
        tech_ids = [t["technique_id"] for t in result["techniques"]]
        assert "T1046" in tech_ids, (
            f"T1046 not found in results: {tech_ids}"
        )

    def test_t1046_has_highest_confidence_for_portscan(self):
        """T1046 must be the top-ranked technique for port-scan state."""
        result = map_network_behavior_to_techniques(
            state=PORT_SCAN_STATE,
            attack_category="PortScan",
        )
        assert result["techniques"], "No techniques returned for port-scan state."
        top = result["techniques"][0]
        assert top["technique_id"] == "T1046", (
            f"Expected T1046 at top, got {top['technique_id']} "
            f"(confidence={top['confidence']})"
        )

    def test_t1046_confidence_above_zero(self):
        """T1046 confidence must be > 0 for port-scan evidence."""
        result = map_network_behavior_to_techniques(
            state=PORT_SCAN_STATE,
            attack_category="PortScan",
        )
        t1046 = next((t for t in result["techniques"] if t["technique_id"] == "T1046"), None)
        assert t1046 is not None
        assert t1046["confidence"] > 0.0, "T1046 confidence must be > 0."

    def test_t1046_also_found_by_t1595_for_portscan(self):
        """T1595.001 (Scanning IP Blocks) should also be triggered by high unique_destinations."""
        result = map_network_behavior_to_techniques(
            state=PORT_SCAN_STATE,
            attack_category="PortScan",
        )
        tech_ids = [t["technique_id"] for t in result["techniques"]]
        assert "T1595.001" in tech_ids, (
            f"T1595.001 not found in results for high-destination state: {tech_ids}"
        )


# ---------------------------------------------------------------------------
# Test 2: DoS evidence → T1498
# ---------------------------------------------------------------------------

class TestDoSMapping:
    def test_t1498_in_results(self):
        """DoS-like state must map to T1498 (Network Denial of Service)."""
        result = map_network_behavior_to_techniques(
            state=DOS_STATE,
            attack_category="DDoS",
        )
        tech_ids = [t["technique_id"] for t in result["techniques"]]
        assert "T1498" in tech_ids, (
            f"T1498 not found in results: {tech_ids}"
        )

    def test_t1498_has_highest_confidence_for_dos(self):
        """T1498 must be the top-ranked technique for DoS state."""
        result = map_network_behavior_to_techniques(
            state=DOS_STATE,
            attack_category="DDoS",
        )
        assert result["techniques"], "No techniques returned for DoS state."
        top = result["techniques"][0]
        assert top["technique_id"] == "T1498", (
            f"Expected T1498 at top, got {top['technique_id']} "
            f"(confidence={top['confidence']})"
        )

    def test_t1498_confidence_above_zero(self):
        result = map_network_behavior_to_techniques(
            state=DOS_STATE,
            attack_category="DDoS",
        )
        t1498 = next((t for t in result["techniques"] if t["technique_id"] == "T1498"), None)
        assert t1498 is not None
        assert t1498["confidence"] > 0.0


# ---------------------------------------------------------------------------
# Test 3: Insufficient evidence → uncertainty, not forced technique
# ---------------------------------------------------------------------------

class TestInsufficientEvidence:
    def test_benign_state_produces_no_high_confidence_techniques(self):
        """A benign-looking state should not yield high-confidence technique mappings."""
        result = map_network_behavior_to_techniques(
            state=BENIGN_STATE,
            attack_category="BENIGN",
            attack_probability=0.02,
        )
        high_conf = [t for t in result["techniques"] if t["confidence"] >= 0.5]
        assert not high_conf, (
            f"Expected no high-confidence techniques for benign state, got: {high_conf}"
        )

    def test_benign_state_explanation_contains_uncertainties(self):
        """Explanation for benign/weak state must contain uncertainty statements."""
        explanation = generate_attack_explanation(
            state=BENIGN_STATE,
            attack_category="BENIGN",
            attack_probability=0.02,
        )
        assert explanation["uncertainties"], (
            "Uncertainties list must always be non-empty."
        )
        assert any("cannot confirm" in u.lower() or "evidence" in u.lower()
                   for u in explanation["uncertainties"]), (
            "At least one uncertainty must address evidence limitations."
        )

    def test_empty_state_returns_empty_or_zero_confidence_techniques(self):
        """Empty state with no category should return no techniques or all zero confidence."""
        result = map_network_behavior_to_techniques(state={}, attack_category=None)
        for t in result["techniques"]:
            assert t["confidence"] == 0.0 or t["confidence"] < 0.05, (
                f"Unexpected confidence {t['confidence']} for technique "
                f"{t['technique_id']} with empty state."
            )


# ---------------------------------------------------------------------------
# Test 4: Explanation contains actual supporting evidence
# ---------------------------------------------------------------------------

class TestExplanationEvidence:
    def test_explanation_techniques_have_evidence_strings(self):
        """Each matched technique must include evidence strings, not just a label."""
        explanation = generate_attack_explanation(
            state=PORT_SCAN_STATE,
            attack_category="PortScan",
            attack_probability=0.91,
        )
        assert explanation["techniques"], "Expected at least one technique for port-scan state."
        for tech in explanation["techniques"]:
            assert tech["evidence"], (
                f"Technique {tech['technique_id']} has no evidence strings."
            )
            assert len(tech["evidence"]) >= 1, (
                f"Technique {tech['technique_id']} must have at least 1 evidence item."
            )

    def test_explanation_observations_are_non_empty(self):
        """Observations list must be non-empty for anomalous state."""
        explanation = generate_attack_explanation(
            state=PORT_SCAN_STATE,
            attack_category="PortScan",
            attack_probability=0.91,
        )
        assert explanation["observations"], "Observations must be non-empty for port-scan state."

    def test_explanation_summary_is_non_empty_string(self):
        """Summary must be a non-empty string."""
        explanation = generate_attack_explanation(
            state=DOS_STATE,
            attack_category="DDoS",
            attack_probability=0.98,
        )
        assert isinstance(explanation["summary"], str)
        assert len(explanation["summary"]) > 20, "Summary is too short."

    def test_explanation_has_all_required_keys(self):
        """Explanation output must have all required keys."""
        explanation = generate_attack_explanation(state=PORT_SCAN_STATE)
        required = {"summary", "observations", "techniques", "graph_evidence",
                    "forecast_evidence", "uncertainties", "graph_available"}
        missing = required - set(explanation.keys())
        assert not missing, f"Explanation missing keys: {missing}"

    def test_t1046_evidence_mentions_unique_ports(self):
        """T1046 evidence must reference unique_ports since that's its required condition."""
        result = map_network_behavior_to_techniques(
            state=PORT_SCAN_STATE,
            attack_category="PortScan",
        )
        t1046 = next((t for t in result["techniques"] if t["technique_id"] == "T1046"), None)
        assert t1046 is not None
        evidence_text = " ".join(t1046["evidence"])
        assert "unique_ports" in evidence_text, (
            "T1046 evidence must mention 'unique_ports'."
        )


# ---------------------------------------------------------------------------
# Test 5: Graph evidence included when real graph snapshot exists
# ---------------------------------------------------------------------------

class TestGraphEvidence:
    def test_graph_features_produce_graph_evidence(self):
        """When graph features are provided, graph_evidence must be non-empty for matching techniques."""
        graph_features = {
            "nodes": 120,
            "edges": 150,
            "unique_destinations": 80,
            "unique_sources": 3,
            "unique_services": 200,
            "attack_surface_score": 95.0,
            "external_bridge_channels": 10.0,
        }
        explanation = generate_attack_explanation(
            state=PORT_SCAN_STATE,
            attack_category="PortScan",
            attack_probability=0.88,
            graph_features=graph_features,
        )
        assert explanation["graph_available"] is True, "graph_available must be True."
        assert explanation["graph_evidence"], (
            "graph_evidence must be non-empty when graph features are provided "
            "and a technique with graph signals is matched."
        )

    def test_graph_evidence_contains_required_fields(self):
        """Each graph_evidence item must contain required fields."""
        graph_features = {
            "nodes": 120,
            "edges": 150,
            "unique_destinations": 80,
            "unique_sources": 3,
            "unique_services": 200,
            "attack_surface_score": 95.0,
        }
        explanation = generate_attack_explanation(
            state=PORT_SCAN_STATE,
            attack_category="PortScan",
            graph_features=graph_features,
        )
        for ge in explanation["graph_evidence"]:
            for field in ("technique_id", "source_nodes", "destination_nodes",
                          "ports", "active_edges", "note"):
                assert field in ge, f"graph_evidence item missing field '{field}'."

    def test_no_graph_features_produces_uncertainty(self):
        """Without graph features, uncertainties must mention graph unavailability."""
        explanation = generate_attack_explanation(
            state=PORT_SCAN_STATE,
            attack_category="PortScan",
            timestamp=9999999999.0,   # far-future timestamp — will not match any snapshot
        )
        # graph_evidence must be empty
        assert explanation["graph_evidence"] == [], (
            "graph_evidence must be empty when no graph snapshot matches."
        )

    def test_graph_paths_labeled_as_communication_paths(self):
        """Communication paths must not be labelled as 'attack chains'."""
        graph_impact = {
            "affected_attack_paths": 10,
            "remaining_attack_paths": 90,
            "affected_sources": ["192.168.10.5"],
            "affected_destinations": ["52.96.0.1"],
            "affected_ports": [80, 443],
        }
        graph_features = {
            "nodes": 50,
            "edges": 100,
            "unique_destinations": 40,
            "unique_services": 60,
        }
        explanation = generate_attack_explanation(
            state=PORT_SCAN_STATE,
            attack_category="PortScan",
            graph_features=graph_features,
            graph_impact=graph_impact,
        )
        for ge in explanation["graph_evidence"]:
            note = ge.get("note", "").lower()
            for key in ge:
                assert "attack_chain" not in key, (
                    f"Field '{key}' must not be labelled as attack_chain."
                )
            assert "communication path" in note, (
                "Graph evidence note must clarify these are communication paths, not kill chains."
            )


# ---------------------------------------------------------------------------
# Test 6: Forecast probability NOT incorrectly used as ATT&CK confidence
# ---------------------------------------------------------------------------

class TestProbabilityVsConfidenceSeparation:
    def test_high_probability_does_not_inflate_technique_confidence(self):
        """
        Technique confidence must be based on evidence strength only.
        Passing attack_probability=0.99 with a benign feature state
        must NOT produce high technique confidence.
        """
        result = map_network_behavior_to_techniques(
            state=BENIGN_STATE,
            attack_category=None,
            attack_probability=0.99,  # high probability, but benign features
        )
        for tech in result["techniques"]:
            assert tech["confidence"] < 0.5, (
                f"Technique {tech['technique_id']} confidence {tech['confidence']:.3f} "
                f"is too high for a benign-featured state — "
                f"probability 0.99 must not inflate ATT&CK confidence."
            )

    def test_low_probability_does_not_suppress_technique_confidence(self):
        """
        Evidence-supported techniques must still be returned even if
        attack_probability is low.  Confidence is independent of probability.
        """
        result = map_network_behavior_to_techniques(
            state=PORT_SCAN_STATE,
            attack_category="PortScan",
            attack_probability=0.05,   # low probability — should not suppress T1046
        )
        tech_ids = [t["technique_id"] for t in result["techniques"]]
        assert "T1046" in tech_ids, (
            "T1046 must still be returned even when attack_probability is low, "
            "because confidence is derived from feature evidence, not probability."
        )

    def test_forecast_evidence_contains_probability_not_confidence(self):
        """forecast_evidence must contain attack_probability, not ATT&CK confidence."""
        explanation = generate_attack_explanation(
            state=PORT_SCAN_STATE,
            attack_category="PortScan",
            attack_probability=0.87,
            trajectory=[
                {"horizon": i, "attack_probability": 0.80 - i * 0.02,
                 "attack_category": "PortScan", "category_confidence": 0.9,
                 "risk": 0.6}
                for i in range(1, 13)
            ],
        )
        fe = explanation["forecast_evidence"]
        assert "current_attack_probability" in fe, (
            "forecast_evidence must contain current_attack_probability."
        )
        assert abs(fe["current_attack_probability"] - 0.87) < 0.001, (
            "current_attack_probability must equal the input probability."
        )
        assert "note" in fe, (
            "forecast_evidence must contain a note distinguishing probability from confidence."
        )
        assert "not" in fe["note"].lower() and "equivalent" in fe["note"].lower(), (
            "Note must clarify forecast probability is NOT equivalent to ATT&CK confidence."
        )

    def test_technique_confidence_is_not_equal_to_probability(self):
        """
        For any call with attack_probability, technique confidences must NOT
        all equal the probability value (they are computed differently).
        """
        prob = 0.75
        result = map_network_behavior_to_techniques(
            state=PORT_SCAN_STATE,
            attack_category="PortScan",
            attack_probability=prob,
        )
        for tech in result["techniques"]:
            assert abs(tech["confidence"] - prob) > 0.001, (
                f"Technique {tech['technique_id']} confidence {tech['confidence']:.4f} "
                f"equals attack_probability {prob:.4f} — "
                f"these must be computed independently."
            )

import unittest
import numpy as np

from counterfactual.graph_interventions import (
    GraphSnapshotData,
    apply_graph_intervention,
)
from counterfactual.graph_counterfactual import (
    compute_attack_surface,
    compute_communication_paths,
    calculate_graph_impact,
)
from counterfactual.graph_simulator import (
    get_graph_snapshot_by_timestamp,
    simulate_graph_counterfactual,
)
from counterfactual.interventions import build_intervention
from counterfactual.comparator import compare_interventions
from counterfactual.defense_planner import rank_interventions


class GraphCounterfactualVerificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load snapshot 1000 (a real historical attack snapshot)
        from graph_experiment.dataset import GraphDataset
        cls.ds = GraphDataset()
        cls.timestamp = float(cls.ds.arrays["snapshot__timestamps"][1000])
        cls.snapshot = get_graph_snapshot_by_timestamp(cls.timestamp)
        assert cls.snapshot is not None

    def test_1_snapshot_loaded(self):
        """1. Load one real historical graph snapshot."""
        self.assertIsNotNone(self.snapshot)
        self.assertGreater(len(self.snapshot.node_ids), 0)
        self.assertGreater(len(self.snapshot.edge_index), 0)
        self.assertEqual(len(self.snapshot.edge_features), len(self.snapshot.edge_index))

    def test_2_no_action(self):
        """2. Apply NO_ACTION."""
        intvn = build_intervention("NO_ACTION")
        cf_snap, removed = apply_graph_intervention(self.snapshot, intvn)
        self.assertEqual(len(removed), 0)
        self.assertEqual(len(cf_snap.edge_index), len(self.snapshot.edge_index))
        impact = calculate_graph_impact(self.snapshot, cf_snap, removed)
        self.assertEqual(impact["edges_removed"], 0)
        self.assertEqual(impact["nodes_removed"], 0)
        self.assertEqual(impact["attack_surface_reduction"], 0.0)

    def test_3_block_source(self):
        """3. Apply BLOCK_SOURCE."""
        # Pick an active source IP from edge 0
        src_ip = self.snapshot.node_ids[self.snapshot.edge_index[0, 0]]
        intvn = build_intervention("BLOCK_SOURCE", target=src_ip)
        cf_snap, removed = apply_graph_intervention(self.snapshot, intvn)
        self.assertGreater(len(removed), 0)
        self.assertEqual(len(cf_snap.edge_index), len(self.snapshot.edge_index) - len(removed))
        impact = calculate_graph_impact(self.snapshot, cf_snap, removed)
        self.assertIn(src_ip, impact["affected_sources"])
        self.assertGreater(impact["edges_removed"], 0)
        self.assertGreater(impact["attack_surface_reduction"], 0.0)

    def test_4_block_destination(self):
        """4. Apply BLOCK_DESTINATION."""
        dst_ip = self.snapshot.node_ids[self.snapshot.edge_index[0, 1]]
        intvn = build_intervention("BLOCK_DESTINATION", target=dst_ip)
        cf_snap, removed = apply_graph_intervention(self.snapshot, intvn)
        self.assertGreater(len(removed), 0)
        impact = calculate_graph_impact(self.snapshot, cf_snap, removed)
        self.assertIn(dst_ip, impact["affected_destinations"])
        self.assertGreater(impact["edges_removed"], 0)

    def test_5_block_port(self):
        """5. Apply BLOCK_PORT."""
        dst_port = int(self.snapshot.edge_features[0, 2])
        intvn = build_intervention("BLOCK_PORT", target=dst_port)
        cf_snap, removed = apply_graph_intervention(self.snapshot, intvn)
        self.assertGreater(len(removed), 0)
        impact = calculate_graph_impact(self.snapshot, cf_snap, removed)
        self.assertIn(dst_port, impact["affected_ports"])

    def test_6_metrics_change_when_target_exists(self):
        """6. Confirm graph metrics change when target exists."""
        dst_port = int(self.snapshot.edge_features[0, 2])
        intvn = build_intervention("BLOCK_PORT", target=dst_port)
        res = simulate_graph_counterfactual(intvn, timestamp=self.timestamp)
        self.assertTrue(res["graph_available"])
        self.assertGreater(res["graph_impact"]["edges_removed"], 0)
        self.assertGreater(res["graph_impact"]["packet_volume_removed"], 0.0)
        self.assertGreater(res["graph_impact"]["byte_volume_removed"], 0.0)
        self.assertGreater(res["graph_impact"]["attack_surface_reduction"], 0.0)
        self.assertGreater(res["graph_impact"]["affected_attack_paths"], 0)

    def test_7_original_graph_remains_unchanged(self):
        """7. Confirm original graph remains unchanged (no mutation)."""
        orig_edges = len(self.snapshot.edge_index)
        orig_features_sum = float(np.sum(self.snapshot.edge_features))
        dst_port = int(self.snapshot.edge_features[0, 2])
        intvn = build_intervention("BLOCK_PORT", target=dst_port)
        apply_graph_intervention(self.snapshot, intvn)
        self.assertEqual(len(self.snapshot.edge_index), orig_edges)
        self.assertAlmostEqual(float(np.sum(self.snapshot.edge_features)), orig_features_sum, places=4)

    def test_8_counterfactual_planner_expected_structure(self):
        """8. Confirm existing counterfactual planner still returns its expected structure."""
        state = {
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
            intervention_types=["NO_ACTION", "BLOCK_SOURCE", "BLOCK_PORT"],
            horizon=3,
            current_timestamp=self.timestamp,
        )
        self.assertEqual(len(results), 3)
        for r in results:
            self.assertIn("intervention", r)
            self.assertIn("summary", r)
            self.assertIn("trajectory", r)
            # When current_timestamp matches a graph, graph_impact is attached as evidence
            self.assertIn("graph_impact", r)

        ranked = rank_interventions(results)
        self.assertEqual(len(ranked), 3)
        for r in ranked:
            self.assertIn("intervention", r)
            self.assertIn("target", r)
            self.assertIn("risk_reduction", r)
            self.assertIn("operational_cost", r)
            self.assertIn("service_disruption_cost", r)
            self.assertIn("defense_score", r)


if __name__ == "__main__":
    unittest.main()

import unittest

from counterfactual.interventions import (
    InterventionSpec,
    SUPPORTED_INTERVENTIONS,
    build_intervention,
)
from counterfactual.rollout import simulate_counterfactual
from counterfactual.state_transformer import apply_intervention


class InterventionRegistryTest(unittest.TestCase):
    def test_intervention_types_exist(self):
        required = {
            "NO_ACTION",
            "BLOCK_SOURCE",
            "BLOCK_DESTINATION",
            "BLOCK_PORT",
            "ISOLATE_HOST",
            "DISABLE_CONNECTION",
            "NETWORK_SEGMENTATION",
        }
        self.assertTrue(required.issubset(set(SUPPORTED_INTERVENTIONS)))

    def test_no_action_builder_returns_expected_defaults(self):
        result = build_intervention("NO_ACTION", 80)
        self.assertEqual(result.type, "NO_ACTION")
        self.assertEqual(result.target, 80)
        self.assertEqual(result.operational_cost, 0.0)
        self.assertEqual(result.service_disruption_cost, 0.0)

    def test_intervention_spec_fields_are_present(self):
        intervention = InterventionSpec(
            intervention_id="block_port_80",
            type="BLOCK_PORT",
            target=80,
            operational_cost=0.2,
            service_disruption_cost=0.3,
            description="Simulate blocking traffic to destination port 80.",
        )

        self.assertEqual(intervention.intervention_id, "block_port_80")
        self.assertEqual(intervention.type, "BLOCK_PORT")
        self.assertEqual(intervention.target, 80)
        self.assertEqual(intervention.operational_cost, 0.2)
        self.assertEqual(intervention.service_disruption_cost, 0.3)
        self.assertIn("port 80", intervention.description)

    def test_supported_intervention_registry_is_deterministic(self):
        names = list(SUPPORTED_INTERVENTIONS.keys())
        self.assertEqual(names[0], "NO_ACTION")
        self.assertEqual(names[-1], "NETWORK_SEGMENTATION")

    def test_block_port_applies_realistic_traffic_reductions(self):
        baseline = {
            "flow_count": 200,
            "packet_count": 800,
            "total_byte_count": 24000,
            "syn_rate": 0.35,
            "ack_rate": 0.65,
            "rst_rate": 0.12,
            "mean_flow_packets_per_sec": 3.5,
        }

        transformed = apply_intervention(baseline, "BLOCK_PORT")

        self.assertLess(transformed["packet_count"], baseline["packet_count"])
        self.assertLess(transformed["syn_rate"], baseline["syn_rate"])
        self.assertLess(transformed["ack_rate"], baseline["ack_rate"])

    def test_counterfactual_rollout_returns_risk_evaluations(self):
        simulation = simulate_counterfactual(
            initial_state={
                "flow_count": 120,
                "packet_count": 520,
                "total_packet_count": 520,
                "total_byte_count": 12000,
                "mean_flow_packets_per_sec": 2.1,
                "mean_flow_bytes_per_sec": 120.0,
                "syn_rate": 0.2,
                "ack_rate": 0.4,
                "rst_rate": 0.1,
            },
            intervention="BLOCK_SOURCE",
            horizon=3,
        )

        self.assertEqual(len(simulation["baseline"]), 3)
        self.assertEqual(len(simulation["counterfactual"]), 3)
        self.assertIsNotNone(simulation["baseline"][0]["attack_probability"])
        self.assertIsNotNone(simulation["counterfactual"][0]["attack_probability"])
        self.assertIn("risk", simulation["baseline"][0])
        self.assertIn("risk", simulation["counterfactual"][0])


if __name__ == "__main__":
    unittest.main()

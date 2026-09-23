import unittest

from counterfactual.defense_planner import rank_interventions


class DefensePlannerTest(unittest.TestCase):
    def test_rank_interventions_nested_comparator_schema(self):
        # Three representative nested results from comparator
        result_low = {
            "intervention": {
                "id": "block_source_0",
                "type": "BLOCK_SOURCE",
                "target": "192.168.1.100",
            },
            "summary": {
                "risk_reduction": 0.35,
                "operational_cost": 0.15,
                "service_disruption_cost": 0.10,
                "defense_score": 0.10,
            },
            "trajectory": [],
        }

        result_high = {
            "intervention": {
                "id": "block_port_80",
                "type": "BLOCK_PORT",
                "target": 80,
            },
            "summary": {
                "risk_reduction": 0.85,
                "operational_cost": 0.20,
                "service_disruption_cost": 0.15,
                "defense_score": 0.50,
            },
            "trajectory": [],
        }

        result_medium = {
            "intervention": {
                "id": "isolate_host_0",
                "type": "ISOLATE_HOST",
                "target": "server-1",
            },
            "summary": {
                "risk_reduction": 0.60,
                "operational_cost": 0.25,
                "service_disruption_cost": 0.10,
                "defense_score": 0.25,
            },
            "trajectory": [],
        }

        # Input in arbitrary order: low, high, medium
        ranked = rank_interventions([result_low, result_high, result_medium])

        # 1. Verify sorting by nested defense_score descending (high -> medium -> low)
        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked[0]["defense_score"], 0.50)
        self.assertEqual(ranked[1]["defense_score"], 0.25)
        self.assertEqual(ranked[2]["defense_score"], 0.10)

        # 2. Verify complete intervention dictionary is preserved
        self.assertEqual(
            ranked[0]["intervention"],
            {
                "id": "block_port_80",
                "type": "BLOCK_PORT",
                "target": 80,
            },
        )
        self.assertEqual(ranked[1]["intervention"]["type"], "ISOLATE_HOST")
        self.assertEqual(ranked[2]["intervention"]["type"], "BLOCK_SOURCE")

        # 3. Verify target is extracted from item["intervention"]["target"]
        self.assertEqual(ranked[0]["target"], 80)
        self.assertEqual(ranked[1]["target"], "server-1")
        self.assertEqual(ranked[2]["target"], "192.168.1.100")

        # 4. Verify nested risk_reduction is returned correctly
        self.assertEqual(ranked[0]["risk_reduction"], 0.85)
        self.assertEqual(ranked[1]["risk_reduction"], 0.60)
        self.assertEqual(ranked[2]["risk_reduction"], 0.35)

        # 5. Verify nested costs are returned correctly
        self.assertEqual(ranked[0]["operational_cost"], 0.20)
        self.assertEqual(ranked[0]["service_disruption_cost"], 0.15)
        self.assertEqual(ranked[1]["operational_cost"], 0.25)
        self.assertEqual(ranked[1]["service_disruption_cost"], 0.10)
        self.assertEqual(ranked[2]["operational_cost"], 0.15)
        self.assertEqual(ranked[2]["service_disruption_cost"], 0.10)

    def test_rank_interventions_missing_required_fields_raises_key_error(self):
        # Verify that malformed entries without nested summary/defense_score raise KeyError
        malformed = [{"intervention": {"type": "NO_ACTION", "target": 0}}]
        with self.assertRaises(KeyError):
            rank_interventions(malformed)


if __name__ == "__main__":
    unittest.main()

import unittest

from counterfactual.comparator import compare_all_interventions, compare_interventions


class CounterfactualComparatorTest(unittest.TestCase):
    def test_basic_comparison(self):
        baseline = [
            {"risk": 0.8, "attack_probability": 0.75, "attack_category": "ATTACK", "category_confidence": 0.7},
            {"risk": 0.7, "attack_probability": 0.68, "attack_category": "ATTACK", "category_confidence": 0.65},
            {"risk": 0.6, "attack_probability": 0.62, "attack_category": "ATTACK", "category_confidence": 0.6},
        ]
        counterfactual = [
            {"risk": 0.6, "attack_probability": 0.56, "attack_category": "BENIGN", "category_confidence": 0.4},
            {"risk": 0.5, "attack_probability": 0.48, "attack_category": "BENIGN", "category_confidence": 0.35},
            {"risk": 0.4, "attack_probability": 0.42, "attack_category": "BENIGN", "category_confidence": 0.3},
        ]
        intervention = {
            "id": "block_port_80",
            "type": "BLOCK_PORT",
            "target": 80,
            "operational_cost": 0.2,
            "service_disruption_cost": 0.3,
        }

        result = compare_interventions(baseline, counterfactual, intervention)

        self.assertGreater(result["summary"]["risk_reduction"], 0.0)
        self.assertAlmostEqual(result["summary"]["baseline_cumulative_risk"], 2.1)
        self.assertAlmostEqual(result["summary"]["counterfactual_cumulative_risk"], 1.5)
        self.assertAlmostEqual(result["summary"]["baseline_average_risk"], 0.7)
        self.assertAlmostEqual(result["summary"]["counterfactual_average_risk"], 0.5)
        self.assertAlmostEqual(result["summary"]["baseline_peak_risk"], 0.8)
        self.assertAlmostEqual(result["summary"]["counterfactual_peak_risk"], 0.6)
        self.assertAlmostEqual(result["summary"]["baseline_final_risk"], 0.6)
        self.assertAlmostEqual(result["summary"]["counterfactual_final_risk"], 0.4)
        self.assertAlmostEqual(result["summary"]["risk_reduction_percentage"], 28.5714285714)
        self.assertEqual(result["intervention"]["type"], "BLOCK_PORT")

    def test_zero_baseline_risk_has_safe_percentage(self):
        baseline = [{"risk": 0.0}, {"risk": 0.0}]
        counterfactual = [{"risk": 0.1}, {"risk": 0.0}]
        intervention = {"id": "no_action", "type": "NO_ACTION", "target": 0, "operational_cost": 0.0, "service_disruption_cost": 0.0}

        result = compare_interventions(baseline, counterfactual, intervention)

        self.assertEqual(result["summary"]["risk_reduction_percentage"], 0.0)
        self.assertAlmostEqual(result["summary"]["risk_reduction"], -0.1)

    def test_negative_risk_reduction_is_preserved(self):
        baseline = [{"risk": 0.4}, {"risk": 0.4}]
        counterfactual = [{"risk": 0.6}, {"risk": 0.6}]
        intervention = {"id": "isolate_host", "type": "ISOLATE_HOST", "target": "host_7", "operational_cost": 0.05, "service_disruption_cost": 0.05}

        result = compare_interventions(baseline, counterfactual, intervention)

        self.assertAlmostEqual(result["summary"]["risk_reduction"], -0.4)
        self.assertLess(result["summary"]["risk_reduction"], 0.0)

    def test_intervention_costs_and_defense_score_are_preserved(self):
        baseline = [{"risk": 0.5}, {"risk": 0.5}]
        counterfactual = [{"risk": 0.3}, {"risk": 0.2}]
        intervention = {"id": "block_source", "type": "BLOCK_SOURCE", "target": "10.0.0.5", "operational_cost": 0.12, "service_disruption_cost": 0.18}

        result = compare_interventions(baseline, counterfactual, intervention)

        self.assertAlmostEqual(result["summary"]["operational_cost"], 0.12)
        self.assertAlmostEqual(result["summary"]["service_disruption_cost"], 0.18)
        self.assertAlmostEqual(result["summary"]["defense_score"], result["summary"]["risk_reduction"] - 0.12 - 0.18)

    def test_category_and_confidence_are_preserved(self):
        baseline = [{"risk": 0.7, "attack_category": "ATTACK", "category_confidence": 0.75}, {"risk": 0.6, "attack_category": "ATTACK", "category_confidence": 0.72}]
        counterfactual = [{"risk": 0.5, "attack_category": "BENIGN", "category_confidence": 0.41}, {"risk": 0.4, "attack_category": "BENIGN", "category_confidence": 0.33}]
        intervention = {"id": "block_destination", "type": "BLOCK_DESTINATION", "target": 443, "operational_cost": 0.1, "service_disruption_cost": 0.2}

        result = compare_interventions(baseline, counterfactual, intervention)

        self.assertEqual(result["trajectory"][0]["baseline_category"], "ATTACK")
        self.assertEqual(result["trajectory"][0]["counterfactual_category"], "BENIGN")
        self.assertAlmostEqual(result["trajectory"][0]["baseline_category_confidence"], 0.75)
        self.assertAlmostEqual(result["trajectory"][0]["counterfactual_category_confidence"], 0.41)

    def test_twelve_horizon_trajectory_returns_exactly_twelve_records(self):
        baseline = [{"risk": float(i) / 10.0} for i in range(1, 13)]
        counterfactual = [{"risk": max(0.0, float(i) / 12.0)} for i in range(1, 13)]
        intervention = {"id": "network_segmentation", "type": "NETWORK_SEGMENTATION", "target": "segment_b", "operational_cost": 0.4, "service_disruption_cost": 0.6}

        result = compare_interventions(baseline, counterfactual, intervention)

        self.assertEqual(len(result["trajectory"]), 12)

    def test_compare_all_interventions_returns_one_result_per_intervention(self):
        baseline = [{"risk": 0.9}, {"risk": 0.8}]
        counterfactuals = [
            [{"risk": 0.7}, {"risk": 0.5}],
            [{"risk": 0.6}, {"risk": 0.4}],
            [{"risk": 0.8}, {"risk": 0.7}],
        ]
        interventions = [
            {"id": "block_source", "type": "BLOCK_SOURCE", "target": "10.0.0.4", "operational_cost": 0.15, "service_disruption_cost": 0.1},
            {"id": "block_destination", "type": "BLOCK_DESTINATION", "target": 443, "operational_cost": 0.2, "service_disruption_cost": 0.15},
            {"id": "block_port", "type": "BLOCK_PORT", "target": 80, "operational_cost": 0.18, "service_disruption_cost": 0.12},
        ]

        results = compare_all_interventions(baseline, counterfactuals, interventions)

        self.assertEqual(len(results), 3)
        self.assertTrue(all("summary" in row for row in results))


if __name__ == "__main__":
    unittest.main()

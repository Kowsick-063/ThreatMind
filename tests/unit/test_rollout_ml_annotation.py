import math
import unittest

from counterfactual.rollout import simulate_counterfactual

VALID_CATEGORIES = {"BENIGN", "Bot", "PortScan"}

_DEFAULT_STATE = {
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


class RolloutMLAnnotationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = simulate_counterfactual(
            initial_state=_DEFAULT_STATE,
            intervention="BLOCK_SOURCE",
            horizon=12,
        )
        cls.baseline = cls.sim["baseline"]
        cls.counterfactual = cls.sim["counterfactual"]

    # 1. Exactly 12 trajectory points
    def test_baseline_has_twelve_points(self):
        self.assertEqual(len(self.baseline), 12)

    def test_counterfactual_has_twelve_points(self):
        self.assertEqual(len(self.counterfactual), 12)

    # 2. attack_probability is not near-zero (heuristic produced ~0 always)
    def test_baseline_probability_not_all_zero(self):
        max_prob = max(pt["attack_probability"] for pt in self.baseline)
        self.assertGreater(max_prob, 0.01,
            "All baseline probabilities are near zero — old heuristic may still be active")

    # 3. attack_probability in [0, 1] — no NaN, no Inf
    def test_baseline_probability_valid_range(self):
        for pt in self.baseline:
            p = pt["attack_probability"]
            self.assertTrue(math.isfinite(p), f"H{pt['horizon']} attack_probability={p}")
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_counterfactual_probability_valid_range(self):
        for pt in self.counterfactual:
            p = pt["attack_probability"]
            self.assertTrue(math.isfinite(p), f"H{pt['horizon']} attack_probability={p}")
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    # 4. attack_category is one of the three trained classes
    def test_baseline_category_from_trained_classes(self):
        for pt in self.baseline:
            self.assertIn(pt["attack_category"], VALID_CATEGORIES,
                f"H{pt['horizon']} category {pt['attack_category']!r} not in trained classes")

    def test_counterfactual_category_from_trained_classes(self):
        for pt in self.counterfactual:
            self.assertIn(pt["attack_category"], VALID_CATEGORIES,
                f"H{pt['horizon']} category {pt['attack_category']!r} not in trained classes")

    # 5. category_confidence in [0, 1] — no NaN, no Inf
    def test_baseline_confidence_valid_range(self):
        for pt in self.baseline:
            cf = pt["category_confidence"]
            self.assertTrue(math.isfinite(cf), f"H{pt['horizon']} category_confidence={cf}")
            self.assertGreaterEqual(cf, 0.0)
            self.assertLessEqual(cf, 1.0)

    def test_counterfactual_confidence_valid_range(self):
        for pt in self.counterfactual:
            cf = pt["category_confidence"]
            self.assertTrue(math.isfinite(cf), f"H{pt['horizon']} category_confidence={cf}")
            self.assertGreaterEqual(cf, 0.0)
            self.assertLessEqual(cf, 1.0)

    # 6. category_confidence is NOT a copy of attack_probability
    def test_confidence_independent_of_probability(self):
        identical = sum(
            abs(pt["attack_probability"] - pt["category_confidence"]) < 1e-9
            for pt in self.baseline
        )
        self.assertLess(identical, 12,
            "All category_confidence values equal attack_probability — ACE not used")

    # 7. risk in [0, 1] — no NaN, no Inf
    def test_baseline_risk_valid_range(self):
        for pt in self.baseline:
            r = pt["risk"]
            self.assertTrue(math.isfinite(r), f"H{pt['horizon']} risk={r}")
            self.assertGreaterEqual(r, 0.0)
            self.assertLessEqual(r, 1.0)

    # 8. Both legs have all required fields
    def test_baseline_required_fields(self):
        required = {"horizon", "risk", "attack_probability", "attack_category",
                    "category_confidence", "state"}
        for pt in self.baseline:
            missing = required - set(pt.keys())
            self.assertFalse(missing, f"H{pt['horizon']} missing fields: {missing}")

    def test_counterfactual_required_fields(self):
        required = {"horizon", "risk", "attack_probability", "attack_category",
                    "category_confidence", "state"}
        for pt in self.counterfactual:
            missing = required - set(pt.keys())
            self.assertFalse(missing, f"H{pt['horizon']} missing fields: {missing}")


if __name__ == "__main__":
    unittest.main()

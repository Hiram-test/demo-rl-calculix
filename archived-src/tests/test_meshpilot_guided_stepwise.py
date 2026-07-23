import unittest

from meshpilot_guided_stepwise_pso import (
    budgeted_stepwise_config,
    priority_guided_seed,
)
from meshpilot_pso import PSOConfig


class GuidedStepwiseTests(unittest.TestCase):
    def test_priority_seed_refines_strongest_hotspots_first(self):
        seed = priority_guided_seed((0.2, 1.0, 0.4, 0.8, 0.1, 0.6), 3)
        self.assertEqual(seed[1], 2)
        self.assertEqual(seed[3], 2)
        self.assertEqual(seed[4], 0)
        self.assertEqual(sorted(seed), [0, 0, 1, 1, 2, 2])

    def test_budget_is_reallocated_to_more_staircase_generations(self):
        base = PSOConfig(
            particles=8,
            iterations=8,
            max_unique_evaluations=32,
        )
        tuned = budgeted_stepwise_config(base, particles=4, iterations=12)
        self.assertEqual(tuned.particles, 4)
        self.assertEqual(tuned.iterations, 12)
        self.assertEqual(tuned.max_unique_evaluations, 32)


if __name__ == "__main__":
    unittest.main()

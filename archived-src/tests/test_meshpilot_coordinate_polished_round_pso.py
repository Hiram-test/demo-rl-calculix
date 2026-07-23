import unittest

from meshpilot_coordinate_polished_round_pso import CoordinatePolishedRoundedPSO
from meshpilot_pso import ObjectiveValue, PSOConfig


class CoordinatePolishedRoundedPSOTests(unittest.TestCase):
    def test_polish_uses_real_callback_and_respects_budget(self):
        optimum = (3, 1, 2, 0)
        calls = []

        def evaluator(position):
            calls.append(tuple(position))
            objective = float(sum((a - b) ** 2 for a, b in zip(position, optimum)))
            return ObjectiveValue(
                position=tuple(position),
                objective=objective,
                relative_error=objective,
                element_count=100,
                feasible=True,
            )

        coarse = evaluator((0, 0, 0, 0))
        calls.clear()
        config = PSOConfig(
            particles=8,
            iterations=8,
            max_level=3,
            max_unique_evaluations=28,
            seed=17,
            stagnation_iterations=8,
        )
        result = CoordinatePolishedRoundedPSO(
            config,
            swarm_particles=4,
            minimum_swarm_budget=12,
        ).optimize(
            evaluator,
            dimensions=4,
            initial_values=(coarse,),
            dimension_priority=(1.0, 0.8, 0.6, 0.4),
        )

        self.assertTrue(result.best.feasible)
        self.assertLessEqual(result.best.objective, coarse.objective)
        self.assertLessEqual(result.unique_evaluations, 28)
        self.assertEqual(len(set(calls)), len(calls))
        self.assertGreater(result.coordinate_evaluations, 0)
        self.assertLess(result.swarm_evaluation_budget, 28)

    def test_warm_start_does_not_remove_coarse_or_budget(self):
        optimum = (2, 2, 1)

        def evaluator(position):
            objective = float(sum((a - b) ** 2 for a, b in zip(position, optimum)))
            return ObjectiveValue(
                position=tuple(position),
                objective=objective,
                relative_error=objective,
                element_count=100,
                feasible=True,
            )

        coarse = evaluator((0, 0, 0))
        warm = evaluator((2, 1, 1))
        config = PSOConfig(
            particles=8,
            iterations=8,
            max_level=3,
            max_unique_evaluations=20,
            seed=19,
        )
        result = CoordinatePolishedRoundedPSO(config).optimize(
            evaluator,
            dimensions=3,
            warm_start=warm.position,
            initial_values=(coarse, warm),
            charged_initial_evaluations=1,
            dimension_priority=(1.0, 0.7, 0.5),
        )
        self.assertTrue(result.used_warm_start)
        self.assertLessEqual(result.unique_evaluations, 20)
        self.assertLessEqual(result.best.objective, warm.objective)


if __name__ == "__main__":
    unittest.main()

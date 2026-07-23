import unittest

from meshpilot_pso import (
    DiscretePSO,
    ObjectiveValue,
    PSOConfig,
    project_level_field,
    similarity_adaptive_config,
)


class MeshPilotPSOTests(unittest.TestCase):
    def test_cache_and_warm_start_reduce_expensive_calls(self):
        optimum = (2,) * 8

        def evaluator(position):
            objective = float(sum((value - 2) ** 2 for value in position))
            return ObjectiveValue(
                position=position,
                objective=objective,
                relative_error=objective,
                element_count=100,
                feasible=True,
            )

        base = PSOConfig(
            particles=10,
            iterations=8,
            max_level=3,
            stagnation_iterations=8,
            target_objective=0.0,
            seed=9,
        )
        cold = DiscretePSO(base).optimize(evaluator, dimensions=8)
        adapted = similarity_adaptive_config(base, normalized_distance=0.05)
        warm = DiscretePSO(adapted.config).optimize(
            evaluator,
            dimensions=8,
            warm_start=optimum,
        )
        self.assertEqual(warm.best.objective, 0.0)
        self.assertLess(warm.unique_evaluations, cold.unique_evaluations)
        self.assertTrue(warm.used_warm_start)

    def test_project_level_field(self):
        projected = project_level_field({2: 3, 5: 1}, [5, 2, 9])
        self.assertEqual(projected, (1, 3, 0))

    def test_similarity_budget_uses_full_budget_without_source(self):
        base = PSOConfig(particles=8, iterations=7)
        budget = similarity_adaptive_config(base, None)
        self.assertEqual(budget.config.particles, 8)
        self.assertEqual(budget.config.iterations, 7)


if __name__ == "__main__":
    unittest.main()

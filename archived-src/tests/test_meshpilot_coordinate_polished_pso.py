import unittest

from meshpilot_coordinate_polished_pso import CoordinatePolishedStepwisePSO
from meshpilot_pso import ObjectiveValue, PSOConfig


class CoordinatePolishedPSOTests(unittest.TestCase):
    def test_coordinate_polish_respects_budget_and_improves(self):
        optimum = (3, 1, 2)

        def evaluator(position):
            objective = float(sum((a - b) ** 2 for a, b in zip(position, optimum)))
            return ObjectiveValue(
                position=position,
                objective=objective,
                relative_error=objective,
                element_count=100,
                feasible=True,
            )

        coarse = evaluator((0, 0, 0))
        config = PSOConfig(
            particles=8,
            iterations=8,
            max_level=3,
            max_unique_evaluations=24,
            seed=5,
        )
        result = CoordinatePolishedStepwisePSO(config).optimize(
            evaluator,
            dimensions=3,
            initial_values=(coarse,),
            dimension_priority=(1.0, 0.5, 0.8),
        )
        self.assertLessEqual(result.best.objective, coarse.objective)
        self.assertLessEqual(result.unique_evaluations, 24)
        self.assertGreater(result.coordinate_evaluations, 0)
        self.assertTrue(result.best.feasible)


if __name__ == "__main__":
    unittest.main()

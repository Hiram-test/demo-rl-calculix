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
            max_unique_evaluations=50,
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
        base = PSOConfig(particles=8, iterations=7, max_unique_evaluations=32)
        budget = similarity_adaptive_config(base, None)
        self.assertEqual(budget.config.particles, 8)
        self.assertEqual(budget.config.iterations, 7)
        self.assertEqual(budget.config.max_unique_evaluations, 32)

    def test_feasible_solution_always_beats_lower_objective_infeasible_solution(self):
        def evaluator(position):
            feasible = position == (0, 0)
            return ObjectiveValue(
                position=position,
                objective=10.0 if feasible else 0.0,
                relative_error=10.0 if feasible else 0.0,
                element_count=100 if feasible else 120,
                feasible=feasible,
                constraint_violation=0.0 if feasible else 0.2,
            )

        config = PSOConfig(
            particles=4,
            iterations=3,
            max_level=1,
            stagnation_iterations=3,
            seed=4,
        )
        result = DiscretePSO(config).optimize(evaluator, dimensions=2)
        self.assertTrue(result.best.feasible)
        self.assertEqual(result.best.position, (0, 0))

    def test_warm_start_keeps_all_l0_baseline(self):
        evaluated = []

        def evaluator(position):
            evaluated.append(position)
            return ObjectiveValue(
                position=position,
                objective=float(sum(position)),
                relative_error=0.0,
                element_count=100,
                feasible=True,
            )

        config = PSOConfig(
            particles=4,
            iterations=1,
            max_level=3,
            refill_repeats=False,
            seed=3,
        )
        DiscretePSO(config).optimize(
            evaluator,
            dimensions=2,
            warm_start=(3, 2),
        )
        self.assertEqual(evaluated[0], (0, 0))
        self.assertEqual(evaluated[1], (3, 2))

    def test_duplicate_particles_are_refilled_until_space_is_exhausted(self):
        evaluated = []

        def evaluator(position):
            evaluated.append(position)
            return ObjectiveValue(
                position=position,
                objective=float(sum(position)),
                relative_error=0.0,
                element_count=100,
                feasible=True,
            )

        config = PSOConfig(
            particles=8,
            iterations=4,
            max_level=1,
            stagnation_iterations=4,
            refill_repeats=True,
            seed=1,
        )
        result = DiscretePSO(config).optimize(evaluator, dimensions=2)
        self.assertEqual(result.unique_evaluations, 4)
        self.assertEqual(len(set(evaluated)), 4)
        self.assertTrue(result.search_space_exhausted)
        self.assertGreater(result.duplicate_refills, 0)

    def test_real_evaluation_budget_is_a_hard_cap(self):
        def evaluator(position):
            return ObjectiveValue(
                position=position,
                objective=float(sum(position)),
                relative_error=0.0,
                element_count=100,
                feasible=True,
            )

        config = PSOConfig(
            particles=8,
            iterations=10,
            max_level=3,
            max_unique_evaluations=5,
            stagnation_iterations=10,
            seed=5,
        )
        result = DiscretePSO(config).optimize(evaluator, dimensions=4)
        self.assertEqual(result.unique_evaluations, 5)
        self.assertTrue(result.budget_exhausted)

    def test_free_preprocessed_value_is_reused_without_fe_call(self):
        calls = []
        coarse = ObjectiveValue(
            position=(0, 0),
            objective=1.0,
            relative_error=1.0,
            element_count=100,
            feasible=True,
        )

        def evaluator(position):
            calls.append(position)
            return ObjectiveValue(
                position=position,
                objective=2.0,
                relative_error=2.0,
                element_count=100,
                feasible=True,
            )

        config = PSOConfig(particles=2, iterations=1, max_level=1, seed=2)
        result = DiscretePSO(config).optimize(
            evaluator,
            dimensions=2,
            initial_values=(coarse,),
            charged_initial_evaluations=0,
        )
        self.assertEqual(result.best.position, (0, 0))
        self.assertNotIn((0, 0), calls)
        self.assertEqual(result.unique_evaluations, len(calls))


if __name__ == "__main__":
    unittest.main()

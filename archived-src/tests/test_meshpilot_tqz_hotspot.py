import unittest

from meshpilot_pso import ObjectiveValue, PSOConfig
from meshpilot_stepwise_pso import StepwiseDiscretePSO, one_step_neighbours
from meshpilot_tqz_backend import TQZCase, TQZMeshSpec
from meshpilot_tqz_hotspot_backend import (
    HotspotCandidate,
    HotspotCell,
    HotspotSpec,
    _compact_number,
    match_hotspot_levels,
    select_hotspot_candidates,
)
from meshpilot_tqz_hotspot_batch import HotspotBenchmarkRequest


class StepwisePSOTests(unittest.TestCase):
    def test_one_step_neighbours_never_jump_levels(self):
        neighbours = one_step_neighbours((0, 2, 3), max_level=3)
        self.assertEqual(len(neighbours), 4)
        for candidate in neighbours:
            differences = [abs(a - b) for a, b in zip(candidate, (0, 2, 3))]
            self.assertEqual(sum(differences), 1)
            self.assertEqual(max(differences), 1)

    def test_stepwise_optimizer_improves_real_callback_budget(self):
        optimum = (2, 2, 2)

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
            particles=6,
            iterations=8,
            max_level=3,
            max_unique_evaluations=30,
            seed=11,
        )
        result = StepwiseDiscretePSO(
            config,
            local_search_trigger=2,
            exploration_probability=0.10,
        ).optimize(
            evaluator,
            dimensions=3,
            initial_values=(coarse,),
            dimension_priority=(1.0, 0.8, 0.6),
        )
        self.assertLessEqual(result.best.objective, coarse.objective)
        self.assertLessEqual(result.unique_evaluations, 30)
        self.assertGreater(result.step_moves, 0)
        self.assertTrue(result.best.feasible)


class HotspotCandidateTests(unittest.TestCase):
    def _case(self):
        return TQZCase(
            case_id="test",
            bearing_model="TQZ",
            nominal_vertical_capacity_kN=4500.0,
            ag=0.15,
            A=900.0,
            B=620.0,
            C=730.0,
            D=300.0,
            H=165.0,
        )

    def test_near_zero_coordinates_are_compacted(self):
        self.assertEqual(_compact_number(-1.4e-12), "0")
        self.assertEqual(_compact_number(0.0), "0")
        self.assertNotEqual(_compact_number(12.5), "0")

    def test_adjacent_high_cells_are_merged_before_top_k(self):
        cells = (
            HotspotCell((0, 0, 0), (-700, -500, 0, -250, 120, 300), 10, 9, 10, 11, 10, 4, 1.0),
            HotspotCell((1, 0, 0), (-250, 0, 0, 250, 120, 300), 10, 8, 9, 10, 9, 3, 0.95),
            HotspotCell((2, 0, 0), (0, 250, 0, 250, 120, 300), 10, 7, 8, 9, 8, 2, 0.80),
            HotspotCell((3, 0, 0), (250, 700, 0, 250, 120, 300), 10, 6, 7, 8, 7, 2, 0.70),
            HotspotCell((0, 1, 0), (-700, -500, 250, 500, 120, 300), 10, 5, 6, 7, 6, 1, 0.60),
            HotspotCell((3, 1, 0), (250, 700, 250, 500, 120, 300), 10, 4, 5, 6, 5, 1, 0.50),
        )
        spec = HotspotSpec(
            grid_x=4,
            grid_y=2,
            grid_z=1,
            candidate_count=3,
            merge_ratio=0.90,
            max_cells_per_region=2,
        )
        candidates = select_hotspot_candidates(
            self._case(),
            TQZMeshSpec(),
            spec,
            cells,
        )
        self.assertEqual(len(candidates), 3)
        self.assertEqual(set(candidates[0].cells), {(0, 0, 0), (1, 0, 0)})

    def test_hotspot_transfer_matches_physical_normalized_location(self):
        def candidate(candidate_id, center, score=0.8):
            return HotspotCandidate(
                candidate_id=candidate_id,
                bounds=(0, 1, 0, 1, 0, 1),
                center_normalized=center,
                size_normalized=(0.2, 0.2, 0.2),
                score=score,
                stress_signal=10.0,
                contrast=2.0,
                element_count=20,
                cells=((candidate_id, 0, 0),),
            )

        source = (
            candidate(1, (0.2, 0.3, 0.8)),
            candidate(2, (0.8, 0.3, 0.8)),
        )
        target = (
            candidate(1, (0.79, 0.31, 0.8)),
            candidate(2, (0.21, 0.29, 0.8)),
        )
        warm, matches = match_hotspot_levels(
            source,
            (3, 1),
            target,
            max_cost=0.5,
        )
        self.assertEqual(warm, (1, 3))
        self.assertEqual(len(matches), 2)


class HotspotRequestTests(unittest.TestCase):
    def test_real_solver_hotspot_contract(self):
        request = HotspotBenchmarkRequest.from_json(
            "examples/meshpilot_tqz_hotspot_request.json"
        )
        self.assertEqual(len(request.family.cases), 6)
        self.assertEqual(request.hotspot.candidate_count, 6)
        self.assertEqual(
            request.stepwise.local_search_trigger,
            2,
        )
        self.assertEqual(request.family.pso.max_unique_evaluations, 32)


if __name__ == "__main__":
    unittest.main()

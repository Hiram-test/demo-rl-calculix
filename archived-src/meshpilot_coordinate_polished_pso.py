"""Coordinate-polished stepwise PSO under a fixed real-FE budget.

With six hotspots and four levels, a coordinate line contains only four points.
The optimizer reserves part of the FE budget, lets a guided stepwise swarm locate a
promising basin, then checks every still-unseen level along each hotspot coordinate
in hotspot-score order.  The final spare calls inspect one-step neighbours.

This is still a real black-box optimizer: the supplied evaluator is called for each
new point and no surrogate or analytic response approximation is used.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence

from meshpilot_guided_stepwise_pso import (
    GuidedStepwiseDiscretePSO,
    budgeted_stepwise_config,
)
from meshpilot_pso import ObjectiveValue, PSOConfig, Position, is_better
from meshpilot_stepwise_pso import (
    StepwiseHistoryEntry,
    StepwisePSOResult,
    one_step_neighbours,
)


@dataclass(frozen=True)
class CoordinatePolishedPSOResult(StepwisePSOResult):
    coordinate_evaluations: int
    coordinate_improvements: int
    swarm_evaluation_budget: int


class _SharedRealEvaluator:
    def __init__(
        self,
        evaluator: Callable[[Position], ObjectiveValue],
        initial_values: Optional[Iterable[ObjectiveValue]],
        charged_initial_evaluations: int,
    ) -> None:
        self.evaluator = evaluator
        self.values: Dict[Position, ObjectiveValue] = {}
        for value in initial_values or ():
            self.values[tuple(int(item) for item in value.position)] = value
        self.real_evaluations = int(charged_initial_evaluations)
        if self.real_evaluations < 0 or self.real_evaluations > len(self.values):
            raise ValueError("invalid charged_initial_evaluations")

    def __call__(self, position: Position) -> ObjectiveValue:
        key = tuple(int(value) for value in position)
        if key in self.values:
            return self.values[key]
        value = self.evaluator(key)
        self.values[key] = value
        self.real_evaluations += 1
        return value


class CoordinatePolishedStepwisePSO:
    def __init__(
        self,
        config: PSOConfig,
        *,
        local_search_trigger: int = 2,
        exploration_probability: float = 0.08,
        agreement_move_probability: float = 0.75,
        swarm_particles: int = 4,
        swarm_iterations: int = 12,
    ) -> None:
        self.config = config.validated()
        self.local_search_trigger = int(local_search_trigger)
        self.exploration_probability = float(exploration_probability)
        self.agreement_move_probability = float(agreement_move_probability)
        self.swarm_particles = int(swarm_particles)
        self.swarm_iterations = int(swarm_iterations)

    def optimize(
        self,
        evaluator: Callable[[Position], ObjectiveValue],
        dimensions: int,
        warm_start: Optional[Sequence[int]] = None,
        *,
        initial_values: Optional[Iterable[ObjectiveValue]] = None,
        charged_initial_evaluations: int = 0,
        dimension_priority: Optional[Sequence[float]] = None,
    ) -> CoordinatePolishedPSOResult:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        full_budget = int(
            self.config.max_unique_evaluations
            if self.config.max_unique_evaluations is not None
            else self.config.particles * self.config.iterations
        )
        coordinate_reserve = min(
            dimensions * self.config.max_level,
            max(0, full_budget - max(self.swarm_particles * 2, 8)),
        )
        swarm_budget = max(
            charged_initial_evaluations + 2,
            full_budget - coordinate_reserve,
        )
        swarm_budget = min(full_budget, swarm_budget)
        shared = _SharedRealEvaluator(
            evaluator,
            initial_values,
            charged_initial_evaluations,
        )
        tuned = budgeted_stepwise_config(
            self.config,
            particles=self.swarm_particles,
            iterations=self.swarm_iterations,
        )
        tuned = PSOConfig(
            **{
                **tuned.__dict__,
                "max_unique_evaluations": swarm_budget,
            }
        ).validated()
        swarm = GuidedStepwiseDiscretePSO(
            tuned,
            local_search_trigger=self.local_search_trigger,
            exploration_probability=self.exploration_probability,
            agreement_move_probability=self.agreement_move_probability,
        )
        started = time.perf_counter()
        swarm_result = swarm.optimize(
            shared,
            dimensions,
            warm_start,
            initial_values=initial_values,
            charged_initial_evaluations=charged_initial_evaluations,
            dimension_priority=dimension_priority,
        )

        best = swarm_result.best
        best_position = tuple(int(value) for value in best.position)
        coordinate_evaluations = 0
        coordinate_improvements = 0
        priorities = (
            [0.0] * dimensions
            if dimension_priority is None
            else [float(value) for value in dimension_priority]
        )
        order = sorted(range(dimensions), key=lambda index: (-priorities[index], index))

        for dimension in order:
            if shared.real_evaluations >= full_budget:
                break
            base = best_position
            candidates = []
            for level in range(self.config.max_level + 1):
                if level == base[dimension]:
                    continue
                candidate = list(base)
                candidate[dimension] = level
                key = tuple(candidate)
                if key not in shared.values:
                    candidates.append(key)
            dimension_best = best
            dimension_position = best_position
            for candidate in candidates:
                if shared.real_evaluations >= full_budget:
                    break
                before = shared.real_evaluations
                value = shared(candidate)
                coordinate_evaluations += shared.real_evaluations - before
                if is_better(value, dimension_best):
                    dimension_best = value
                    dimension_position = candidate
            if is_better(dimension_best, best):
                best = dimension_best
                best_position = dimension_position
                coordinate_improvements += 1

        if shared.real_evaluations < full_budget:
            for candidate in one_step_neighbours(
                best_position,
                self.config.max_level,
                dimension_priority=priorities,
            ):
                if shared.real_evaluations >= full_budget:
                    break
                if candidate in shared.values:
                    continue
                before = shared.real_evaluations
                value = shared(candidate)
                coordinate_evaluations += shared.real_evaluations - before
                if is_better(value, best):
                    best = value
                    best_position = candidate
                    coordinate_improvements += 1

        elapsed = time.perf_counter() - started
        history = list(swarm_result.history)
        history.append(
            StepwiseHistoryEntry(
                iteration=swarm_result.iterations_completed + 1,
                best_objective=float(best.objective),
                best_error=float(best.relative_error),
                best_elements=int(best.element_count),
                best_feasible=bool(best.feasible),
                unique_evaluations=shared.real_evaluations,
                duplicate_refills=swarm_result.duplicate_refills,
                local_neighbor_evaluations=(
                    swarm_result.local_neighbor_evaluations
                    + coordinate_evaluations
                ),
                step_moves=swarm_result.step_moves,
                evaluation_budget=full_budget,
            )
        )
        return CoordinatePolishedPSOResult(
            best=best,
            history=tuple(history),
            unique_evaluations=shared.real_evaluations,
            cache_hits=swarm_result.cache_hits,
            duplicate_refills=swarm_result.duplicate_refills,
            local_neighbor_evaluations=(
                swarm_result.local_neighbor_evaluations + coordinate_evaluations
            ),
            step_moves=swarm_result.step_moves,
            iterations_completed=swarm_result.iterations_completed + 1,
            wall_time_seconds=float(elapsed),
            used_warm_start=warm_start is not None,
            budget_exhausted=shared.real_evaluations >= full_budget,
            search_space_exhausted=False,
            local_optimum_confirmed=False,
            config=self.config,
            local_search_trigger=self.local_search_trigger,
            exploration_probability=self.exploration_probability,
            coordinate_evaluations=coordinate_evaluations,
            coordinate_improvements=coordinate_improvements,
            swarm_evaluation_budget=swarm_budget,
        )

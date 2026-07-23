"""Coordinate-polished rounded PSO for the six-hotspot, four-level search.

The standard rounded PSO is retained for global basin finding, but part of the
same real-FE budget is reserved for a deterministic coordinate sweep.  With only
four levels per hotspot, testing the remaining levels along one coordinate is
cheap and avoids spending the final calls on near-duplicate swarm motion.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from meshpilot_guided_stepwise_pso import priority_guided_seed
from meshpilot_pso import (
    DiscretePSO,
    ObjectiveValue,
    PSOConfig,
    PSOHistoryEntry,
    PSOResult,
    Position,
    is_better,
)


@dataclass(frozen=True)
class CoordinatePolishedRoundedResult(PSOResult):
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

    def __call__(self, position: Position) -> ObjectiveValue:
        key = tuple(int(value) for value in position)
        if key in self.values:
            return self.values[key]
        value = self.evaluator(key)
        self.values[key] = value
        self.real_evaluations += 1
        return value


class _GuidedRoundedPSO(DiscretePSO):
    def __init__(self, config: PSOConfig, guided_seed: Optional[Position]) -> None:
        super().__init__(config)
        self.guided_seed = guided_seed

    def _initial_positions(self, dimensions, warm_start, rng):
        positions = super()._initial_positions(dimensions, warm_start, rng)
        if self.guided_seed is None:
            return positions
        guided = np.asarray(self.guided_seed, dtype=np.float64)
        slot = 1 if warm_start is None else 2
        if slot < positions.shape[0]:
            positions[slot, :] = np.clip(guided, 0, self.config.max_level)
        return positions


class CoordinatePolishedRoundedPSO:
    def __init__(
        self,
        config: PSOConfig,
        *,
        swarm_particles: int = 4,
        minimum_swarm_budget: int = 12,
    ) -> None:
        self.config = config.validated()
        self.swarm_particles = int(swarm_particles)
        self.minimum_swarm_budget = int(minimum_swarm_budget)

    def optimize(
        self,
        evaluator: Callable[[Position], ObjectiveValue],
        dimensions: int,
        warm_start: Optional[Sequence[int]] = None,
        *,
        initial_values: Optional[Iterable[ObjectiveValue]] = None,
        charged_initial_evaluations: int = 0,
        dimension_priority: Optional[Sequence[float]] = None,
    ) -> CoordinatePolishedRoundedResult:
        full_budget = int(
            self.config.max_unique_evaluations
            if self.config.max_unique_evaluations is not None
            else self.config.particles * self.config.iterations
        )
        coordinate_reserve = min(
            dimensions * self.config.max_level,
            max(0, full_budget - self.minimum_swarm_budget),
        )
        swarm_budget = max(
            charged_initial_evaluations + 2,
            full_budget - coordinate_reserve,
        )
        swarm_budget = min(full_budget, swarm_budget)
        particle_count = max(2, min(self.swarm_particles, swarm_budget))
        tuned = replace(
            self.config,
            particles=particle_count,
            iterations=max(self.config.iterations, 12),
            max_unique_evaluations=swarm_budget,
        ).validated()
        priorities = (
            [0.0] * dimensions
            if dimension_priority is None
            else [float(value) for value in dimension_priority]
        )
        guided = priority_guided_seed(priorities, self.config.max_level)
        shared = _SharedRealEvaluator(
            evaluator,
            initial_values,
            charged_initial_evaluations,
        )
        started = time.perf_counter()
        swarm = _GuidedRoundedPSO(tuned, guided).optimize(
            shared,
            dimensions,
            warm_start,
            initial_values=initial_values,
            charged_initial_evaluations=charged_initial_evaluations,
        )
        best = swarm.best
        best_position = tuple(int(value) for value in best.position)
        coordinate_evaluations = 0
        coordinate_improvements = 0
        order = sorted(range(dimensions), key=lambda index: (-priorities[index], index))

        for dimension in order:
            if shared.real_evaluations >= full_budget:
                break
            base = best_position
            dimension_best = best
            dimension_position = best_position
            for level in range(self.config.max_level + 1):
                if level == base[dimension]:
                    continue
                candidate_list = list(base)
                candidate_list[dimension] = level
                candidate = tuple(candidate_list)
                if candidate in shared.values:
                    value = shared.values[candidate]
                else:
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

        history: List[PSOHistoryEntry] = list(swarm.history)
        history.append(
            PSOHistoryEntry(
                iteration=swarm.iterations_completed + 1,
                best_objective=float(best.objective),
                best_error=float(best.relative_error),
                best_elements=int(best.element_count),
                best_feasible=bool(best.feasible),
                unique_evaluations=shared.real_evaluations,
                cache_hits=swarm.cache_hits,
                duplicate_refills=swarm.duplicate_refills,
                evaluation_budget=full_budget,
            )
        )
        elapsed = time.perf_counter() - started
        return CoordinatePolishedRoundedResult(
            best=best,
            history=tuple(history),
            unique_evaluations=shared.real_evaluations,
            cache_hits=swarm.cache_hits,
            duplicate_refills=swarm.duplicate_refills,
            iterations_completed=swarm.iterations_completed + 1,
            wall_time_seconds=float(elapsed),
            used_warm_start=warm_start is not None,
            budget_exhausted=shared.real_evaluations >= full_budget,
            search_space_exhausted=False,
            config=self.config,
            coordinate_evaluations=coordinate_evaluations,
            coordinate_improvements=coordinate_improvements,
            swarm_evaluation_budget=swarm_budget,
        )

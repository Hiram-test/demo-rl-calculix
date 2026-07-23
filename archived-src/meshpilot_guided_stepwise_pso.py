"""Budget-aware guided variant of the stepwise discrete PSO.

A one-level-per-iteration optimizer needs more generations than a rounded PSO that
can jump several levels at once.  Under the same 32-call cap this variant uses a
smaller four-particle swarm, more iterations, and one deterministic seed derived
from the real coarse-hotspot ranking.  No objective approximation is introduced.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional, Sequence

import numpy as np

from meshpilot_pso import PSOConfig
from meshpilot_stepwise_pso import StepwiseDiscretePSO


def priority_guided_seed(
    dimension_priority: Sequence[float],
    max_level: int,
) -> tuple[int, ...]:
    """Assign L2/L1/L0 to the top/middle/bottom thirds of coarse hotspots."""

    values = [float(value) for value in dimension_priority]
    if not values:
        raise ValueError("dimension_priority cannot be empty")
    order = sorted(range(len(values)), key=lambda index: (-values[index], index))
    result = [0] * len(values)
    first_cut = max(1, int(np.ceil(len(values) / 3.0)))
    second_cut = max(first_cut + 1, int(np.ceil(2.0 * len(values) / 3.0)))
    high_level = min(2, int(max_level))
    middle_level = min(1, int(max_level))
    for rank, dimension in enumerate(order):
        if rank < first_cut:
            result[dimension] = high_level
        elif rank < second_cut:
            result[dimension] = middle_level
    return tuple(result)


def budgeted_stepwise_config(
    config: PSOConfig,
    *,
    particles: int = 4,
    iterations: int = 12,
) -> PSOConfig:
    """Reallocate a fixed FE budget from swarm width to staircase depth."""

    budget = config.max_unique_evaluations
    particle_count = max(2, int(particles))
    if budget is not None:
        particle_count = min(particle_count, int(budget))
    return replace(
        config,
        particles=particle_count,
        iterations=max(int(iterations), config.iterations),
    ).validated()


class GuidedStepwiseDiscretePSO(StepwiseDiscretePSO):
    """Stepwise PSO with a real-hotspot-priority seed in the initial swarm."""

    def __init__(self, config: PSOConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)
        self._guided_seed: Optional[tuple[int, ...]] = None

    def _initial_positions(
        self,
        dimensions: int,
        warm_start,
        rng: np.random.Generator,
    ) -> np.ndarray:
        positions = super()._initial_positions(dimensions, warm_start, rng)
        if self._guided_seed is None:
            return positions
        guided = np.asarray(self._guided_seed, dtype=np.int64)
        if guided.shape != (dimensions,):
            raise ValueError("guided seed shape differs from dimensions")
        slot = 1 if warm_start is None else 2
        if slot < positions.shape[0]:
            positions[slot, :] = np.clip(guided, 0, self.config.max_level)
        return positions

    def optimize(
        self,
        evaluator,
        dimensions: int,
        warm_start=None,
        *,
        initial_values=None,
        charged_initial_evaluations: int = 0,
        dimension_priority=None,
    ):
        self._guided_seed = (
            None
            if dimension_priority is None
            else priority_guided_seed(
                dimension_priority,
                self.config.max_level,
            )
        )
        return super().optimize(
            evaluator,
            dimensions,
            warm_start,
            initial_values=initial_values,
            charged_initial_evaluations=charged_initial_evaluations,
            dimension_priority=dimension_priority,
        )

"""Stepwise discrete PSO for expensive L0-Lk mesh-level searches.

The ordinary MeshPilot baseline updates a continuous position and rounds it back to
integer mesh levels.  This module works directly on the staircase: every dimension
can move by at most one level per iteration.  When the swarm stalls, it evaluates
the still-unseen one-step neighbours of the current global best before declaring a
local finish.  All objective calls remain real calls supplied by the caller; this
module contains no surrogate or simulated evaluator.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
import time
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from meshpilot_pso import (
    CachedObjective,
    ObjectiveValue,
    PSOConfig,
    Position,
    is_better,
    nearest_unseen_position,
)


@dataclass(frozen=True)
class StepwiseHistoryEntry:
    iteration: int
    best_objective: float
    best_error: float
    best_elements: int
    best_feasible: bool
    unique_evaluations: int
    duplicate_refills: int
    local_neighbor_evaluations: int
    step_moves: int
    evaluation_budget: Optional[int]


@dataclass(frozen=True)
class StepwisePSOResult:
    best: ObjectiveValue
    history: Tuple[StepwiseHistoryEntry, ...]
    unique_evaluations: int
    cache_hits: int
    duplicate_refills: int
    local_neighbor_evaluations: int
    step_moves: int
    iterations_completed: int
    wall_time_seconds: float
    used_warm_start: bool
    budget_exhausted: bool
    search_space_exhausted: bool
    local_optimum_confirmed: bool
    config: PSOConfig
    local_search_trigger: int
    exploration_probability: float


def one_step_neighbours(
    position: Sequence[int],
    max_level: int,
    dimension_priority: Optional[Sequence[float]] = None,
) -> Tuple[Position, ...]:
    """Return all valid positions differing by exactly one level in one dimension."""

    base = tuple(int(value) for value in position)
    if not base:
        raise ValueError("position cannot be empty")
    if max_level < 1:
        raise ValueError("max_level must be positive")
    if dimension_priority is None:
        priority = [0.0] * len(base)
    else:
        priority = [float(value) for value in dimension_priority]
        if len(priority) != len(base):
            raise ValueError("dimension_priority length must match position")

    candidates = []
    for dimension, value in enumerate(base):
        for delta in (-1, 1):
            next_value = value + delta
            if 0 <= next_value <= max_level:
                candidate = list(base)
                candidate[dimension] = next_value
                candidates.append(
                    (
                        -priority[dimension],
                        dimension,
                        0 if delta > 0 else 1,
                        tuple(candidate),
                    )
                )
    candidates.sort()
    return tuple(item[-1] for item in candidates)


def _mutate_integer_seed(
    seed: np.ndarray,
    rng: np.random.Generator,
    max_level: int,
    mutation_probability: float,
) -> np.ndarray:
    result = seed.astype(np.int64, copy=True)
    for index in range(result.shape[0]):
        if rng.random() < mutation_probability:
            result[index] += int(rng.choice((-1, 1)))
    return np.clip(result, 0, max_level).astype(np.int64)


class StepwiseDiscretePSO:
    """PSO whose positions live directly on the integer mesh-level staircase."""

    def __init__(
        self,
        config: PSOConfig,
        *,
        local_search_trigger: int = 2,
        exploration_probability: float = 0.08,
        agreement_move_probability: float = 0.75,
    ) -> None:
        self.config = config.validated()
        self.local_search_trigger = int(local_search_trigger)
        self.exploration_probability = float(exploration_probability)
        self.agreement_move_probability = float(agreement_move_probability)
        if self.local_search_trigger < 1:
            raise ValueError("local_search_trigger must be positive")
        if not 0.0 <= self.exploration_probability <= 1.0:
            raise ValueError("exploration_probability must lie in [0, 1]")
        if not 0.0 <= self.agreement_move_probability <= 1.0:
            raise ValueError("agreement_move_probability must lie in [0, 1]")

    def _initial_positions(
        self,
        dimensions: int,
        warm_start: Optional[Sequence[int]],
        rng: np.random.Generator,
    ) -> np.ndarray:
        cfg = self.config
        positions = rng.integers(
            0,
            cfg.max_level + 1,
            size=(cfg.particles, dimensions),
            dtype=np.int64,
        )
        positions[0, :] = 0
        if warm_start is None:
            return positions

        warm = np.asarray(list(warm_start), dtype=np.int64)
        if warm.shape != (dimensions,):
            raise ValueError(
                f"warm_start has shape {warm.shape}, expected {(dimensions,)}"
            )
        warm = np.clip(warm, 0, cfg.max_level)
        positions[1, :] = warm
        transfer_slots = max(1, int(math.ceil(cfg.particles * cfg.transfer_fraction)))
        transfer_slots = min(transfer_slots, cfg.particles - 1)
        mutation_probability = max(1.0 / max(dimensions, 1), 0.20)
        for offset in range(1, transfer_slots):
            positions[1 + offset, :] = _mutate_integer_seed(
                warm,
                rng,
                cfg.max_level,
                mutation_probability,
            )
        return positions

    def optimize(
        self,
        evaluator: Callable[[Position], ObjectiveValue],
        dimensions: int,
        warm_start: Optional[Sequence[int]] = None,
        *,
        initial_values: Optional[Iterable[ObjectiveValue]] = None,
        charged_initial_evaluations: int = 0,
        dimension_priority: Optional[Sequence[float]] = None,
    ) -> StepwisePSOResult:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        if dimension_priority is not None and len(dimension_priority) != dimensions:
            raise ValueError("dimension_priority length must equal dimensions")

        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        random.seed(cfg.seed)
        cached = CachedObjective(
            evaluator,
            initial_values=initial_values,
            charged_initial_evaluations=charged_initial_evaluations,
        )
        if (
            cfg.max_unique_evaluations is not None
            and cached.unique_evaluations > cfg.max_unique_evaluations
        ):
            raise ValueError("charged initial evaluations exceed max_unique_evaluations")

        positions = self._initial_positions(dimensions, warm_start, rng)
        direction_memory = rng.uniform(-0.20, 0.20, size=(cfg.particles, dimensions))
        pbest_positions = positions.copy()
        pbest_values: List[Optional[ObjectiveValue]] = [None] * cfg.particles

        global_best: Optional[ObjectiveValue] = None
        global_best_position: Optional[np.ndarray] = None
        for value in cached.values.values():
            if len(value.position) != dimensions:
                raise ValueError("initial objective position has the wrong dimension")
            if is_better(value, global_best):
                global_best = value
                global_best_position = np.asarray(value.position, dtype=np.int64)

        history: List[StepwiseHistoryEntry] = []
        stagnation = 0
        duplicate_refills = 0
        local_neighbor_evaluations = 0
        step_moves = 0
        budget_exhausted = False
        search_space_exhausted = False
        local_optimum_confirmed = False
        started = time.perf_counter()

        for iteration in range(cfg.iterations):
            if (
                global_best is not None
                and global_best.feasible
                and cfg.target_objective is not None
                and global_best.objective <= cfg.target_objective
            ):
                break

            improved_this_iteration = False
            for particle in range(cfg.particles):
                if (
                    cfg.max_unique_evaluations is not None
                    and cached.unique_evaluations >= cfg.max_unique_evaluations
                ):
                    budget_exhausted = True
                    break

                anchor = positions[particle].astype(np.float64)
                discrete = tuple(int(value) for value in positions[particle].tolist())
                if cfg.refill_repeats and discrete in cached.values:
                    replacement = nearest_unseen_position(
                        anchor,
                        cfg.max_level,
                        cached.values.keys(),
                    )
                    if replacement is None:
                        search_space_exhausted = True
                        break
                    discrete = replacement
                    positions[particle, :] = np.asarray(discrete, dtype=np.int64)
                    duplicate_refills += 1

                value = cached(discrete)
                if is_better(value, pbest_values[particle]):
                    pbest_values[particle] = value
                    pbest_positions[particle, :] = np.asarray(
                        discrete, dtype=np.int64
                    )
                if is_better(value, global_best):
                    global_best = value
                    global_best_position = np.asarray(discrete, dtype=np.int64)
                    improved_this_iteration = True

            if global_best is None or global_best_position is None:
                raise RuntimeError("stepwise PSO produced no objective value")

            if improved_this_iteration:
                stagnation = 0
            else:
                stagnation += 1

            if (
                not budget_exhausted
                and not search_space_exhausted
                and stagnation >= self.local_search_trigger
            ):
                before_ring = global_best
                neighbours = one_step_neighbours(
                    global_best_position.tolist(),
                    cfg.max_level,
                    dimension_priority=dimension_priority,
                )
                unseen = [
                    candidate for candidate in neighbours if candidate not in cached.values
                ]
                if not unseen:
                    local_optimum_confirmed = True
                    search_space_exhausted = True
                else:
                    ring_completed = True
                    for candidate in unseen:
                        if (
                            cfg.max_unique_evaluations is not None
                            and cached.unique_evaluations >= cfg.max_unique_evaluations
                        ):
                            budget_exhausted = True
                            ring_completed = False
                            break
                        value = cached(candidate)
                        local_neighbor_evaluations += 1
                        if is_better(value, global_best):
                            global_best = value
                            global_best_position = np.asarray(
                                candidate, dtype=np.int64
                            )
                    if is_better(global_best, before_ring):
                        stagnation = 0
                        positions[0, :] = global_best_position
                        improved_this_iteration = True
                    elif ring_completed:
                        local_optimum_confirmed = True

            history.append(
                StepwiseHistoryEntry(
                    iteration=iteration + 1,
                    best_objective=float(global_best.objective),
                    best_error=float(global_best.relative_error),
                    best_elements=int(global_best.element_count),
                    best_feasible=bool(global_best.feasible),
                    unique_evaluations=cached.unique_evaluations,
                    duplicate_refills=duplicate_refills,
                    local_neighbor_evaluations=local_neighbor_evaluations,
                    step_moves=step_moves,
                    evaluation_budget=cfg.max_unique_evaluations,
                )
            )

            if (
                global_best.feasible
                and cfg.target_objective is not None
                and global_best.objective <= cfg.target_objective
            ):
                break
            if budget_exhausted or search_space_exhausted or local_optimum_confirmed:
                break

            progress = iteration / max(cfg.iterations - 1, 1)
            inertia = cfg.inertia_start + progress * (
                cfg.inertia_end - cfg.inertia_start
            )
            for particle in range(cfg.particles):
                personal_target = (
                    pbest_positions[particle]
                    if pbest_values[particle] is not None
                    else positions[particle]
                )
                personal_direction = np.sign(
                    personal_target - positions[particle]
                ).astype(np.int64)
                global_direction = np.sign(
                    global_best_position - positions[particle]
                ).astype(np.int64)
                r1 = rng.random(dimensions)
                r2 = rng.random(dimensions)
                direction_memory[particle] = (
                    inertia * direction_memory[particle]
                    + cfg.cognitive * r1 * personal_direction
                    + cfg.social * r2 * global_direction
                )
                direction_memory[particle] = np.clip(
                    direction_memory[particle],
                    -cfg.velocity_limit,
                    cfg.velocity_limit,
                )

                for dimension in range(dimensions):
                    step = 0
                    if rng.random() < self.exploration_probability:
                        step = int(rng.choice((-1, 1)))
                    else:
                        memory = float(direction_memory[particle, dimension])
                        probability = abs(memory) / (1.0 + abs(memory))
                        same_nonzero = (
                            personal_direction[dimension]
                            == global_direction[dimension]
                            and personal_direction[dimension] != 0
                        )
                        conflicting = (
                            personal_direction[dimension]
                            * global_direction[dimension]
                            < 0
                        )
                        if same_nonzero:
                            probability = max(
                                probability,
                                self.agreement_move_probability,
                            )
                        elif conflicting:
                            probability *= 0.5
                        if rng.random() < probability:
                            if memory > 0.0:
                                step = 1
                            elif memory < 0.0:
                                step = -1
                            elif global_direction[dimension] != 0:
                                step = int(global_direction[dimension])
                            else:
                                step = int(personal_direction[dimension])
                    current = int(positions[particle, dimension])
                    updated = int(np.clip(current + step, 0, cfg.max_level))
                    if updated != current:
                        step_moves += 1
                    positions[particle, dimension] = updated

        elapsed = time.perf_counter() - started
        if global_best is None:
            raise RuntimeError("stepwise PSO produced no objective value")
        return StepwisePSOResult(
            best=global_best,
            history=tuple(history),
            unique_evaluations=cached.unique_evaluations,
            cache_hits=cached.cache_hits,
            duplicate_refills=duplicate_refills,
            local_neighbor_evaluations=local_neighbor_evaluations,
            step_moves=step_moves,
            iterations_completed=len(history),
            wall_time_seconds=float(elapsed),
            used_warm_start=warm_start is not None,
            budget_exhausted=budget_exhausted,
            search_space_exhausted=search_space_exhausted,
            local_optimum_confirmed=local_optimum_confirmed,
            config=cfg,
            local_search_trigger=self.local_search_trigger,
            exploration_probability=self.exploration_probability,
        )

"""Discrete PSO primitives for MeshPilot batch mesh optimization.

The optimizer is deliberately solver-agnostic.  A particle is a short vector of
integer mesh levels (L0..L3) associated with hotspot candidate regions.  The
expensive objective is supplied by the finite-element backend.

The batch variant is not a new velocity equation.  Its improvement is more
practical for expensive FEA: reuse the best level field from a nearby solved
case, seed part of the next swarm around it, and reduce swarm/iteration budgets
when the new case is similar.  The paper-level claim is therefore measured in
real FE calls and wall-clock time, not only in PSO arithmetic.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import random
import time
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


Position = Tuple[int, ...]


@dataclass(frozen=True)
class ObjectiveValue:
    """One expensive objective evaluation."""

    position: Position
    objective: float
    relative_error: float
    element_count: int
    feasible: bool
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PSOConfig:
    particles: int = 8
    iterations: int = 8
    max_level: int = 3
    inertia_start: float = 0.85
    inertia_end: float = 0.35
    cognitive: float = 1.55
    social: float = 1.55
    velocity_limit: float = 2.0
    stagnation_iterations: int = 4
    target_objective: Optional[float] = None
    transfer_fraction: float = 0.50
    seed: int = 7

    def validated(self) -> "PSOConfig":
        if self.particles < 2:
            raise ValueError("particles must be at least 2")
        if self.iterations < 1:
            raise ValueError("iterations must be positive")
        if self.max_level < 1:
            raise ValueError("max_level must be positive")
        if not 0.0 <= self.transfer_fraction <= 1.0:
            raise ValueError("transfer_fraction must lie in [0, 1]")
        if self.velocity_limit <= 0:
            raise ValueError("velocity_limit must be positive")
        return self


@dataclass(frozen=True)
class PSOHistoryEntry:
    iteration: int
    best_objective: float
    best_error: float
    best_elements: int
    unique_evaluations: int
    cache_hits: int


@dataclass(frozen=True)
class PSOResult:
    best: ObjectiveValue
    history: Tuple[PSOHistoryEntry, ...]
    unique_evaluations: int
    cache_hits: int
    iterations_completed: int
    wall_time_seconds: float
    used_warm_start: bool
    config: PSOConfig


class CachedObjective:
    """Exact per-case cache around an expensive evaluator."""

    def __init__(self, evaluator: Callable[[Position], ObjectiveValue]) -> None:
        self._evaluator = evaluator
        self._cache: Dict[Position, ObjectiveValue] = {}
        self.cache_hits = 0

    def __call__(self, position: Sequence[int]) -> ObjectiveValue:
        key = tuple(int(value) for value in position)
        if key in self._cache:
            self.cache_hits += 1
            return self._cache[key]
        value = self._evaluator(key)
        if value.position != key:
            value = replace(value, position=key)
        self._cache[key] = value
        return value

    @property
    def unique_evaluations(self) -> int:
        return len(self._cache)

    @property
    def values(self) -> Mapping[Position, ObjectiveValue]:
        return self._cache


def _clip_round(values: np.ndarray, max_level: int) -> Position:
    clipped = np.clip(np.rint(values), 0, max_level).astype(np.int64)
    return tuple(int(value) for value in clipped.tolist())


def _mutate_seed(
    seed: np.ndarray,
    rng: np.random.Generator,
    max_level: int,
    mutation_probability: float,
) -> np.ndarray:
    result = seed.astype(np.float64, copy=True)
    for index in range(result.shape[0]):
        if rng.random() < mutation_probability:
            result[index] += rng.choice((-1.0, 1.0))
    return np.clip(result, 0, max_level)


class DiscretePSO:
    """Cold-start or warm-start discrete PSO over L0..Lk mesh levels."""

    def __init__(self, config: PSOConfig) -> None:
        self.config = config.validated()

    def _initial_positions(
        self,
        dimensions: int,
        warm_start: Optional[Sequence[int]],
        rng: np.random.Generator,
    ) -> np.ndarray:
        cfg = self.config
        positions = rng.integers(
            low=0,
            high=cfg.max_level + 1,
            size=(cfg.particles, dimensions),
        ).astype(np.float64)

        # Always include the all-L0 configuration.  It is the meaningful coarse
        # baseline and prevents a small swarm from missing the low-cost corner.
        positions[0, :] = 0.0

        if warm_start is None:
            return positions

        warm = np.asarray(list(warm_start), dtype=np.float64)
        if warm.shape != (dimensions,):
            raise ValueError(
                f"warm_start has shape {warm.shape}, expected {(dimensions,)}"
            )
        warm = np.clip(warm, 0, cfg.max_level)
        transfer_count = max(1, int(math.ceil(cfg.particles * cfg.transfer_fraction)))
        transfer_count = min(transfer_count, cfg.particles)
        positions[0, :] = warm
        mutation_probability = max(1.0 / max(dimensions, 1), 0.20)
        for particle in range(1, transfer_count):
            positions[particle, :] = _mutate_seed(
                warm,
                rng,
                cfg.max_level,
                mutation_probability=mutation_probability,
            )
        return positions

    def optimize(
        self,
        evaluator: Callable[[Position], ObjectiveValue],
        dimensions: int,
        warm_start: Optional[Sequence[int]] = None,
    ) -> PSOResult:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)
        random.seed(cfg.seed)
        cached = CachedObjective(evaluator)
        positions = self._initial_positions(dimensions, warm_start, rng)
        velocities = rng.uniform(
            -0.25,
            0.25,
            size=(cfg.particles, dimensions),
        )

        pbest_positions = positions.copy()
        pbest_values: List[Optional[ObjectiveValue]] = [None] * cfg.particles
        global_best: Optional[ObjectiveValue] = None
        global_best_position: Optional[np.ndarray] = None
        history: List[PSOHistoryEntry] = []
        stagnation = 0
        started = time.perf_counter()

        for iteration in range(cfg.iterations):
            improved_this_iteration = False
            for particle in range(cfg.particles):
                discrete = _clip_round(positions[particle], cfg.max_level)
                value = cached(discrete)
                previous = pbest_values[particle]
                if previous is None or value.objective < previous.objective:
                    pbest_values[particle] = value
                    pbest_positions[particle, :] = np.asarray(discrete, dtype=np.float64)
                if global_best is None or value.objective < global_best.objective:
                    global_best = value
                    global_best_position = np.asarray(discrete, dtype=np.float64)
                    improved_this_iteration = True

            assert global_best is not None
            assert global_best_position is not None
            history.append(
                PSOHistoryEntry(
                    iteration=iteration + 1,
                    best_objective=float(global_best.objective),
                    best_error=float(global_best.relative_error),
                    best_elements=int(global_best.element_count),
                    unique_evaluations=cached.unique_evaluations,
                    cache_hits=cached.cache_hits,
                )
            )

            if cfg.target_objective is not None and global_best.objective <= cfg.target_objective:
                break
            if improved_this_iteration:
                stagnation = 0
            else:
                stagnation += 1
                if stagnation >= cfg.stagnation_iterations:
                    break

            progress = iteration / max(cfg.iterations - 1, 1)
            inertia = cfg.inertia_start + progress * (cfg.inertia_end - cfg.inertia_start)
            for particle in range(cfg.particles):
                r1 = rng.random(dimensions)
                r2 = rng.random(dimensions)
                cognitive = cfg.cognitive * r1 * (
                    pbest_positions[particle] - positions[particle]
                )
                social = cfg.social * r2 * (global_best_position - positions[particle])
                velocities[particle] = inertia * velocities[particle] + cognitive + social
                velocities[particle] = np.clip(
                    velocities[particle],
                    -cfg.velocity_limit,
                    cfg.velocity_limit,
                )
                positions[particle] = np.clip(
                    positions[particle] + velocities[particle],
                    0,
                    cfg.max_level,
                )

        elapsed = time.perf_counter() - started
        assert global_best is not None
        return PSOResult(
            best=global_best,
            history=tuple(history),
            unique_evaluations=cached.unique_evaluations,
            cache_hits=cached.cache_hits,
            iterations_completed=len(history),
            wall_time_seconds=float(elapsed),
            used_warm_start=warm_start is not None,
            config=cfg,
        )


@dataclass(frozen=True)
class TransferBudget:
    """Similarity-adaptive swarm budget used for later batch cases."""

    config: PSOConfig
    similarity: float
    source_distance: Optional[float]


def similarity_adaptive_config(
    base: PSOConfig,
    normalized_distance: Optional[float],
    *,
    min_particles: int = 4,
    min_iterations: int = 3,
    distance_scale: float = 0.75,
) -> TransferBudget:
    """Reduce the expensive search budget only when a nearby solved case exists.

    ``similarity = exp(-distance / distance_scale)``.  A very similar case uses a
    compact swarm and fewer iterations; an unrelated or first case uses the full
    cold-start budget.  This is the algorithmic bridge between batch use and PSO
    acceleration.
    """

    base = base.validated()
    if normalized_distance is None:
        return TransferBudget(base, similarity=0.0, source_distance=None)
    distance = max(0.0, float(normalized_distance))
    similarity = math.exp(-distance / max(distance_scale, 1.0e-12))
    particles = int(round(base.particles - similarity * (base.particles - min_particles)))
    iterations = int(round(base.iterations - similarity * (base.iterations - min_iterations)))
    particles = max(2, min(base.particles, particles))
    iterations = max(1, min(base.iterations, iterations))
    transfer_fraction = min(0.80, max(base.transfer_fraction, 0.40 + 0.40 * similarity))
    adapted = replace(
        base,
        particles=particles,
        iterations=iterations,
        transfer_fraction=transfer_fraction,
    )
    return TransferBudget(adapted.validated(), similarity=similarity, source_distance=distance)


def project_level_field(
    source_levels: Mapping[int, int],
    target_candidate_cells: Sequence[int],
    default_level: int = 0,
) -> Position:
    """Project a prior full patch-level solution onto a new hotspot candidate set."""

    return tuple(
        int(source_levels.get(int(cell_id), default_level))
        for cell_id in target_candidate_cells
    )

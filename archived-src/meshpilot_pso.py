"""Discrete PSO primitives for MeshPilot batch mesh optimization.

A particle is a short vector of integer mesh levels (L0..L3) associated with
hotspot candidate regions.  The expensive objective is supplied by a finite-
element backend.

The batch variant uses a deliberately simple rule: nearby solved cases donate a
warm start and similarity controls a *real unique-evaluation budget*.  Inside a
case, rounded duplicate particles are moved to the nearest unevaluated discrete
position so a particle slot is not wasted looking at a cached mesh again.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import heapq
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
    constraint_violation: float = 0.0
    metadata: Mapping[str, object] = field(default_factory=dict)


def objective_rank(value: ObjectiveValue) -> Tuple[float, ...]:
    """Return a deterministic feasibility-first ordering key.

    A feasible mesh always beats an infeasible mesh.  Feasible meshes are then
    ordered by the engineering objective.  Infeasible meshes are ordered by
    normalized constraint violation before their objective/error values.
    """

    objective = float(value.objective)
    error = float(value.relative_error)
    elements = float(value.element_count)
    if value.feasible:
        return (0.0, objective, error, elements)
    violation = float(value.constraint_violation)
    if math.isnan(violation):
        violation = math.inf
    return (1.0, max(0.0, violation), objective, error, elements)


def is_better(candidate: ObjectiveValue, incumbent: Optional[ObjectiveValue]) -> bool:
    """Whether ``candidate`` should replace ``incumbent``."""

    return incumbent is None or objective_rank(candidate) < objective_rank(incumbent)


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
    max_unique_evaluations: Optional[int] = None
    refill_repeats: bool = True
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
        if self.stagnation_iterations < 1:
            raise ValueError("stagnation_iterations must be positive")
        if self.max_unique_evaluations is not None and self.max_unique_evaluations < 2:
            raise ValueError("max_unique_evaluations must be at least 2 when supplied")
        return self


@dataclass(frozen=True)
class PSOHistoryEntry:
    iteration: int
    best_objective: float
    best_error: float
    best_elements: int
    best_feasible: bool
    unique_evaluations: int
    cache_hits: int
    duplicate_refills: int
    evaluation_budget: Optional[int]


@dataclass(frozen=True)
class PSOResult:
    best: ObjectiveValue
    history: Tuple[PSOHistoryEntry, ...]
    unique_evaluations: int
    cache_hits: int
    duplicate_refills: int
    iterations_completed: int
    wall_time_seconds: float
    used_warm_start: bool
    budget_exhausted: bool
    search_space_exhausted: bool
    config: PSOConfig


class CachedObjective:
    """Exact per-case cache around an expensive evaluator.

    ``initial_values`` can contain results already available from preprocessing
    or a warm-start probe.  ``charged_initial_evaluations`` says how many of
    those values should count against the optimizer's FE budget.  For example,
    the uniform coarse preprocessing result is free to reuse, while a newly run
    warm-start probe counts as one real evaluation.
    """

    def __init__(
        self,
        evaluator: Callable[[Position], ObjectiveValue],
        initial_values: Optional[Iterable[ObjectiveValue]] = None,
        charged_initial_evaluations: int = 0,
    ) -> None:
        self._evaluator = evaluator
        self._cache: Dict[Position, ObjectiveValue] = {}
        for value in initial_values or ():
            key = tuple(int(item) for item in value.position)
            if value.position != key:
                value = replace(value, position=key)
            self._cache[key] = value
        charged = int(charged_initial_evaluations)
        if charged < 0 or charged > len(self._cache):
            raise ValueError(
                "charged_initial_evaluations must lie between zero and the number "
                "of distinct initial values"
            )
        self._evaluation_count = charged
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
        self._evaluation_count += 1
        return value

    @property
    def unique_evaluations(self) -> int:
        """Real evaluations charged to this optimizer run."""

        return self._evaluation_count

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


def _distance_to_anchor(position: Position, anchor: np.ndarray) -> float:
    vector = np.asarray(position, dtype=np.float64)
    return float(np.sum((vector - anchor) ** 2))


def nearest_unseen_position(
    anchor: Sequence[float],
    max_level: int,
    seen: Iterable[Position],
) -> Optional[Position]:
    """Find the nearest unvisited discrete position by best-first grid search.

    The search starts at the rounded PSO position and expands one level step at a
    time.  It is deterministic and normally examines only a handful of cached
    neighbours; if every point has been evaluated it returns ``None``.
    """

    anchor_array = np.asarray(list(anchor), dtype=np.float64)
    if anchor_array.ndim != 1 or anchor_array.size < 1:
        raise ValueError("anchor must be a non-empty one-dimensional sequence")
    visited = set(tuple(int(item) for item in position) for position in seen)
    start = _clip_round(anchor_array, max_level)
    queue: List[Tuple[float, Position]] = [
        (_distance_to_anchor(start, anchor_array), start)
    ]
    queued = {start}
    while queue:
        _, current = heapq.heappop(queue)
        if current not in visited:
            return current
        for dimension in range(len(current)):
            for delta in (-1, 1):
                value = current[dimension] + delta
                if value < 0 or value > max_level:
                    continue
                neighbour_list = list(current)
                neighbour_list[dimension] = value
                neighbour = tuple(neighbour_list)
                if neighbour in queued:
                    continue
                queued.add(neighbour)
                heapq.heappush(
                    queue,
                    (_distance_to_anchor(neighbour, anchor_array), neighbour),
                )
    return None


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

        # The coarse all-L0 design is always retained as the safe baseline.
        positions[0, :] = 0.0

        if warm_start is None:
            return positions

        warm = np.asarray(list(warm_start), dtype=np.float64)
        if warm.shape != (dimensions,):
            raise ValueError(
                f"warm_start has shape {warm.shape}, expected {(dimensions,)}"
            )
        warm = np.clip(warm, 0, cfg.max_level)

        # Keep L0 in slot 0 and put the exact warm start in slot 1.  Earlier code
        # overwrote the L0 slot, removing the safe baseline in transfer runs.
        transfer_slots = max(1, int(math.ceil(cfg.particles * cfg.transfer_fraction)))
        transfer_slots = min(transfer_slots, cfg.particles - 1)
        positions[1, :] = warm
        mutation_probability = max(1.0 / max(dimensions, 1), 0.20)
        for offset in range(1, transfer_slots):
            positions[1 + offset, :] = _mutate_seed(
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
        *,
        initial_values: Optional[Iterable[ObjectiveValue]] = None,
        charged_initial_evaluations: int = 0,
    ) -> PSOResult:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
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
        velocities = rng.uniform(
            -0.25,
            0.25,
            size=(cfg.particles, dimensions),
        )

        pbest_positions = positions.copy()
        pbest_values: List[Optional[ObjectiveValue]] = [None] * cfg.particles
        global_best: Optional[ObjectiveValue] = None
        global_best_position: Optional[np.ndarray] = None
        for value in cached.values.values():
            if len(value.position) != dimensions:
                raise ValueError("initial objective position has the wrong dimension")
            if is_better(value, global_best):
                global_best = value
                global_best_position = np.asarray(value.position, dtype=np.float64)

        history: List[PSOHistoryEntry] = []
        stagnation = 0
        duplicate_refills = 0
        budget_exhausted = False
        search_space_exhausted = False
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

                anchor = positions[particle].copy()
                discrete = _clip_round(anchor, cfg.max_level)
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
                    positions[particle, :] = np.asarray(discrete, dtype=np.float64)
                    duplicate_refills += 1

                value = cached(discrete)
                previous = pbest_values[particle]
                if is_better(value, previous):
                    pbest_values[particle] = value
                    pbest_positions[particle, :] = np.asarray(discrete, dtype=np.float64)
                if is_better(value, global_best):
                    global_best = value
                    global_best_position = np.asarray(discrete, dtype=np.float64)
                    improved_this_iteration = True

            if global_best is None or global_best_position is None:
                raise RuntimeError("PSO produced no objective value")

            history.append(
                PSOHistoryEntry(
                    iteration=iteration + 1,
                    best_objective=float(global_best.objective),
                    best_error=float(global_best.relative_error),
                    best_elements=int(global_best.element_count),
                    best_feasible=bool(global_best.feasible),
                    unique_evaluations=cached.unique_evaluations,
                    cache_hits=cached.cache_hits,
                    duplicate_refills=duplicate_refills,
                    evaluation_budget=cfg.max_unique_evaluations,
                )
            )

            if (
                global_best.feasible
                and cfg.target_objective is not None
                and global_best.objective <= cfg.target_objective
            ):
                break
            if budget_exhausted or search_space_exhausted:
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
        if global_best is None:
            raise RuntimeError("PSO produced no objective value")
        return PSOResult(
            best=global_best,
            history=tuple(history),
            unique_evaluations=cached.unique_evaluations,
            cache_hits=cached.cache_hits,
            duplicate_refills=duplicate_refills,
            iterations_completed=len(history),
            wall_time_seconds=float(elapsed),
            used_warm_start=warm_start is not None,
            budget_exhausted=budget_exhausted,
            search_space_exhausted=search_space_exhausted,
            config=cfg,
        )


@dataclass(frozen=True)
class TransferBudget:
    """Similarity-adaptive real evaluation budget for later batch cases."""

    config: PSOConfig
    similarity: float
    source_distance: Optional[float]
    evaluation_budget: Optional[int]


def similarity_adaptive_config(
    base: PSOConfig,
    normalized_distance: Optional[float],
    *,
    min_unique_evaluations: int = 8,
    distance_scale: float = 0.75,
) -> TransferBudget:
    """Reduce the *real unique FE budget* when a nearby solved case exists.

    ``similarity = exp(-distance / distance_scale)``.  A first or unrelated case
    keeps the original cold configuration.  A nearby case keeps enough swarm
    machinery to move, but receives an explicit cap on unique objective calls.
    """

    base = base.validated()
    if normalized_distance is None:
        return TransferBudget(
            base,
            similarity=0.0,
            source_distance=None,
            evaluation_budget=base.max_unique_evaluations,
        )

    distance = max(0.0, float(normalized_distance))
    similarity = math.exp(-distance / max(distance_scale, 1.0e-12))
    full_budget = int(
        base.max_unique_evaluations
        if base.max_unique_evaluations is not None
        else base.particles * base.iterations
    )
    minimum = max(2, min(int(min_unique_evaluations), full_budget))
    budget = int(round(full_budget - similarity * (full_budget - minimum)))
    budget = max(minimum, min(full_budget, budget))
    particles = max(2, min(base.particles, budget))
    transfer_fraction = min(0.80, max(base.transfer_fraction, 0.40 + 0.40 * similarity))
    adapted = replace(
        base,
        particles=particles,
        transfer_fraction=transfer_fraction,
        max_unique_evaluations=budget,
    )
    return TransferBudget(
        adapted.validated(),
        similarity=similarity,
        source_distance=distance,
        evaluation_budget=budget,
    )


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

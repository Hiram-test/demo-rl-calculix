"""State-aware wrapper around the archived Abaqus adaptive-mesh environment.

The original environment exposed mostly normalized stress/strain features.  The
current mesh-density state, remaining resource budget, invalid actions and the
engineering objective were not part of the graph state passed to the DQN.  This
wrapper adds those quantities without changing the Abaqus scripts themselves.

It deliberately keeps the low-level action space unchanged:

* ``0``: refine the selected cell;
* ``1``: coarsen the selected cell.

Only one global ``(cell, action)`` candidate is applied per environment step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np

from abaqus_env import AbaqusEnv
from state_aware_dqn_agent import COARSEN, REFINE, GraphState, build_graph_state


@dataclass(frozen=True)
class GoalCondition:
    """Structured task contract that can later be produced by an LLM planner.

    The first three priorities are normalized before they are sent to the DQN.
    Keeping this interface numeric and schema-constrained avoids feeding opaque
    language-model embeddings into the numerical policy.
    """

    accuracy_priority: float = 0.60
    resource_priority: float = 0.30
    localization_priority: float = 0.10
    reserve_budget_fraction: float = 0.05
    target_relative_error: float = 0.01

    def normalized(self) -> "GoalCondition":
        priorities = np.asarray(
            [
                max(0.0, float(self.accuracy_priority)),
                max(0.0, float(self.resource_priority)),
                max(0.0, float(self.localization_priority)),
            ],
            dtype=np.float64,
        )
        total = float(priorities.sum())
        if total <= 1.0e-12:
            priorities[:] = (1.0, 0.0, 0.0)
        else:
            priorities /= total
        return GoalCondition(
            accuracy_priority=float(priorities[0]),
            resource_priority=float(priorities[1]),
            localization_priority=float(priorities[2]),
            reserve_budget_fraction=float(
                np.clip(self.reserve_budget_fraction, 0.0, 0.95)
            ),
            target_relative_error=max(0.0, float(self.target_relative_error)),
        )

    def to_dict(self) -> Dict[str, float]:
        return asdict(self.normalized())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GoalCondition":
        supported = {field.name for field in cls.__dataclass_fields__.values()}
        kwargs = {name: value[name] for name in supported if name in value}
        return cls(**kwargs).normalized()

    @classmethod
    def from_json(cls, filepath: str | Path) -> "GoalCondition":
        with open(filepath, "r", encoding="utf-8") as stream:
            data = json.load(stream)
        if not isinstance(data, Mapping):
            raise ValueError("Goal JSON must contain one object")
        return cls.from_mapping(data)


class StateAwareAbaqusEnv(AbaqusEnv):
    """Abaqus environment exposing a Markov state and valid-action mask."""

    # The archived extraction pipeline builds 42 aggregated physical values,
    # seven geometric values and three resource values per cell.
    BASE_CELL_FEATURE_DIM = 52
    DYNAMIC_CELL_FEATURE_NAMES = (
        "mesh_size_over_global",
        "log_mesh_size_over_global",
        "mesh_size_position_in_bounds",
        "refine_is_valid",
        "coarsen_is_valid",
        "last_action_was_refine",
        "last_action_was_coarsen",
        "steps_since_last_action",
        "relative_cell_energy_error",
        "last_action_was_ineffective",
    )
    CELL_FEATURE_DIM = BASE_CELL_FEATURE_DIM + len(DYNAMIC_CELL_FEATURE_NAMES)

    GLOBAL_FEATURE_NAMES = (
        "resource_usage",
        "remaining_budget",
        "load_or_step_fraction",
        "consecutive_failure_fraction",
        "last_reward",
        "relative_allse_error",
        "accuracy_progress",
        "mean_mesh_size_over_global",
        "std_mesh_size_over_global",
        "min_mesh_size_over_global",
        "max_mesh_size_over_global",
        "last_mesh_was_unchanged",
        "last_transition_rolled_back",
        "goal_accuracy_priority",
        "goal_resource_priority",
        "goal_localization_priority",
        "goal_reserve_budget_fraction",
        "goal_target_relative_error",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._last_info_v2: Dict[str, Any] = {}
        self.current_goal = GoalCondition().normalized()
        self._last_action_by_cell: Dict[int, int] = {}
        self._last_action_step_by_cell: Dict[int, int] = {}
        # The pair that just failed or produced no mesh change is blocked for
        # the immediately following decision.  This prevents deterministic
        # argmax from endlessly repeating an ineffective action.
        self._temporarily_blocked_until: Dict[tuple[int, int], int] = {}

    @property
    def global_feature_dim(self) -> int:
        return len(self.GLOBAL_FEATURE_NAMES)

    def set_goal(self, goal: GoalCondition) -> None:
        self.current_goal = goal.normalized()
        # The existing reward already separates accuracy and resource terms.
        self.accuracy_weight = self.current_goal.accuracy_priority
        self.resource_weight = self.current_goal.resource_priority

    def reset(self, run_id: Optional[str] = None):
        self._last_info_v2 = {}
        self._last_action_by_cell.clear()
        self._last_action_step_by_cell.clear()
        self._temporarily_blocked_until.clear()
        return super().reset(run_id=run_id)

    def step(self, action_params: dict):
        previous_energy_map = self._last_info_v2.get("cell_strain_energy", {})
        previous_local_errors: Dict[int, float] = {}
        if isinstance(previous_energy_map, Mapping):
            for raw_cell_id in action_params.keys():
                cell_id = int(raw_cell_id)
                has_current = (
                    cell_id in previous_energy_map
                    or str(cell_id) in previous_energy_map
                )
                if has_current and cell_id in self.baseline_cell_strain_energy:
                    previous_local_errors[cell_id] = (
                        self._relative_cell_energy_error(cell_id)
                    )
        obs, reward, done, info = super().step(action_params)
        info = dict(info or {})
        self._last_info_v2 = info

        # Add a small, interpretable goal-conditioned local component.  It is
        # zero when cell-wise reference energies are unavailable.
        localization_improvement = 0.0
        compared_cells = 0
        for raw_cell_id in action_params.keys():
            cell_id = int(raw_cell_id)
            if cell_id not in previous_local_errors:
                continue
            localization_improvement += (
                previous_local_errors[cell_id]
                - self._relative_cell_energy_error(cell_id)
            )
            compared_cells += 1
        if compared_cells:
            localization_improvement /= compared_cells
        localization_component = (
            self.current_goal.localization_priority * localization_improvement
        )
        reward = float(reward) + float(localization_component)
        components = info.setdefault("reward_components", {})
        if isinstance(components, dict):
            components["localization_improvement"] = localization_improvement
            components["localization_component"] = localization_component
            components["goal_condition"] = self.current_goal.to_dict()
            components["global_reward_v2"] = reward
        cell_rewards = info.setdefault("cell_rewards", {})
        if isinstance(cell_rewards, dict):
            for raw_cell_id in action_params.keys():
                cell_rewards[int(raw_cell_id)] = reward

        for raw_cell_id, raw_action in action_params.items():
            try:
                cell_id = int(raw_cell_id)
                action = int(raw_action)
            except (TypeError, ValueError):
                continue
            self._last_action_by_cell[cell_id] = action
            self._last_action_step_by_cell[cell_id] = int(self.step_index)

            ineffective = bool(
                info.get("state_rollback", False)
                or info.get("mesh_unchanged", False)
                or info.get("cell_mesh_size_violation", False)
            )
            if ineffective:
                self._temporarily_blocked_until[(cell_id, action)] = int(
                    self.step_index
                )

        # The archived environment places these values at the top level of
        # ``info`` rather than under ``global_features``.  Mirror them into the
        # observation so subsequent states can actually see the changed global
        # solution and resource usage.
        if isinstance(self._last_obs, dict):
            resource_usage = self._extract_resource_usage(info)
            self._last_obs["last_reward"] = reward
            self._last_obs["resource_usage"] = resource_usage
            self._last_obs["global_features"] = {
                "resource_usage": resource_usage,
                "allse": info.get("allse"),
                "total_elements": info.get("total_elements"),
                "state_rollback": bool(info.get("state_rollback", False)),
                "mesh_unchanged": bool(info.get("mesh_unchanged", False)),
            }

        return obs, reward, done, info

    def _extract_resource_usage(self, info: Optional[Mapping[str, Any]] = None) -> float:
        info = info or self._last_info_v2
        reward_components = info.get("reward_components", {})
        if isinstance(reward_components, Mapping):
            value = reward_components.get("resource_usage")
            if value is not None:
                return float(np.clip(float(value), 0.0, 2.0))
        value = info.get("resource_usage")
        if value is not None:
            return float(np.clip(float(value), 0.0, 2.0))
        if isinstance(self._last_obs, Mapping):
            value = self._last_obs.get("resource_usage")
            if value is not None:
                return float(np.clip(float(value), 0.0, 2.0))
        total_elements = sum(len(elements) for elements in self.cell_to_elements_map.values())
        return float(total_elements) / max(float(self.max_elements), 1.0)

    def _physical_action_mask(
        self,
        cell_ids: Iterable[int],
        goal: GoalCondition,
    ) -> Dict[int, list[bool]]:
        goal = goal.normalized()
        resource_usage = self._extract_resource_usage()
        reserve_limit = 1.0 - goal.reserve_budget_fraction
        tolerance = 1.0e-9
        masks: Dict[int, list[bool]] = {}

        for raw_cell_id in cell_ids:
            cell_id = int(raw_cell_id)
            current = float(
                self.cell_mesh_density.get(cell_id, self.global_mesh_size)
            )
            refined = current * (1.0 - float(self.refine_step_size))
            coarsened = current * (1.0 + float(self.coarsen_step_size))

            refine_valid = resource_usage < reserve_limit - tolerance
            if self.cell_min_mesh_size is not None:
                refine_valid = refine_valid and (
                    refined >= float(self.cell_min_mesh_size) - tolerance
                )

            coarsen_valid = True
            if self.cell_max_mesh_size is not None:
                coarsen_valid = coarsen_valid and (
                    coarsened <= float(self.cell_max_mesh_size) + tolerance
                )

            masks[cell_id] = [bool(refine_valid), bool(coarsen_valid)]
        return masks

    def get_action_mask(
        self,
        cell_ids: Optional[Iterable[int]] = None,
        goal: Optional[GoalCondition] = None,
    ) -> Dict[int, list[bool]]:
        """Return a per-cell mask for refine/coarsen candidates.

        Bounds and the reserved resource budget are hard constraints.  A pair
        that just rolled back or left the mesh unchanged is additionally
        blocked for one decision, so a near-tied Q surface cannot cause an
        infinite repetition loop.
        """

        goal = (goal or self.current_goal).normalized()
        if cell_ids is None:
            cell_ids = sorted(self.cell_mesh_density.keys())
        cell_ids = tuple(int(cell_id) for cell_id in cell_ids)
        physical = self._physical_action_mask(cell_ids, goal)
        masked = {cell_id: list(row) for cell_id, row in physical.items()}

        for cell_id in cell_ids:
            for action in (REFINE, COARSEN):
                blocked_until = self._temporarily_blocked_until.get((cell_id, action))
                if blocked_until is not None and int(self.step_index) <= blocked_until:
                    masked[cell_id][action] = False

        if any(any(row) for row in masked.values()):
            return masked
        # Never deadlock only because every currently valid candidate happened
        # to be temporarily blocked.  Physical constraints remain enforced.
        return physical

    @staticmethod
    def _pad_or_trim(values: Sequence[float], length: int) -> list[float]:
        result = [float(value) for value in values[:length]]
        if len(result) < length:
            result.extend([0.0] * (length - len(result)))
        return result

    def _relative_cell_energy_error(self, cell_id: int) -> float:
        current_map = self._last_info_v2.get("cell_strain_energy", {})
        if not isinstance(current_map, Mapping):
            return 0.0
        current = current_map.get(cell_id)
        if current is None:
            current = current_map.get(str(cell_id))
        baseline = self.baseline_cell_strain_energy.get(cell_id)
        if current is None or baseline is None:
            return 0.0
        relative = abs(float(current) - float(baseline)) / (
            abs(float(baseline)) + 1.0e-12
        )
        return float(np.tanh(np.log1p(relative)))

    def get_augmented_cell_observations(
        self,
        goal: Optional[GoalCondition] = None,
    ) -> Dict[int, Dict[str, Any]]:
        goal = (goal or self.current_goal).normalized()
        if not isinstance(self._last_obs, Mapping):
            return {}

        base_observations = super().get_cell_observations()
        cell_ids = sorted(
            set(int(cell_id) for cell_id in self.cell_mesh_density.keys())
            | set(int(cell_id) for cell_id in base_observations.keys())
        )
        masks = self.get_action_mask(cell_ids, goal)

        lower = (
            float(self.cell_min_mesh_size)
            if self.cell_min_mesh_size is not None
            else 0.0
        )
        upper = (
            float(self.cell_max_mesh_size)
            if self.cell_max_mesh_size is not None
            else max(float(self.global_mesh_size) * 2.0, lower + 1.0)
        )
        span = max(upper - lower, 1.0e-12)
        normalizer = max(abs(float(self.global_mesh_size)), 1.0e-12)

        observations: Dict[int, Dict[str, Any]] = {}
        for cell_id in cell_ids:
            base = base_observations.get(cell_id, {}).get("self", [])
            base_features = self._pad_or_trim(base, self.BASE_CELL_FEATURE_DIM)

            mesh_size = float(
                self.cell_mesh_density.get(cell_id, self.global_mesh_size)
            )
            ratio = mesh_size / normalizer
            last_action = self._last_action_by_cell.get(cell_id)
            last_step = self._last_action_step_by_cell.get(cell_id)
            if last_step is None:
                recency = 1.0
            else:
                recency = min(
                    max(float(self.step_index - last_step), 0.0) / 20.0,
                    1.0,
                )
            ineffective = any(
                self._temporarily_blocked_until.get((cell_id, action), -1)
                >= int(self.step_index)
                for action in (REFINE, COARSEN)
            )

            dynamic = [
                ratio,
                float(np.tanh(math.log(max(ratio, 1.0e-12)))),
                float(np.clip((mesh_size - lower) / span, 0.0, 1.0)),
                float(masks[cell_id][REFINE]),
                float(masks[cell_id][COARSEN]),
                float(last_action == REFINE),
                float(last_action == COARSEN),
                recency,
                self._relative_cell_energy_error(cell_id),
                float(ineffective),
            ]
            observations[cell_id] = {
                "self": base_features + dynamic,
                "neighbors": base_observations.get(cell_id, {}).get("neighbors", []),
            }

        return observations

    def _current_allse(self) -> Optional[float]:
        value = self._last_info_v2.get("allse")
        if value is None:
            components = self._last_info_v2.get("reward_components", {})
            if isinstance(components, Mapping):
                value = components.get("current_allse")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def get_global_feature_vector(
        self,
        goal: Optional[GoalCondition] = None,
        max_steps: int = 100,
    ) -> list[float]:
        goal = (goal or self.current_goal).normalized()
        resource_usage = self._extract_resource_usage()
        remaining_budget = max(0.0, 1.0 - resource_usage)
        step_fraction = min(float(self.step_index) / max(float(max_steps), 1.0), 1.0)
        failure_fraction = min(
            float(self._consecutive_failures)
            / max(float(self._max_consecutive_failures), 1.0),
            1.0,
        )
        last_reward = 0.0
        if isinstance(self._last_obs, Mapping):
            last_reward = float(self._last_obs.get("last_reward", 0.0) or 0.0)
        last_reward = float(np.tanh(last_reward / 10.0))

        current_allse = self._current_allse()
        relative_error = 0.0
        accuracy_progress = 0.0
        if current_allse is not None and self.baseline_allse is not None:
            relative_error_raw = abs(current_allse - float(self.baseline_allse)) / (
                abs(float(self.baseline_allse)) + 1.0e-12
            )
            relative_error = float(np.tanh(np.log1p(relative_error_raw)))
            if self.initial_allse is not None:
                initial_error = abs(float(self.initial_allse) - float(self.baseline_allse))
                current_error = abs(current_allse - float(self.baseline_allse))
                if initial_error > 1.0e-12:
                    accuracy_progress = float(
                        np.clip(1.0 - current_error / initial_error, -1.0, 1.0)
                    )

        normalizer = max(abs(float(self.global_mesh_size)), 1.0e-12)
        density_ratios = np.asarray(
            [
                float(value) / normalizer
                for value in self.cell_mesh_density.values()
            ],
            dtype=np.float64,
        )
        if density_ratios.size == 0:
            density_ratios = np.asarray([1.0], dtype=np.float64)

        features = [
            float(np.clip(resource_usage, 0.0, 2.0)),
            remaining_budget,
            step_fraction,
            failure_fraction,
            last_reward,
            relative_error,
            accuracy_progress,
            float(density_ratios.mean()),
            float(density_ratios.std()),
            float(density_ratios.min()),
            float(density_ratios.max()),
            float(bool(self._last_info_v2.get("mesh_unchanged", False))),
            float(bool(self._last_info_v2.get("state_rollback", False))),
            goal.accuracy_priority,
            goal.resource_priority,
            goal.localization_priority,
            goal.reserve_budget_fraction,
            float(np.tanh(np.log1p(goal.target_relative_error))),
        ]
        if len(features) != self.global_feature_dim:
            raise RuntimeError("Global feature schema length changed unexpectedly")
        return features

    def build_state(
        self,
        goal: Optional[GoalCondition] = None,
        max_steps: int = 100,
    ) -> GraphState:
        goal = (goal or self.current_goal).normalized()
        observations = self.get_augmented_cell_observations(goal)
        cell_ids = sorted(observations.keys())
        action_mask = self.get_action_mask(cell_ids, goal)
        return build_graph_state(
            cell_observations=observations,
            cell_adjacency=self.cell_adjacency,
            global_features=self.get_global_feature_vector(goal, max_steps=max_steps),
            action_mask=action_mask,
            num_actions=2,
        )

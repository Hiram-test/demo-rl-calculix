"""Structured engineering goal shared by local solver backends.

The language-model boundary is intentionally a small JSON schema.  A planner may
produce this object, while the numerical policy only receives normalized numeric
features.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class GoalCondition:
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
            reserve_budget_fraction=float(np.clip(self.reserve_budget_fraction, 0.0, 0.95)),
            target_relative_error=max(0.0, float(self.target_relative_error)),
        )

    def to_dict(self) -> dict[str, float]:
        return asdict(self.normalized())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GoalCondition":
        supported = set(cls.__dataclass_fields__)
        return cls(**{key: value[key] for key in supported if key in value}).normalized()

    @classmethod
    def from_json(cls, filepath: str | Path) -> "GoalCondition":
        with open(filepath, "r", encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, Mapping):
            raise ValueError("Goal JSON must contain one object")
        return cls.from_mapping(value)

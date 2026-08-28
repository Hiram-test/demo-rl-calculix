"""Discrete coarseness grades.  The model picks a level; PSO picks h."""

from __future__ import annotations

import numpy as np

# 1 = finest, 5 = coarsest.  Priors are only PSO seeds, not model output.
GRADE_PRIOR = {1: 0.16, 2: 0.26, 3: 0.38, 4: 0.54, 5: 0.72}
GRADE_MIN, GRADE_MAX = 1, 5
MIN_STEP = 1.12  # coarser grade must be at least this times finer h


def parse_grade(value, label: str = "grade") -> int:
    g = int(value)
    if g < GRADE_MIN or g > GRADE_MAX:
        raise ValueError(f"{label} must be {GRADE_MIN}..{GRADE_MAX}, got {value!r}")
    return g


def grade_from_frac(frac: float) -> int:
    f = float(frac)
    if f <= 0.20:
        return 1
    if f <= 0.30:
        return 2
    if f <= 0.45:
        return 3
    if f <= 0.60:
        return 4
    return 5


def prior_h(grade: int, h0: float) -> float:
    return float(GRADE_PRIOR[parse_grade(grade)] * h0)


def grades_for_names(names: list[str], grade_map: dict, default: int = 5) -> np.ndarray:
    return np.array([int(grade_map.get(n, default)) for n in names], dtype=int)

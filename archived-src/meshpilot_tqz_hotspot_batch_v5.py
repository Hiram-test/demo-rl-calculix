"""Launch V4 polished hotspot PSO with a stricter business transfer gate.

A transferred mesh must not merely be within 20% of the current coarse baseline.
It must reduce the current case's real coarse objective by at least 25% before the
unique-FE budget is reduced.  Otherwise the full-budget polished PSO is restored.
"""
from __future__ import annotations

import math

import meshpilot_tqz_hotspot_batch as batch
import meshpilot_tqz_hotspot_batch_v4  # noqa: F401  (installs V4 optimizer)


WARM_TO_COARSE_MAX_RATIO = 0.75


def strict_hotspot_transfer_guard(warm, coarse, _legacy_ratio):
    if warm.feasible != coarse.feasible:
        return bool(warm.feasible)
    if warm.feasible:
        coarse_objective = max(abs(float(coarse.objective)), 1.0e-18)
        return float(warm.objective) <= WARM_TO_COARSE_MAX_RATIO * coarse_objective
    warm_violation = float(warm.constraint_violation)
    coarse_violation = float(coarse.constraint_violation)
    if not math.isfinite(warm_violation):
        return False
    if not math.isfinite(coarse_violation):
        return True
    return warm_violation <= WARM_TO_COARSE_MAX_RATIO * max(
        abs(coarse_violation), 1.0e-18
    )


batch.warm_start_is_acceptable = strict_hotspot_transfer_guard


if __name__ == "__main__":
    batch.main()

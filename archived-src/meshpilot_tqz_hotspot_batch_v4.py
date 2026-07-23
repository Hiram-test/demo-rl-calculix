"""Launch the real-hotspot benchmark with coordinate-polished rounded PSO."""
from __future__ import annotations

import meshpilot_tqz_hotspot_batch as batch
from meshpilot_coordinate_polished_round_pso import (
    CoordinatePolishedRoundedPSO,
    CoordinatePolishedRoundedResult,
)


batch.METHOD_HOTSPOT_STEP = "hotspot_polished_pso_cold"
batch.METHOD_HOTSPOT_TRANSFER = "hotspot_polished_pso_transfer"
batch.METHODS = (
    batch.METHOD_FIXED_ROUND,
    batch.METHOD_HOTSPOT_ROUND,
    batch.METHOD_HOTSPOT_STEP,
    batch.METHOD_HOTSPOT_TRANSFER,
)


def _polished_optimizer(request, config):
    return CoordinatePolishedRoundedPSO(
        config,
        swarm_particles=4,
        minimum_swarm_budget=12,
    )


_original_serialize = batch._serialize_result


def _serialize(result):
    payload = _original_serialize(result)
    if isinstance(result, CoordinatePolishedRoundedResult):
        payload.update(
            {
                "algorithm": "coordinate_polished_rounded_pso",
                "coordinate_evaluations": result.coordinate_evaluations,
                "coordinate_improvements": result.coordinate_improvements,
                "swarm_evaluation_budget": result.swarm_evaluation_budget,
            }
        )
    return payload


batch._stepwise_optimizer = _polished_optimizer
batch._serialize_result = _serialize


if __name__ == "__main__":
    batch.main()

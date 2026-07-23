"""Launch the real-hotspot benchmark with coordinate-polished stepwise PSO."""
from __future__ import annotations

import meshpilot_tqz_hotspot_batch as batch
from meshpilot_coordinate_polished_pso import (
    CoordinatePolishedPSOResult,
    CoordinatePolishedStepwisePSO,
)


def _polished_optimizer(request, config):
    return CoordinatePolishedStepwisePSO(
        config,
        local_search_trigger=request.stepwise.local_search_trigger,
        exploration_probability=request.stepwise.exploration_probability,
        agreement_move_probability=request.stepwise.agreement_move_probability,
        swarm_particles=4,
        swarm_iterations=12,
    )


_original_serialize = batch._serialize_result


def _serialize(result):
    payload = _original_serialize(result)
    if isinstance(result, CoordinatePolishedPSOResult):
        payload.update(
            {
                "algorithm": "coordinate_polished_stepwise_pso",
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

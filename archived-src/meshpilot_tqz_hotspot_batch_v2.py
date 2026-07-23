"""Launch the real-hotspot benchmark with the budget-aware guided stepwise swarm."""
from __future__ import annotations

import meshpilot_tqz_hotspot_batch as batch
from meshpilot_guided_stepwise_pso import (
    GuidedStepwiseDiscretePSO,
    budgeted_stepwise_config,
)


def _guided_optimizer(request, config):
    tuned = budgeted_stepwise_config(config, particles=4, iterations=12)
    return GuidedStepwiseDiscretePSO(
        tuned,
        local_search_trigger=request.stepwise.local_search_trigger,
        exploration_probability=request.stepwise.exploration_probability,
        agreement_move_probability=request.stepwise.agreement_move_probability,
    )


batch._stepwise_optimizer = _guided_optimizer


if __name__ == "__main__":
    batch.main()

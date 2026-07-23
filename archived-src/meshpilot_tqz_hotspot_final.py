"""Accepted real-hotspot MeshPilot entrypoint.

This promotes the empirically successful V5 combination:

- candidates from each case's real coarse CalculiX stress field;
- rounded PSO global exploration plus finite coordinate polishing;
- normalized hotspot matching across a batch;
- a real warm-probe gate requiring at least 25% improvement over the current
  coarse objective before the FE budget can be reduced.
"""
from __future__ import annotations

import json
from pathlib import Path

import meshpilot_tqz_hotspot_batch as batch
import meshpilot_tqz_hotspot_batch_v5  # noqa: F401  installs algorithm and guard


_original_summary = batch._summary
_original_write_outputs = batch._write_outputs


def _final_summary(cases):
    summary = _original_summary(cases)
    if "stepwise_vs_rounded_hotspot" in summary:
        summary["polished_pso_vs_rounded_hotspot"] = summary.pop(
            "stepwise_vs_rounded_hotspot"
        )
    if "batch_transfer_vs_stepwise_cold" in summary:
        summary["batch_transfer_vs_polished_cold"] = summary.pop(
            "batch_transfer_vs_stepwise_cold"
        )
    return summary


def _final_write_outputs(request, cases, output_root: Path, elapsed: float):
    payload = _original_write_outputs(request, cases, output_root, elapsed)
    payload["accepted_algorithm"] = "coordinate_polished_rounded_pso"
    payload["candidate_source"] = "real_uniform_coarse_calculix_stress"
    payload["transfer_policy"] = {
        "mapping": "normalized_hotspot_center_size_score",
        "real_warm_probe_required": True,
        "warm_to_coarse_max_objective_ratio": 0.75,
        "minimum_required_coarse_improvement_fraction": 0.25,
    }
    payload["summary"] = _final_summary(cases)
    (output_root / "hotspot_batch_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


batch._summary = _final_summary
batch._write_outputs = _final_write_outputs


if __name__ == "__main__":
    batch.main()

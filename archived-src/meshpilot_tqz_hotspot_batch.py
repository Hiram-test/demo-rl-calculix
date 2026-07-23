"""Four-way real CalculiX benchmark for hotspot-derived TQZ mesh optimization.

The four methods isolate three effects:

A. fixed 3x2 candidate boxes + rounded continuous-position PSO;
B. real coarse-stress hotspots + the same rounded PSO;
C. the same real hotspots + stepwise discrete PSO and neighbour finish;
D. method C plus normalized-hotspot batch transfer and a real warm-probe guard.

Reference, coarse, and every optimizer objective are genuine Gmsh/CalculiX runs.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from meshpilot_pso import (
    DiscretePSO,
    ObjectiveValue,
    PSOConfig,
    PSOResult,
    Position,
    similarity_adaptive_config,
)
from meshpilot_stepwise_pso import StepwiseDiscretePSO, StepwisePSOResult
from meshpilot_tqz_backend import TQZCase, patch_boxes
from meshpilot_tqz_batch import (
    FamilyRequest,
    normalized_descriptors,
    order_cases,
    warm_start_is_acceptable,
)
from meshpilot_tqz_hotspot_backend import (
    Box,
    HotspotAnalysis,
    HotspotCandidate,
    HotspotSpec,
    match_hotspot_levels,
    run_hotspot_analysis,
    score_hotspot_cells,
    select_hotspot_candidates,
)


METHOD_FIXED_ROUND = "fixed_round_cold"
METHOD_HOTSPOT_ROUND = "hotspot_round_cold"
METHOD_HOTSPOT_STEP = "hotspot_stepwise_cold"
METHOD_HOTSPOT_TRANSFER = "hotspot_stepwise_transfer"
METHODS = (
    METHOD_FIXED_ROUND,
    METHOD_HOTSPOT_ROUND,
    METHOD_HOTSPOT_STEP,
    METHOD_HOTSPOT_TRANSFER,
)


@dataclass(frozen=True)
class StepwiseSpec:
    local_search_trigger: int = 2
    exploration_probability: float = 0.08
    agreement_move_probability: float = 0.75

    def validated(self) -> "StepwiseSpec":
        if self.local_search_trigger < 1:
            raise ValueError("local_search_trigger must be positive")
        if not 0.0 <= self.exploration_probability <= 1.0:
            raise ValueError("exploration_probability must lie in [0, 1]")
        if not 0.0 <= self.agreement_move_probability <= 1.0:
            raise ValueError("agreement_move_probability must lie in [0, 1]")
        return self


@dataclass(frozen=True)
class HotspotBenchmarkRequest:
    family: FamilyRequest
    hotspot: HotspotSpec
    stepwise: StepwiseSpec

    @classmethod
    def from_json(cls, filepath: str | Path) -> "HotspotBenchmarkRequest":
        family = FamilyRequest.from_json(filepath)
        raw = json.loads(Path(filepath).read_text(encoding="utf-8"))
        hotspot_raw = raw.get("hotspot", {})
        grid = hotspot_raw.get("grid", [5, 4, 3])
        if not isinstance(grid, Sequence) or len(grid) != 3:
            raise ValueError("hotspot.grid must contain [x, y, z]")
        hotspot = HotspotSpec(
            grid_x=int(grid[0]),
            grid_y=int(grid[1]),
            grid_z=int(grid[2]),
            candidate_count=int(hotspot_raw.get("candidate_count", 6)),
            z_min_ratio=float(hotspot_raw.get("z_min_ratio", 0.20)),
            stress_weight=float(hotspot_raw.get("stress_weight", 0.70)),
            contrast_weight=float(hotspot_raw.get("contrast_weight", 0.30)),
            merge_ratio=float(hotspot_raw.get("merge_ratio", 0.72)),
            max_cells_per_region=int(
                hotspot_raw.get("max_cells_per_region", 3)
            ),
            expansion_ratio=float(hotspot_raw.get("expansion_ratio", 0.08)),
            match_max_cost=float(hotspot_raw.get("match_max_cost", 0.90)),
        ).validated()
        if hotspot.candidate_count != 6:
            raise ValueError("this benchmark currently requires exactly six hotspots")

        stepwise_raw = raw.get("stepwise_pso", {})
        stepwise = StepwiseSpec(
            local_search_trigger=int(
                stepwise_raw.get("local_search_trigger", 2)
            ),
            exploration_probability=float(
                stepwise_raw.get("exploration_probability", 0.08)
            ),
            agreement_move_probability=float(
                stepwise_raw.get("agreement_move_probability", 0.75)
            ),
        ).validated()
        return cls(family=family, hotspot=hotspot, stepwise=stepwise)


@dataclass(frozen=True)
class HotspotTransferRecord:
    case_id: str
    case_descriptor: Tuple[float, ...]
    candidates: Tuple[HotspotCandidate, ...]
    levels: Position
    best_objective: float


class HotspotTransferArchive:
    def __init__(self) -> None:
        self.records: List[HotspotTransferRecord] = []

    def add(self, record: HotspotTransferRecord) -> None:
        self.records.append(record)

    def nearest(
        self,
        descriptor: Sequence[float],
    ) -> Tuple[Optional[HotspotTransferRecord], Optional[float]]:
        if not self.records:
            return None, None
        vector = np.asarray(descriptor, dtype=np.float64)
        record = min(
            self.records,
            key=lambda item: float(
                np.linalg.norm(
                    vector - np.asarray(item.case_descriptor, dtype=np.float64)
                )
            ),
        )
        distance = float(
            np.linalg.norm(
                vector - np.asarray(record.case_descriptor, dtype=np.float64)
            )
        )
        return record, distance


def _objective_value(
    position: Position,
    analysis: HotspotAnalysis,
    reference: HotspotAnalysis,
    request: HotspotBenchmarkRequest,
) -> ObjectiveValue:
    mesh = request.family.mesh
    relative_error = abs(
        float(analysis.result.qoi) - float(reference.result.qoi)
    ) / (abs(float(reference.result.qoi)) + 1.0e-18)
    resource_ratio = analysis.result.element_count / max(
        float(mesh.element_budget), 1.0
    )
    excess = max(0.0, resource_ratio - 1.0)
    objective = (
        relative_error
        + mesh.resource_weight * resource_ratio
        + mesh.budget_penalty * excess * excess
    )
    stresses = list(analysis.element_von_mises.values())
    return ObjectiveValue(
        position=tuple(int(value) for value in position),
        objective=float(objective),
        relative_error=float(relative_error),
        element_count=int(analysis.result.element_count),
        feasible=bool(analysis.result.element_count <= mesh.element_budget),
        constraint_violation=float(excess),
        metadata={
            "qoi": analysis.result.qoi,
            "mean_vertical_displacement": analysis.result.mean_vertical_displacement,
            "max_displacement": analysis.result.max_displacement,
            "compliance": analysis.result.compliance,
            "node_count": analysis.result.node_count,
            "loaded_node_count": analysis.result.loaded_node_count,
            "total_vertical_force": analysis.result.total_vertical_force,
            "total_horizontal_force": analysis.result.total_horizontal_force,
            "applied_moment_y": analysis.result.applied_moment_y,
            "max_von_mises": max(stresses) if stresses else 0.0,
            "mean_von_mises": float(np.mean(stresses)) if stresses else 0.0,
            "workdir": analysis.result.workdir,
            "mesh_signature": list(analysis.result.mesh_signature),
            "resource_ratio": resource_ratio,
        },
    )


def _make_evaluator(
    case: TQZCase,
    request: HotspotBenchmarkRequest,
    reference: HotspotAnalysis,
    candidate_boxes: Sequence[Box],
    case_root: Path,
    method_name: str,
    gmsh_cmd: str,
    ccx_cmd: str,
):
    counter = 0

    def evaluate(position: Position) -> ObjectiveValue:
        nonlocal counter
        counter += 1
        if len(position) != len(candidate_boxes):
            raise ValueError("position dimension differs from candidate count")
        levels = request.family.mesh.mesh_levels_mm
        sizes = tuple(float(levels[int(level)]) for level in position)
        tag = "-".join(str(int(level)) for level in position)
        workdir = case_root / method_name / f"eval_{counter:04d}_{tag}"
        try:
            analysis = run_hotspot_analysis(
                case,
                request.family.material,
                request.family.mesh,
                workdir,
                gmsh_cmd=gmsh_cmd,
                ccx_cmd=ccx_cmd,
                global_size=float(levels[0]),
                candidate_boxes=tuple(candidate_boxes),
                candidate_sizes=sizes,
            )
            return _objective_value(position, analysis, reference, request)
        except Exception as exc:
            return ObjectiveValue(
                position=tuple(int(value) for value in position),
                objective=1.0e9,
                relative_error=1.0e9,
                element_count=0,
                feasible=False,
                constraint_violation=math.inf,
                metadata={"workdir": str(workdir), "error": repr(exc)},
            )

    return evaluate


def _serialize_result(result: PSOResult | StepwisePSOResult) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "best_position": list(result.best.position),
        "best_objective": result.best.objective,
        "best_relative_error": result.best.relative_error,
        "best_element_count": result.best.element_count,
        "best_feasible": result.best.feasible,
        "best_constraint_violation": result.best.constraint_violation,
        "best_metadata": dict(result.best.metadata),
        "unique_fe_evaluations": result.unique_evaluations,
        "cache_hits": result.cache_hits,
        "duplicate_refills": result.duplicate_refills,
        "iterations_completed": result.iterations_completed,
        "wall_time_seconds": result.wall_time_seconds,
        "used_warm_start": result.used_warm_start,
        "budget_exhausted": result.budget_exhausted,
        "search_space_exhausted": result.search_space_exhausted,
        "config": asdict(result.config),
        "history": [asdict(item) for item in result.history],
    }
    if isinstance(result, StepwisePSOResult):
        payload.update(
            {
                "algorithm": "stepwise_discrete_pso",
                "local_neighbor_evaluations": result.local_neighbor_evaluations,
                "step_moves": result.step_moves,
                "local_optimum_confirmed": result.local_optimum_confirmed,
                "local_search_trigger": result.local_search_trigger,
                "exploration_probability": result.exploration_probability,
            }
        )
    else:
        payload.update(
            {
                "algorithm": "rounded_continuous_position_pso",
                "local_neighbor_evaluations": 0,
                "step_moves": 0,
                "local_optimum_confirmed": False,
            }
        )
    return payload


def _rounded_run(
    config: PSOConfig,
    evaluator,
    dimensions: int,
    coarse_value: ObjectiveValue,
) -> PSOResult:
    return DiscretePSO(config).optimize(
        evaluator,
        dimensions=dimensions,
        warm_start=None,
        initial_values=(coarse_value,),
    )


def _stepwise_optimizer(
    request: HotspotBenchmarkRequest,
    config: PSOConfig,
) -> StepwiseDiscretePSO:
    return StepwiseDiscretePSO(
        config,
        local_search_trigger=request.stepwise.local_search_trigger,
        exploration_probability=request.stepwise.exploration_probability,
        agreement_move_probability=request.stepwise.agreement_move_probability,
    )


def _run_case(
    request: HotspotBenchmarkRequest,
    case: TQZCase,
    descriptor: Tuple[float, ...],
    execution_index: int,
    output_root: Path,
    archive: HotspotTransferArchive,
    gmsh_cmd: str,
    ccx_cmd: str,
) -> Dict[str, Any]:
    family = request.family
    case_root = output_root / case.case_id
    case_root.mkdir(parents=True, exist_ok=True)
    levels = family.mesh.mesh_levels_mm
    zero = tuple(0 for _ in range(request.hotspot.candidate_count))

    reference = run_hotspot_analysis(
        case,
        family.material,
        family.mesh,
        case_root / "preprocess" / "reference",
        gmsh_cmd=gmsh_cmd,
        ccx_cmd=ccx_cmd,
        global_size=float(family.mesh.reference_mesh_size_mm),
    )
    coarse = run_hotspot_analysis(
        case,
        family.material,
        family.mesh,
        case_root / "preprocess" / "coarse",
        gmsh_cmd=gmsh_cmd,
        ccx_cmd=ccx_cmd,
        global_size=float(levels[0]),
    )
    coarse_value = _objective_value(zero, coarse, reference, request)

    hotspot_cells = score_hotspot_cells(
        case,
        family.mesh,
        request.hotspot,
        coarse,
    )
    hotspots = select_hotspot_candidates(
        case,
        family.mesh,
        request.hotspot,
        hotspot_cells,
    )
    hotspot_boxes = tuple(candidate.bounds for candidate in hotspots)
    hotspot_priority = tuple(candidate.score for candidate in hotspots)
    fixed_boxes = tuple(patch_boxes(case, family.mesh))
    if len(fixed_boxes) != len(hotspot_boxes):
        raise RuntimeError("fixed and hotspot candidate counts must match")

    seed = family.pso.seed + case.case_id.__hash__() % 100_000
    # Avoid Python's randomized hash in reproducible runs.
    seed = family.pso.seed + sum(ord(value) for value in case.case_id) * 101
    base_config = replace(family.pso, seed=seed)

    fixed_evaluator = _make_evaluator(
        case,
        request,
        reference,
        fixed_boxes,
        case_root,
        METHOD_FIXED_ROUND,
        gmsh_cmd,
        ccx_cmd,
    )
    fixed_round = _rounded_run(
        base_config,
        fixed_evaluator,
        len(fixed_boxes),
        coarse_value,
    )

    hotspot_round_evaluator = _make_evaluator(
        case,
        request,
        reference,
        hotspot_boxes,
        case_root,
        METHOD_HOTSPOT_ROUND,
        gmsh_cmd,
        ccx_cmd,
    )
    hotspot_round = _rounded_run(
        base_config,
        hotspot_round_evaluator,
        len(hotspot_boxes),
        coarse_value,
    )

    hotspot_step_evaluator = _make_evaluator(
        case,
        request,
        reference,
        hotspot_boxes,
        case_root,
        METHOD_HOTSPOT_STEP,
        gmsh_cmd,
        ccx_cmd,
    )
    hotspot_step = _stepwise_optimizer(request, base_config).optimize(
        hotspot_step_evaluator,
        dimensions=len(hotspot_boxes),
        initial_values=(coarse_value,),
        dimension_priority=hotspot_priority,
    )

    source, source_distance = archive.nearest(descriptor)
    warm_start: Optional[Position] = None
    matches = ()
    source_case: Optional[str] = None
    if source is not None:
        warm_start, matches = match_hotspot_levels(
            source.candidates,
            source.levels,
            hotspots,
            max_cost=request.hotspot.match_max_cost,
        )
        source_case = source.case_id

    transfer_evaluator = _make_evaluator(
        case,
        request,
        reference,
        hotspot_boxes,
        case_root,
        METHOD_HOTSPOT_TRANSFER,
        gmsh_cmd,
        ccx_cmd,
    )
    initial_values = [coarse_value]
    charged_initial = 0
    warm_probe: Optional[ObjectiveValue] = None
    guard_accepted = False
    guard_status = "not_applicable"
    if warm_start is not None:
        if warm_start == zero:
            warm_probe = coarse_value
            guard_accepted = True
            guard_status = "accepted_same_as_coarse"
        else:
            warm_probe = transfer_evaluator(warm_start)
            initial_values.append(warm_probe)
            charged_initial = 1
            guard_accepted = warm_start_is_acceptable(
                warm_probe,
                coarse_value,
                family.transfer.guard_ratio,
            )
            guard_status = "accepted" if guard_accepted else "fallback_full_budget"

    transfer_budget = similarity_adaptive_config(
        family.pso,
        source_distance if warm_start is not None and guard_accepted else None,
        min_unique_evaluations=family.transfer.min_unique_evaluations,
        distance_scale=family.transfer.distance_scale,
    )
    transfer_config = replace(transfer_budget.config, seed=seed)
    transfer = _stepwise_optimizer(request, transfer_config).optimize(
        transfer_evaluator,
        dimensions=len(hotspot_boxes),
        warm_start=warm_start if guard_accepted else None,
        initial_values=tuple(initial_values),
        charged_initial_evaluations=charged_initial,
        dimension_priority=hotspot_priority,
    )

    archive.add(
        HotspotTransferRecord(
            case_id=case.case_id,
            case_descriptor=descriptor,
            candidates=hotspots,
            levels=tuple(int(value) for value in transfer.best.position),
            best_objective=float(transfer.best.objective),
        )
    )

    return {
        "execution_index": execution_index,
        "case_id": case.case_id,
        "bearing_model": case.bearing_model,
        "status": "completed",
        "review_required": True,
        "case": asdict(case),
        "descriptor": list(descriptor),
        "reference": reference.result.to_dict(),
        "coarse": coarse.result.to_dict(),
        "coarse_objective": coarse_value.objective,
        "coarse_relative_error": coarse_value.relative_error,
        "hotspot_source": "real_uniform_coarse_calculix_stress",
        "hotspot_cells": [asdict(cell) for cell in hotspot_cells],
        "hotspot_candidates": [candidate.to_dict() for candidate in hotspots],
        "fixed_candidate_boxes": [list(box) for box in fixed_boxes],
        "transfer_source_case": source_case,
        "transfer_source_distance": source_distance,
        "transfer_hotspot_matches": [item.to_dict() for item in matches],
        "transfer_guard_status": guard_status,
        "transfer_guard_accepted": guard_accepted,
        "transfer_similarity": transfer_budget.similarity,
        "transfer_evaluation_budget": transfer_budget.evaluation_budget,
        "transfer_warm_probe": None
        if warm_probe is None
        else {
            "position": list(warm_probe.position),
            "objective": warm_probe.objective,
            "relative_error": warm_probe.relative_error,
            "element_count": warm_probe.element_count,
            "feasible": warm_probe.feasible,
            "metadata": dict(warm_probe.metadata),
        },
        METHOD_FIXED_ROUND: _serialize_result(fixed_round),
        METHOD_HOTSPOT_ROUND: _serialize_result(hotspot_round),
        METHOD_HOTSPOT_STEP: _serialize_result(hotspot_step),
        METHOD_HOTSPOT_TRANSFER: _serialize_result(transfer),
    }


def _aggregate(cases: Sequence[Mapping[str, Any]], method: str) -> Dict[str, Any]:
    rows = [case[method] for case in cases if case.get("status") == "completed"]
    return {
        "total_unique_fe_evaluations": int(
            sum(row["unique_fe_evaluations"] for row in rows)
        ),
        "mean_best_objective": float(
            np.mean([row["best_objective"] for row in rows])
        ),
        "mean_best_relative_error": float(
            np.mean([row["best_relative_error"] for row in rows])
        ),
        "mean_best_element_count": float(
            np.mean([row["best_element_count"] for row in rows])
        ),
        "mean_wall_time_seconds": float(
            np.mean([row["wall_time_seconds"] for row in rows])
        ),
        "feasible_cases": int(sum(bool(row["best_feasible"]) for row in rows)),
        "total_duplicate_refills": int(
            sum(row["duplicate_refills"] for row in rows)
        ),
        "total_local_neighbor_evaluations": int(
            sum(row.get("local_neighbor_evaluations", 0) for row in rows)
        ),
        "local_optimum_confirmed_cases": int(
            sum(bool(row.get("local_optimum_confirmed")) for row in rows)
        ),
    }


def _relative_change(new: float, old: float) -> float:
    return (float(new) - float(old)) / max(abs(float(old)), 1.0e-18)


def _summary(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    completed = [case for case in cases if case.get("status") == "completed"]
    if not completed:
        return {"completed_cases": 0}
    methods = {method: _aggregate(completed, method) for method in METHODS}
    fixed = methods[METHOD_FIXED_ROUND]
    hotspot_round = methods[METHOD_HOTSPOT_ROUND]
    hotspot_step = methods[METHOD_HOTSPOT_STEP]
    transfer = methods[METHOD_HOTSPOT_TRANSFER]
    return {
        "completed_cases": len(completed),
        "execution_order": [case["case_id"] for case in completed],
        "methods": methods,
        "hotspot_selection_vs_fixed": {
            "fe_change_fraction": _relative_change(
                hotspot_round["total_unique_fe_evaluations"],
                fixed["total_unique_fe_evaluations"],
            ),
            "objective_change_fraction": _relative_change(
                hotspot_round["mean_best_objective"],
                fixed["mean_best_objective"],
            ),
            "error_change_fraction": _relative_change(
                hotspot_round["mean_best_relative_error"],
                fixed["mean_best_relative_error"],
            ),
        },
        "stepwise_vs_rounded_hotspot": {
            "fe_change_fraction": _relative_change(
                hotspot_step["total_unique_fe_evaluations"],
                hotspot_round["total_unique_fe_evaluations"],
            ),
            "objective_change_fraction": _relative_change(
                hotspot_step["mean_best_objective"],
                hotspot_round["mean_best_objective"],
            ),
            "error_change_fraction": _relative_change(
                hotspot_step["mean_best_relative_error"],
                hotspot_round["mean_best_relative_error"],
            ),
        },
        "batch_transfer_vs_stepwise_cold": {
            "fe_change_fraction": _relative_change(
                transfer["total_unique_fe_evaluations"],
                hotspot_step["total_unique_fe_evaluations"],
            ),
            "objective_change_fraction": _relative_change(
                transfer["mean_best_objective"],
                hotspot_step["mean_best_objective"],
            ),
            "error_change_fraction": _relative_change(
                transfer["mean_best_relative_error"],
                hotspot_step["mean_best_relative_error"],
            ),
            "guard_accepted_cases": int(
                sum(bool(case["transfer_guard_accepted"]) for case in completed)
            ),
            "guard_fallback_cases": int(
                sum(
                    case["transfer_guard_status"] == "fallback_full_budget"
                    for case in completed
                )
            ),
        },
    }


def _write_outputs(
    request: HotspotBenchmarkRequest,
    cases: Sequence[Mapping[str, Any]],
    output_root: Path,
    elapsed: float,
) -> Dict[str, Any]:
    summary = _summary(cases)
    payload = {
        "request_id": request.family.request_id,
        "source": dict(request.family.source),
        "material": asdict(request.family.material),
        "mesh": asdict(request.family.mesh),
        "pso": asdict(request.family.pso),
        "transfer": asdict(request.family.transfer),
        "hotspot": asdict(request.hotspot),
        "stepwise_pso": asdict(request.stepwise),
        "real_solver_only": True,
        "surrogate_used": False,
        "elapsed_seconds": elapsed,
        "summary": summary,
        "cases": list(cases),
    }
    (output_root / "hotspot_batch_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    fields = [
        "execution_index",
        "case_id",
        "method",
        "best_objective",
        "best_relative_error",
        "best_element_count",
        "best_feasible",
        "unique_fe_evaluations",
        "duplicate_refills",
        "local_neighbor_evaluations",
        "wall_time_seconds",
        "transfer_source_case",
        "transfer_guard_status",
    ]
    with open(
        output_root / "hotspot_batch_metrics.csv",
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            if case.get("status") != "completed":
                continue
            for method in METHODS:
                row = case[method]
                writer.writerow(
                    {
                        "execution_index": case["execution_index"],
                        "case_id": case["case_id"],
                        "method": method,
                        "best_objective": row["best_objective"],
                        "best_relative_error": row["best_relative_error"],
                        "best_element_count": row["best_element_count"],
                        "best_feasible": row["best_feasible"],
                        "unique_fe_evaluations": row["unique_fe_evaluations"],
                        "duplicate_refills": row["duplicate_refills"],
                        "local_neighbor_evaluations": row.get(
                            "local_neighbor_evaluations", 0
                        ),
                        "wall_time_seconds": row["wall_time_seconds"],
                        "transfer_source_case": case.get("transfer_source_case"),
                        "transfer_guard_status": case.get("transfer_guard_status"),
                    }
                )

    report = [
        "# MeshPilot real-hotspot and stepwise-PSO VM report",
        "",
        f"- Request: `{request.family.request_id}`",
        "- Solver path: real Gmsh + real CalculiX only",
        "- Surrogate or downgraded simulation: no",
        f"- Completed cases: {summary.get('completed_cases', 0)}",
        f"- Total wall time: {elapsed:.3f} s",
        "- Engineering interpretation: benchmark only; review remains required.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Per-case methods",
        "",
        "| Order | Case | Method | FE | Error | Objective | Feasible |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for case in cases:
        if case.get("status") != "completed":
            continue
        for method in METHODS:
            row = case[method]
            report.append(
                "| {order} | {case_id} | {method} | {fe} | {error:.6f} | "
                "{objective:.6f} | {feasible} |".format(
                    order=case["execution_index"],
                    case_id=case["case_id"],
                    method=method,
                    fe=row["unique_fe_evaluations"],
                    error=row["best_relative_error"],
                    objective=row["best_objective"],
                    feasible="yes" if row["best_feasible"] else "no",
                )
            )
    (output_root / "hotspot_batch_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    return payload


def run_batch(
    request: HotspotBenchmarkRequest,
    output_root: str | Path,
    gmsh_cmd: str = "gmsh",
    ccx_cmd: str = "ccx",
) -> Dict[str, Any]:
    root = Path(output_root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    descriptors = normalized_descriptors(request.family.cases)
    ordered = order_cases(request.family.cases, request.family.batch_order)
    archive = HotspotTransferArchive()
    results: List[Mapping[str, Any]] = []
    started = time.perf_counter()
    for execution_index, case in enumerate(ordered, start=1):
        print(
            f"[HOTSPOT TQZ] {execution_index}/{len(ordered)} "
            f"case={case.case_id} capacity={case.nominal_vertical_capacity_kN:.0f}kN "
            f"Ag={case.ag:.2f}",
            flush=True,
        )
        try:
            result = _run_case(
                request,
                case,
                descriptors[case.case_id],
                execution_index,
                root,
                archive,
                gmsh_cmd,
                ccx_cmd,
            )
            print(
                "[HOTSPOT RESULT] {case}: fixed={fixed} hotspot_round={rounded} "
                "hotspot_step={step} transfer={transfer}".format(
                    case=case.case_id,
                    fixed=result[METHOD_FIXED_ROUND]["unique_fe_evaluations"],
                    rounded=result[METHOD_HOTSPOT_ROUND]["unique_fe_evaluations"],
                    step=result[METHOD_HOTSPOT_STEP]["unique_fe_evaluations"],
                    transfer=result[METHOD_HOTSPOT_TRANSFER][
                        "unique_fe_evaluations"
                    ],
                ),
                flush=True,
            )
        except Exception as exc:
            result = {
                "execution_index": execution_index,
                "case_id": case.case_id,
                "bearing_model": case.bearing_model,
                "status": "failed",
                "review_required": True,
                "error": repr(exc),
            }
            print(f"[HOTSPOT FAILURE] {case.case_id}: {exc!r}", flush=True)
        results.append(result)
    elapsed = time.perf_counter() - started
    return _write_outputs(request, results, root, elapsed)


def manifest(request: HotspotBenchmarkRequest) -> Dict[str, Any]:
    descriptors = normalized_descriptors(request.family.cases)
    ordered = order_cases(request.family.cases, request.family.batch_order)
    return {
        "request_id": request.family.request_id,
        "execution_order": [case.case_id for case in ordered],
        "methods": list(METHODS),
        "hotspot": asdict(request.hotspot),
        "stepwise_pso": asdict(request.stepwise),
        "pso": asdict(request.family.pso),
        "real_solver_only": True,
        "surrogate_used": False,
        "cases": [
            {
                **asdict(case),
                "descriptor": list(descriptors[case.case_id]),
                "review_required": True,
            }
            for case in ordered
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the real-hotspot TQZ PSO four-way benchmark"
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", default="meshpilot_tqz_hotspot_results")
    parser.add_argument("--gmsh-cmd", default="gmsh")
    parser.add_argument("--ccx-cmd", default="ccx")
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = HotspotBenchmarkRequest.from_json(args.request)
    if args.manifest_only:
        print(json.dumps(manifest(request), indent=2, ensure_ascii=False))
        return
    payload = run_batch(
        request,
        output_root=args.output,
        gmsh_cmd=args.gmsh_cmd,
        ccx_cmd=args.ccx_cmd,
    )
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

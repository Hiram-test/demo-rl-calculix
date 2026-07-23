"""MeshPilot batch agent and Batch-Transfer PSO execution core.

This module implements the two intentionally narrow paper contributions:

1. A guarded batch contract for adjacent engineering users.  An LLM can compile
   natural language into this JSON contract, while this deterministic layer
   expands parameter sweeps, classifies interpolation/extrapolation, and records
   which cases require expert review.
2. A similarity-adaptive discrete PSO.  The first case is optimized from a cold
   swarm.  Later related cases receive a projected warm start and a smaller
   swarm/iteration budget, with performance measured in real CalculiX calls.

The current VM benchmark reuses the repository's small Gmsh/CalculiX plate model
so the optimization mechanism can be tested quickly.  The same interfaces are
intended for a later 3-D bridge-component template.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field, replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from calculix_backend import PlateConfig, StateAwareCalculixEnv
from meshpilot_pso import (
    DiscretePSO,
    ObjectiveValue,
    PSOConfig,
    PSOResult,
    Position,
    project_level_field,
    similarity_adaptive_config,
)


SUPPORTED_CASE_FIELDS = set(PlateConfig.__dataclass_fields__)
DESCRIPTOR_FIELDS = (
    "hole_radius",
    "hole_center_x",
    "hole_center_y",
    "thickness",
    "load_x",
    "load_y",
)


@dataclass(frozen=True)
class RangeSpec:
    minimum: float
    maximum: float

    def contains(self, value: float) -> bool:
        return self.minimum <= float(value) <= self.maximum

    @property
    def span(self) -> float:
        return max(self.maximum - self.minimum, 1.0e-12)


@dataclass(frozen=True)
class ScopeEnvelope:
    calibration: Mapping[str, RangeSpec]
    hard: Mapping[str, RangeSpec]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ScopeEnvelope":
        def parse_ranges(raw: Mapping[str, Any]) -> Dict[str, RangeSpec]:
            result: Dict[str, RangeSpec] = {}
            for name, bounds in raw.items():
                if name not in SUPPORTED_CASE_FIELDS:
                    raise ValueError(f"Unknown scope field: {name}")
                if not isinstance(bounds, Sequence) or len(bounds) != 2:
                    raise ValueError(f"Scope for {name} must be [min, max]")
                result[name] = RangeSpec(float(bounds[0]), float(bounds[1]))
            return result

        return cls(
            calibration=parse_ranges(value.get("calibration", {})),
            hard=parse_ranges(value.get("hard", {})),
        )


@dataclass(frozen=True)
class MeshSearchSpec:
    global_size: float
    reference_size: float
    level_sizes: Tuple[float, ...]
    hotspot_count: int
    element_budget: int
    mandatory_cells: Tuple[int, ...] = ()
    resource_weight: float = 0.03
    budget_penalty: float = 5.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "MeshSearchSpec":
        levels = tuple(float(item) for item in value.get("level_sizes", []))
        if len(levels) < 2:
            raise ValueError("mesh.level_sizes must contain at least L0 and L1")
        if any(level <= 0 for level in levels):
            raise ValueError("All mesh level sizes must be positive")
        if any(levels[index + 1] >= levels[index] for index in range(len(levels) - 1)):
            raise ValueError("level_sizes must decrease from L0 to the finest level")
        return cls(
            global_size=float(value.get("global_size", levels[0])),
            reference_size=float(value["reference_size"]),
            level_sizes=levels,
            hotspot_count=int(value.get("hotspot_count", 6)),
            element_budget=int(value["element_budget"]),
            mandatory_cells=tuple(int(item) for item in value.get("mandatory_cells", [])),
            resource_weight=float(value.get("resource_weight", 0.03)),
            budget_penalty=float(value.get("budget_penalty", 5.0)),
        )


@dataclass(frozen=True)
class BatchRequest:
    request_id: str
    user_role: str
    intent: str
    base_case: Mapping[str, Any]
    sweep: Mapping[str, Tuple[Any, ...]]
    sweep_mode: str
    allow_extrapolation: bool
    scope: ScopeEnvelope
    mesh: MeshSearchSpec
    pso: PSOConfig
    transfer_distance_scale: float = 0.75
    transfer_to_extrapolation: bool = True

    @classmethod
    def from_json(cls, filepath: str | Path) -> "BatchRequest":
        data = json.loads(Path(filepath).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("Request JSON must contain an object")
        base_case = dict(data.get("base_case", {}))
        unknown = set(base_case) - SUPPORTED_CASE_FIELDS
        if unknown:
            raise ValueError(f"Unknown base_case fields: {sorted(unknown)}")
        sweep_raw = data.get("sweep", {})
        if not isinstance(sweep_raw, Mapping):
            raise ValueError("sweep must be an object")
        sweep: Dict[str, Tuple[Any, ...]] = {}
        for name, values in sweep_raw.items():
            if name not in SUPPORTED_CASE_FIELDS:
                raise ValueError(f"Unknown sweep field: {name}")
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
                raise ValueError(f"Sweep values for {name} must be a list")
            if not values:
                raise ValueError(f"Sweep values for {name} cannot be empty")
            sweep[name] = tuple(values)

        pso_raw = data.get("pso", {})
        pso = PSOConfig(
            particles=int(pso_raw.get("particles", 8)),
            iterations=int(pso_raw.get("iterations", 8)),
            max_level=int(pso_raw.get("max_level", 3)),
            inertia_start=float(pso_raw.get("inertia_start", 0.85)),
            inertia_end=float(pso_raw.get("inertia_end", 0.35)),
            cognitive=float(pso_raw.get("cognitive", 1.55)),
            social=float(pso_raw.get("social", 1.55)),
            velocity_limit=float(pso_raw.get("velocity_limit", 2.0)),
            stagnation_iterations=int(pso_raw.get("stagnation_iterations", 4)),
            target_objective=(
                None
                if pso_raw.get("target_objective") is None
                else float(pso_raw["target_objective"])
            ),
            transfer_fraction=float(pso_raw.get("transfer_fraction", 0.50)),
            seed=int(pso_raw.get("seed", 7)),
        ).validated()
        return cls(
            request_id=str(data.get("request_id", "meshpilot_batch")),
            user_role=str(data.get("user_role", "adjacent_engineer")),
            intent=str(data.get("intent", "batch local mesh optimization")),
            base_case=base_case,
            sweep=sweep,
            sweep_mode=str(data.get("sweep_mode", "zip")),
            allow_extrapolation=bool(data.get("allow_extrapolation", False)),
            scope=ScopeEnvelope.from_mapping(data.get("scope", {})),
            mesh=MeshSearchSpec.from_mapping(data.get("mesh", {})),
            pso=pso,
            transfer_distance_scale=float(data.get("transfer_distance_scale", 0.75)),
            transfer_to_extrapolation=bool(data.get("transfer_to_extrapolation", True)),
        )


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    index: int
    plate: PlateConfig
    scope_status: str
    out_of_calibration: Tuple[str, ...]
    out_of_hard_scope: Tuple[str, ...]
    review_required: bool
    descriptor: Tuple[float, ...]


@dataclass(frozen=True)
class TransferRecord:
    case_id: str
    descriptor: Tuple[float, ...]
    full_level_field: Mapping[int, int]
    best_objective: float


class TransferArchive:
    def __init__(self) -> None:
        self.records: List[TransferRecord] = []

    def add(self, record: TransferRecord) -> None:
        self.records.append(record)

    def nearest(self, descriptor: Sequence[float]) -> Tuple[Optional[TransferRecord], Optional[float]]:
        if not self.records:
            return None, None
        vector = np.asarray(descriptor, dtype=np.float64)
        best_record: Optional[TransferRecord] = None
        best_distance = math.inf
        for record in self.records:
            distance = float(np.linalg.norm(vector - np.asarray(record.descriptor, dtype=np.float64)))
            if distance < best_distance:
                best_distance = distance
                best_record = record
        return best_record, best_distance


def _expand_sweep(request: BatchRequest) -> List[Dict[str, Any]]:
    if not request.sweep:
        return [dict(request.base_case)]
    names = list(request.sweep)
    values = [request.sweep[name] for name in names]
    if request.sweep_mode == "zip":
        lengths = {len(items) for items in values}
        if len(lengths) != 1:
            raise ValueError("zip sweep requires all lists to have the same length")
        combinations = zip(*values)
    elif request.sweep_mode == "cartesian":
        combinations = itertools.product(*values)
    else:
        raise ValueError("sweep_mode must be 'zip' or 'cartesian'")
    cases: List[Dict[str, Any]] = []
    for combination in combinations:
        parameters = dict(request.base_case)
        parameters.update(dict(zip(names, combination)))
        cases.append(parameters)
    return cases


def _scope_status(parameters: Mapping[str, Any], scope: ScopeEnvelope) -> Tuple[str, Tuple[str, ...], Tuple[str, ...]]:
    out_calibration: List[str] = []
    out_hard: List[str] = []
    for name, bounds in scope.hard.items():
        if name in parameters and not bounds.contains(float(parameters[name])):
            out_hard.append(name)
    for name, bounds in scope.calibration.items():
        if name in parameters and not bounds.contains(float(parameters[name])):
            out_calibration.append(name)
    if out_hard:
        return "unsupported", tuple(sorted(out_calibration)), tuple(sorted(out_hard))
    if out_calibration:
        return "extrapolation", tuple(sorted(out_calibration)), ()
    return "interpolation", (), ()


def _descriptor(parameters: Mapping[str, Any], scope: ScopeEnvelope) -> Tuple[float, ...]:
    values: List[float] = []
    defaults = asdict(PlateConfig())
    for name in DESCRIPTOR_FIELDS:
        raw = float(parameters.get(name, defaults[name]))
        bounds = scope.calibration.get(name)
        if bounds is None:
            values.append(raw)
        else:
            values.append((raw - bounds.minimum) / bounds.span)
    return tuple(values)


def expand_cases(request: BatchRequest) -> List[CaseSpec]:
    cases: List[CaseSpec] = []
    for index, parameters in enumerate(_expand_sweep(request), start=1):
        plate = PlateConfig(**parameters).validated()
        status, out_calibration, out_hard = _scope_status(parameters, request.scope)
        review_required = status != "interpolation"
        cases.append(
            CaseSpec(
                case_id=f"case_{index:03d}",
                index=index,
                plate=plate,
                scope_status=status,
                out_of_calibration=out_calibration,
                out_of_hard_scope=out_hard,
                review_required=review_required,
                descriptor=_descriptor(parameters, request.scope),
            )
        )
    return cases


def hotspot_candidates(
    result: Any,
    hotspot_count: int,
    mandatory_cells: Iterable[int],
) -> Tuple[int, ...]:
    """Coarse screening only: rank patches by max Mises stress."""

    stress_index = StateAwareCalculixEnv.BASE_CELL_FEATURE_NAMES.index("max_mises")
    scored = sorted(
        (
            (float(features[stress_index]), int(cell_id))
            for cell_id, features in result.cell_features.items()
            if result.cell_to_elements.get(int(cell_id), [])
        ),
        reverse=True,
    )
    selected: List[int] = []
    for cell_id in mandatory_cells:
        if int(cell_id) in result.cell_features and int(cell_id) not in selected:
            selected.append(int(cell_id))
    for _, cell_id in scored:
        if cell_id not in selected:
            selected.append(cell_id)
        if len(selected) >= hotspot_count:
            break
    if not selected:
        raise RuntimeError("Hotspot screening produced no active candidate cells")
    return tuple(selected[: max(hotspot_count, len(tuple(mandatory_cells)))])


def _case_hash(case: CaseSpec) -> str:
    payload = json.dumps(asdict(case.plate), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _position_to_mesh_sizes(
    env: StateAwareCalculixEnv,
    candidates: Sequence[int],
    levels: Sequence[float],
    position: Position,
) -> Dict[int, float]:
    if len(position) != len(candidates):
        raise ValueError("Position dimension does not match hotspot candidates")
    mesh_sizes = {cell_id: float(levels[0]) for cell_id in env.virtual_cells}
    for cell_id, level in zip(candidates, position):
        mesh_sizes[int(cell_id)] = float(levels[int(level)])
    return mesh_sizes


def _full_level_field(candidates: Sequence[int], position: Position) -> Dict[int, int]:
    return {int(cell_id): int(level) for cell_id, level in zip(candidates, position)}


def _result_to_dict(result: PSOResult) -> Dict[str, Any]:
    return {
        "best_position": list(result.best.position),
        "best_objective": result.best.objective,
        "best_relative_error": result.best.relative_error,
        "best_element_count": result.best.element_count,
        "best_feasible": result.best.feasible,
        "best_metadata": dict(result.best.metadata),
        "unique_fe_evaluations": result.unique_evaluations,
        "cache_hits": result.cache_hits,
        "iterations_completed": result.iterations_completed,
        "wall_time_seconds": result.wall_time_seconds,
        "used_warm_start": result.used_warm_start,
        "config": asdict(result.config),
        "history": [asdict(entry) for entry in result.history],
    }


def _make_evaluator(
    env: StateAwareCalculixEnv,
    case_dir: Path,
    candidates: Sequence[int],
    mesh: MeshSearchSpec,
    baseline_qoi: float,
    method_name: str,
):
    counter = {"value": 0}

    def evaluate(position: Position) -> ObjectiveValue:
        counter["value"] += 1
        tag = "-".join(str(value) for value in position)
        workdir = case_dir / method_name / f"eval_{counter['value']:04d}_{tag}"
        mesh_sizes = _position_to_mesh_sizes(env, candidates, mesh.level_sizes, position)
        try:
            fe_result = env._run_analysis(workdir, mesh_sizes)
            relative_error = abs(float(fe_result.qoi) - baseline_qoi) / (
                abs(baseline_qoi) + 1.0e-12
            )
            resource_ratio = fe_result.element_count / max(float(mesh.element_budget), 1.0)
            excess = max(0.0, resource_ratio - 1.0)
            objective = (
                relative_error
                + mesh.resource_weight * resource_ratio
                + mesh.budget_penalty * excess * excess
            )
            feasible = fe_result.element_count <= mesh.element_budget
            metadata = {
                "workdir": str(workdir),
                "qoi": fe_result.qoi,
                "resource_ratio": resource_ratio,
                "mesh_signature": [
                    fe_result.mesh_signature[0],
                    [list(item) for item in fe_result.mesh_signature[1]],
                ],
            }
            return ObjectiveValue(
                position=position,
                objective=float(objective),
                relative_error=float(relative_error),
                element_count=int(fe_result.element_count),
                feasible=bool(feasible),
                metadata=metadata,
            )
        except Exception as exc:  # solver failures are valid black-box outcomes
            return ObjectiveValue(
                position=position,
                objective=1.0e6,
                relative_error=1.0e6,
                element_count=0,
                feasible=False,
                metadata={"workdir": str(workdir), "error": repr(exc)},
            )

    return evaluate


def _run_one_case(
    request: BatchRequest,
    case: CaseSpec,
    output_root: Path,
    gmsh_cmd: str,
    ccx_cmd: str,
    archive: TransferArchive,
) -> Dict[str, Any]:
    case_dir = output_root / case.case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    if case.scope_status == "unsupported":
        return {
            "case_id": case.case_id,
            "scope_status": case.scope_status,
            "review_required": True,
            "status": "rejected",
            "out_of_hard_scope": list(case.out_of_hard_scope),
        }
    if case.scope_status == "extrapolation" and not request.allow_extrapolation:
        return {
            "case_id": case.case_id,
            "scope_status": case.scope_status,
            "review_required": True,
            "status": "held_for_expert_review",
            "out_of_calibration": list(case.out_of_calibration),
        }

    env = StateAwareCalculixEnv(
        plate=case.plate,
        simulations_root=str(case_dir / "solver"),
        gmsh_cmd=gmsh_cmd,
        ccx_cmd=ccx_cmd,
        global_mesh_size=request.mesh.global_size,
        cell_min_mesh_size=min(request.mesh.level_sizes),
        cell_max_mesh_size=max(request.mesh.level_sizes),
        max_elements=max(request.mesh.element_budget * 3, request.mesh.element_budget + 100),
        min_elements=1,
        solver_timeout_seconds=300,
    )
    uniform_reference = {
        cell_id: request.mesh.reference_size for cell_id in env.virtual_cells
    }
    uniform_coarse = {
        cell_id: request.mesh.global_size for cell_id in env.virtual_cells
    }
    case_hash = _case_hash(case)
    baseline = env._run_analysis(case_dir / "preprocess" / f"reference_{case_hash}", uniform_reference)
    coarse = env._run_analysis(case_dir / "preprocess" / f"coarse_{case_hash}", uniform_coarse)
    candidates = hotspot_candidates(
        coarse,
        request.mesh.hotspot_count,
        request.mesh.mandatory_cells,
    )

    cold_evaluator = _make_evaluator(
        env,
        case_dir,
        candidates,
        request.mesh,
        baseline.qoi,
        "cold_pso",
    )
    cold_config = replace(request.pso, seed=request.pso.seed + case.index * 101)
    cold_result = DiscretePSO(cold_config).optimize(
        cold_evaluator,
        dimensions=len(candidates),
        warm_start=None,
    )

    source, distance = archive.nearest(case.descriptor)
    allow_transfer = source is not None and (
        case.scope_status != "extrapolation" or request.transfer_to_extrapolation
    )
    warm_start: Optional[Position] = None
    source_case_id: Optional[str] = None
    if allow_transfer and source is not None:
        warm_start = project_level_field(source.full_level_field, candidates)
        source_case_id = source.case_id
    transfer_budget = similarity_adaptive_config(
        request.pso,
        distance if warm_start is not None else None,
        distance_scale=request.transfer_distance_scale,
    )
    transfer_config = replace(
        transfer_budget.config,
        seed=request.pso.seed + case.index * 101,
    )
    transfer_evaluator = _make_evaluator(
        env,
        case_dir,
        candidates,
        request.mesh,
        baseline.qoi,
        "transfer_pso",
    )
    transfer_result = DiscretePSO(transfer_config).optimize(
        transfer_evaluator,
        dimensions=len(candidates),
        warm_start=warm_start,
    )
    archive.add(
        TransferRecord(
            case_id=case.case_id,
            descriptor=case.descriptor,
            full_level_field=_full_level_field(candidates, transfer_result.best.position),
            best_objective=transfer_result.best.objective,
        )
    )

    return {
        "case_id": case.case_id,
        "status": "completed",
        "plate": asdict(case.plate),
        "scope_status": case.scope_status,
        "out_of_calibration": list(case.out_of_calibration),
        "review_required": case.review_required,
        "descriptor": list(case.descriptor),
        "reference_qoi": baseline.qoi,
        "reference_elements": baseline.element_count,
        "coarse_qoi": coarse.qoi,
        "coarse_elements": coarse.element_count,
        "coarse_relative_error": abs(coarse.qoi - baseline.qoi) / (abs(baseline.qoi) + 1.0e-12),
        "hotspot_candidates": list(candidates),
        "transfer_source_case": source_case_id,
        "transfer_source_distance": distance,
        "transfer_similarity": transfer_budget.similarity,
        "cold": _result_to_dict(cold_result),
        "transfer": _result_to_dict(transfer_result),
    }


def _summary(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    completed = [case for case in cases if case.get("status") == "completed"]
    if not completed:
        return {"completed_cases": 0}

    def aggregate(method: str) -> Dict[str, Any]:
        rows = [case[method] for case in completed]
        return {
            "mean_best_objective": float(np.mean([row["best_objective"] for row in rows])),
            "mean_best_relative_error": float(
                np.mean([row["best_relative_error"] for row in rows])
            ),
            "total_unique_fe_evaluations": int(
                sum(row["unique_fe_evaluations"] for row in rows)
            ),
            "total_cache_hits": int(sum(row["cache_hits"] for row in rows)),
            "mean_wall_time_seconds": float(
                np.mean([row["wall_time_seconds"] for row in rows])
            ),
            "feasible_cases": int(sum(bool(row["best_feasible"]) for row in rows)),
        }

    cold = aggregate("cold")
    transfer = aggregate("transfer")
    cold_calls = max(cold["total_unique_fe_evaluations"], 1)
    return {
        "completed_cases": len(completed),
        "interpolation_cases": sum(case["scope_status"] == "interpolation" for case in completed),
        "extrapolation_cases": sum(case["scope_status"] == "extrapolation" for case in completed),
        "cold": cold,
        "transfer": transfer,
        "fe_call_reduction_fraction": 1.0
        - transfer["total_unique_fe_evaluations"] / cold_calls,
        "objective_change_fraction": (
            transfer["mean_best_objective"] - cold["mean_best_objective"]
        )
        / max(abs(cold["mean_best_objective"]), 1.0e-12),
    }


def _write_outputs(
    request: BatchRequest,
    cases: Sequence[Mapping[str, Any]],
    output_root: Path,
    elapsed: float,
) -> Dict[str, Any]:
    summary = _summary(cases)
    payload = {
        "request": {
            "request_id": request.request_id,
            "user_role": request.user_role,
            "intent": request.intent,
            "allow_extrapolation": request.allow_extrapolation,
            "mesh": asdict(request.mesh),
            "pso": asdict(request.pso),
        },
        "elapsed_seconds": elapsed,
        "summary": summary,
        "cases": list(cases),
    }
    (output_root / "batch_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with open(output_root / "batch_metrics.csv", "w", newline="", encoding="utf-8") as stream:
        fieldnames = [
            "case_id",
            "scope_status",
            "review_required",
            "method",
            "best_objective",
            "best_relative_error",
            "best_element_count",
            "unique_fe_evaluations",
            "iterations_completed",
            "wall_time_seconds",
            "transfer_source_case",
            "transfer_source_distance",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for case in cases:
            if case.get("status") != "completed":
                continue
            for method in ("cold", "transfer"):
                row = case[method]
                writer.writerow(
                    {
                        "case_id": case["case_id"],
                        "scope_status": case["scope_status"],
                        "review_required": case["review_required"],
                        "method": method,
                        "best_objective": row["best_objective"],
                        "best_relative_error": row["best_relative_error"],
                        "best_element_count": row["best_element_count"],
                        "unique_fe_evaluations": row["unique_fe_evaluations"],
                        "iterations_completed": row["iterations_completed"],
                        "wall_time_seconds": row["wall_time_seconds"],
                        "transfer_source_case": case.get("transfer_source_case"),
                        "transfer_source_distance": case.get("transfer_source_distance"),
                    }
                )

    report_lines = [
        "# MeshPilot batch-transfer PSO VM report",
        "",
        f"- Request: `{request.request_id}`",
        f"- User role: `{request.user_role}`",
        f"- Intent: {request.intent}",
        f"- Total wall time: {elapsed:.3f} s",
        "",
        "## Batch summary",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Case-level results",
        "",
        "| Case | Scope | Review | Cold FE calls | Transfer FE calls | Cold error | Transfer error | Warm source |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for case in cases:
        if case.get("status") != "completed":
            report_lines.append(
                f"| {case['case_id']} | {case['scope_status']} | yes | - | - | - | - | {case['status']} |"
            )
            continue
        report_lines.append(
            "| {case_id} | {scope} | {review} | {cold_calls} | {transfer_calls} | "
            "{cold_error:.5f} | {transfer_error:.5f} | {source} |".format(
                case_id=case["case_id"],
                scope=case["scope_status"],
                review="yes" if case["review_required"] else "no",
                cold_calls=case["cold"]["unique_fe_evaluations"],
                transfer_calls=case["transfer"]["unique_fe_evaluations"],
                cold_error=case["cold"]["best_relative_error"],
                transfer_error=case["transfer"]["best_relative_error"],
                source=case.get("transfer_source_case") or "cold start",
            )
        )
    report_lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "This benchmark tests batch transfer and expensive-evaluation reduction on the existing small CalculiX template.  It does not by itself prove usability for non-experts or generalization to a 3-D bridge component; those require a later user study and a validated 3-D template.",
        ]
    )
    (output_root / "batch_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )
    return payload


def run_batch(
    request: BatchRequest,
    output_root: str | Path,
    gmsh_cmd: str = "gmsh",
    ccx_cmd: str = "ccx",
) -> Dict[str, Any]:
    root = Path(output_root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    cases = expand_cases(request)
    archive = TransferArchive()
    results: List[Mapping[str, Any]] = []
    started = time.perf_counter()
    for case in cases:
        print(
            f"[MESHPILOT] {case.case_id} scope={case.scope_status} "
            f"review_required={case.review_required}"
        )
        result = _run_one_case(
            request,
            case,
            root,
            gmsh_cmd=gmsh_cmd,
            ccx_cmd=ccx_cmd,
            archive=archive,
        )
        results.append(result)
        if result.get("status") == "completed":
            print(
                "[RESULT] {case}: cold_calls={cold}, transfer_calls={transfer}, "
                "cold_error={cold_error:.6f}, transfer_error={transfer_error:.6f}".format(
                    case=case.case_id,
                    cold=result["cold"]["unique_fe_evaluations"],
                    transfer=result["transfer"]["unique_fe_evaluations"],
                    cold_error=result["cold"]["best_relative_error"],
                    transfer_error=result["transfer"]["best_relative_error"],
                )
            )
    elapsed = time.perf_counter() - started
    return _write_outputs(request, results, root, elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MeshPilot guarded batch expansion and Batch-Transfer PSO"
    )
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", default="meshpilot_batch_results")
    parser.add_argument("--gmsh-cmd", default="gmsh")
    parser.add_argument("--ccx-cmd", default="ccx")
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Expand and scope-check the user batch without running FEA",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = BatchRequest.from_json(args.request)
    if args.manifest_only:
        manifest = [asdict(case) for case in expand_cases(request)]
        print(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
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

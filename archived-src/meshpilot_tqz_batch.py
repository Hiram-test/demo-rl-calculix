"""Guarded Batch-Transfer PSO driver for the drawing-derived TQZ support family."""
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
    project_level_field,
    similarity_adaptive_config,
)
from meshpilot_tqz_backend import (
    SupportRunResult,
    TQZCase,
    TQZMaterial,
    TQZMeshSpec,
    command_available,
    run_support_analysis,
)


DESCRIPTOR_FIELDS = (
    "nominal_vertical_capacity_kN",
    "ag",
    "A",
    "B",
    "C",
    "D",
    "H",
)
PATCH_IDS = tuple(range(1, 7))


@dataclass(frozen=True)
class TransferSpec:
    min_unique_evaluations: int = 10
    distance_scale: float = 0.75
    guard_ratio: float = 1.20

    def validated(self) -> "TransferSpec":
        if self.min_unique_evaluations < 2:
            raise ValueError("min_unique_evaluations must be at least 2")
        if self.distance_scale <= 0.0:
            raise ValueError("distance_scale must be positive")
        if self.guard_ratio < 1.0:
            raise ValueError("guard_ratio must be at least 1")
        return self


@dataclass(frozen=True)
class FamilyRequest:
    request_id: str
    source: Mapping[str, Any]
    material: TQZMaterial
    batch_order: str
    cases: Tuple[TQZCase, ...]
    mesh: TQZMeshSpec
    pso: PSOConfig
    transfer: TransferSpec
    review_gate: Mapping[str, Any]

    @classmethod
    def from_json(cls, filepath: str | Path) -> "FamilyRequest":
        data = json.loads(Path(filepath).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("TQZ request must contain one JSON object")
        material_raw = data.get("material", {})
        material = TQZMaterial(
            name=str(material_raw.get("name", "C50 concrete")),
            young_modulus_mpa=float(material_raw.get("young_modulus_mpa", 35_500.0)),
            poisson_ratio=float(material_raw.get("poisson_ratio", 0.20)),
        ).validated()
        cases = tuple(
            TQZCase(
                case_id=str(item["case_id"]),
                bearing_model=str(item["bearing_model"]),
                nominal_vertical_capacity_kN=float(item["nominal_vertical_capacity_kN"]),
                ag=float(item["ag"]),
                A=float(item["A"]),
                B=float(item["B"]),
                C=float(item["C"]),
                D=float(item["D"]),
                H=float(item["H"]),
                scope_status=str(item.get("scope_status", "interpolation")),
            ).validated()
            for item in data.get("cases", [])
        )
        if not cases:
            raise ValueError("TQZ request contains no cases")
        model_raw = data.get("first_vm_model", {})
        mesh = TQZMeshSpec(
            mesh_levels_mm=tuple(float(value) for value in model_raw.get("mesh_levels_mm", (180, 130, 95, 70))),
            reference_mesh_size_mm=float(model_raw.get("reference_mesh_size_mm", 55.0)),
            block_margin_x_mm=float(model_raw.get("block_margin_x_mm", 260.0)),
            block_margin_y_mm=float(model_raw.get("block_margin_y_mm", 260.0)),
            block_depth_mm=float(model_raw.get("block_depth_mm", 600.0)),
            patch_depth_mm=float(model_raw.get("patch_depth_mm", 280.0)),
            element_budget=int(model_raw.get("element_budget", 18_000)),
            resource_weight=float(model_raw.get("resource_weight", 0.025)),
            budget_penalty=float(model_raw.get("budget_penalty", 5.0)),
            solver_timeout_seconds=int(model_raw.get("solver_timeout_seconds", 300)),
        ).validated()
        pso_raw = data.get("pso", {})
        pso = PSOConfig(
            particles=int(pso_raw.get("particles", 8)),
            iterations=int(pso_raw.get("iterations", 8)),
            max_level=int(pso_raw.get("max_level", len(mesh.mesh_levels_mm) - 1)),
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
            max_unique_evaluations=int(
                pso_raw.get("max_unique_evaluations", model_raw.get("max_unique_evaluations", 32))
            ),
            refill_repeats=bool(pso_raw.get("refill_repeats", True)),
            seed=int(pso_raw.get("seed", 23)),
        ).validated()
        if pso.max_level != len(mesh.mesh_levels_mm) - 1:
            raise ValueError("pso.max_level must match the number of mesh levels")
        transfer_raw = data.get("transfer", {})
        transfer = TransferSpec(
            min_unique_evaluations=int(transfer_raw.get("min_unique_evaluations", 10)),
            distance_scale=float(transfer_raw.get("distance_scale", 0.75)),
            guard_ratio=float(transfer_raw.get("guard_ratio", 1.20)),
        ).validated()
        return cls(
            request_id=str(data.get("request_id", "tqz_support_family")),
            source=dict(data.get("source", {})),
            material=material,
            batch_order=str(data.get("batch_order", "nearest_path")),
            cases=cases,
            mesh=mesh,
            pso=pso,
            transfer=transfer,
            review_gate=dict(data.get("review_gate", {})),
        )


def normalized_descriptors(cases: Sequence[TQZCase]) -> Dict[str, Tuple[float, ...]]:
    matrix = np.asarray(
        [[float(getattr(case, field)) for field in DESCRIPTOR_FIELDS] for case in cases],
        dtype=np.float64,
    )
    minimum = matrix.min(axis=0)
    maximum = matrix.max(axis=0)
    span = np.where(maximum - minimum > 1.0e-12, maximum - minimum, 1.0)
    normalized = (matrix - minimum) / span
    return {
        case.case_id: tuple(float(value) for value in normalized[index].tolist())
        for index, case in enumerate(cases)
    }


def order_cases(cases: Sequence[TQZCase], mode: str) -> List[TQZCase]:
    values = list(cases)
    if mode == "input" or len(values) < 2:
        return values
    if mode != "nearest_path":
        raise ValueError(f"unknown batch order: {mode}")
    descriptors = normalized_descriptors(values)
    matrix = np.asarray([descriptors[case.case_id] for case in values], dtype=np.float64)
    centre = np.median(matrix, axis=0)
    first = min(
        values,
        key=lambda case: (
            float(np.linalg.norm(np.asarray(descriptors[case.case_id]) - centre)),
            case.case_id,
        ),
    )
    ordered = [first]
    pending = [case for case in values if case is not first]
    while pending:
        next_case = min(
            pending,
            key=lambda case: (
                min(
                    float(
                        np.linalg.norm(
                            np.asarray(descriptors[case.case_id])
                            - np.asarray(descriptors[done.case_id])
                        )
                    )
                    for done in ordered
                ),
                case.case_id,
            ),
        )
        ordered.append(next_case)
        pending.remove(next_case)
    return ordered


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
        record = min(
            self.records,
            key=lambda item: float(
                np.linalg.norm(vector - np.asarray(item.descriptor, dtype=np.float64))
            ),
        )
        distance = float(np.linalg.norm(vector - np.asarray(record.descriptor, dtype=np.float64)))
        return record, distance


def warm_start_is_acceptable(
    warm: ObjectiveValue,
    coarse: ObjectiveValue,
    guard_ratio: float,
) -> bool:
    ratio = max(1.0, float(guard_ratio))
    if warm.feasible != coarse.feasible:
        return bool(warm.feasible)
    field = "objective" if warm.feasible else "constraint_violation"
    warm_value = float(getattr(warm, field))
    coarse_value = float(getattr(coarse, field))
    allowance = (ratio - 1.0) * max(abs(coarse_value), 1.0e-12)
    return warm_value <= coarse_value + allowance


def _objective_value(
    position: Position,
    result: SupportRunResult,
    reference: SupportRunResult,
    mesh: TQZMeshSpec,
) -> ObjectiveValue:
    relative_error = abs(float(result.qoi) - float(reference.qoi)) / (
        abs(float(reference.qoi)) + 1.0e-18
    )
    resource_ratio = result.element_count / max(float(mesh.element_budget), 1.0)
    excess = max(0.0, resource_ratio - 1.0)
    objective = (
        relative_error
        + mesh.resource_weight * resource_ratio
        + mesh.budget_penalty * excess * excess
    )
    return ObjectiveValue(
        position=tuple(int(value) for value in position),
        objective=float(objective),
        relative_error=float(relative_error),
        element_count=int(result.element_count),
        feasible=bool(result.element_count <= mesh.element_budget),
        constraint_violation=float(excess),
        metadata={
            "qoi": result.qoi,
            "mean_vertical_displacement": result.mean_vertical_displacement,
            "max_displacement": result.max_displacement,
            "compliance": result.compliance,
            "node_count": result.node_count,
            "loaded_node_count": result.loaded_node_count,
            "total_vertical_force": result.total_vertical_force,
            "total_horizontal_force": result.total_horizontal_force,
            "applied_moment_y": result.applied_moment_y,
            "workdir": result.workdir,
            "mesh_signature": list(result.mesh_signature),
            "resource_ratio": resource_ratio,
        },
    )


def _make_evaluator(
    case: TQZCase,
    request: FamilyRequest,
    reference: SupportRunResult,
    case_root: Path,
    method_name: str,
    gmsh_cmd: str,
    ccx_cmd: str,
):
    counter = 0

    def evaluate(position: Position) -> ObjectiveValue:
        nonlocal counter
        counter += 1
        levels = request.mesh.mesh_levels_mm
        patch_sizes = tuple(float(levels[int(level)]) for level in position)
        tag = "-".join(str(int(level)) for level in position)
        workdir = case_root / method_name / f"eval_{counter:04d}_{tag}"
        try:
            result = run_support_analysis(
                case,
                request.material,
                request.mesh,
                workdir,
                gmsh_cmd=gmsh_cmd,
                ccx_cmd=ccx_cmd,
                global_size=float(levels[0]),
                patch_sizes=patch_sizes,
            )
            return _objective_value(position, result, reference, request.mesh)
        except Exception as exc:
            return ObjectiveValue(
                position=tuple(position),
                objective=1.0e9,
                relative_error=1.0e9,
                element_count=0,
                feasible=False,
                constraint_violation=math.inf,
                metadata={"workdir": str(workdir), "error": repr(exc)},
            )

    return evaluate


def _result_to_dict(result: PSOResult) -> Dict[str, Any]:
    return {
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


def _run_case(
    request: FamilyRequest,
    case: TQZCase,
    descriptor: Tuple[float, ...],
    execution_index: int,
    root: Path,
    archive: TransferArchive,
    gmsh_cmd: str,
    ccx_cmd: str,
) -> Dict[str, Any]:
    case_root = root / case.case_id
    case_root.mkdir(parents=True, exist_ok=True)
    levels = request.mesh.mesh_levels_mm
    reference = run_support_analysis(
        case,
        request.material,
        request.mesh,
        case_root / "preprocess" / "reference",
        gmsh_cmd=gmsh_cmd,
        ccx_cmd=ccx_cmd,
        global_size=request.mesh.reference_mesh_size_mm,
        patch_sizes=(request.mesh.reference_mesh_size_mm,) * 6,
    )
    coarse = run_support_analysis(
        case,
        request.material,
        request.mesh,
        case_root / "preprocess" / "coarse",
        gmsh_cmd=gmsh_cmd,
        ccx_cmd=ccx_cmd,
        global_size=levels[0],
        patch_sizes=(levels[0],) * 6,
    )
    zero = (0,) * 6
    coarse_value = _objective_value(zero, coarse, reference, request.mesh)
    seed = request.pso.seed + execution_index * 101

    cold_evaluator = _make_evaluator(
        case,
        request,
        reference,
        case_root,
        "cold_pso",
        gmsh_cmd,
        ccx_cmd,
    )
    cold = DiscretePSO(replace(request.pso, seed=seed)).optimize(
        cold_evaluator,
        dimensions=6,
        initial_values=(coarse_value,),
    )

    source, distance = archive.nearest(descriptor)
    warm_start: Optional[Position] = None
    source_case: Optional[str] = None
    if source is not None:
        warm_start = project_level_field(source.full_level_field, PATCH_IDS)
        source_case = source.case_id

    transfer_evaluator = _make_evaluator(
        case,
        request,
        reference,
        case_root,
        "transfer_pso",
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
                request.transfer.guard_ratio,
            )
            guard_status = "accepted" if guard_accepted else "fallback_full_budget"

    transfer_budget = similarity_adaptive_config(
        request.pso,
        distance if warm_start is not None and guard_accepted else None,
        min_unique_evaluations=request.transfer.min_unique_evaluations,
        distance_scale=request.transfer.distance_scale,
    )
    transfer_config = replace(transfer_budget.config, seed=seed)
    search_warm_start = warm_start if guard_accepted else None
    transfer = DiscretePSO(transfer_config).optimize(
        transfer_evaluator,
        dimensions=6,
        warm_start=search_warm_start,
        initial_values=tuple(initial_values),
        charged_initial_evaluations=charged_initial,
    )
    archive.add(
        TransferRecord(
            case_id=case.case_id,
            descriptor=descriptor,
            full_level_field={patch_id: int(level) for patch_id, level in zip(PATCH_IDS, transfer.best.position)},
            best_objective=float(transfer.best.objective),
        )
    )
    return {
        "execution_index": execution_index,
        "case_id": case.case_id,
        "bearing_model": case.bearing_model,
        "scope_status": case.scope_status,
        "review_required": True,
        "descriptor": list(descriptor),
        "case": asdict(case),
        "reference": reference.to_dict(),
        "coarse": coarse.to_dict(),
        "coarse_objective": coarse_value.objective,
        "coarse_relative_error": coarse_value.relative_error,
        "transfer_source_case": source_case,
        "transfer_source_distance": distance,
        "transfer_similarity": transfer_budget.similarity,
        "transfer_evaluation_budget": transfer_budget.evaluation_budget,
        "transfer_guard_status": guard_status,
        "transfer_guard_accepted": guard_accepted,
        "transfer_warm_probe": None if warm_probe is None else {
            "position": list(warm_probe.position),
            "objective": warm_probe.objective,
            "relative_error": warm_probe.relative_error,
            "element_count": warm_probe.element_count,
            "feasible": warm_probe.feasible,
            "metadata": dict(warm_probe.metadata),
        },
        "cold": _result_to_dict(cold),
        "transfer": _result_to_dict(transfer),
    }


def _summary(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    completed = [case for case in cases if case.get("status") == "completed"]
    if not completed:
        return {"completed_cases": 0}

    def aggregate(method: str) -> Dict[str, Any]:
        rows = [case[method] for case in completed]
        return {
            "total_unique_fe_evaluations": int(sum(row["unique_fe_evaluations"] for row in rows)),
            "mean_best_objective": float(np.mean([row["best_objective"] for row in rows])),
            "mean_best_relative_error": float(
                np.mean([row["best_relative_error"] for row in rows])
            ),
            "mean_best_element_count": float(
                np.mean([row["best_element_count"] for row in rows])
            ),
            "mean_wall_time_seconds": float(np.mean([row["wall_time_seconds"] for row in rows])),
            "feasible_cases": int(sum(bool(row["best_feasible"]) for row in rows)),
            "total_duplicate_refills": int(sum(row["duplicate_refills"] for row in rows)),
            "budget_exhausted_cases": int(sum(bool(row["budget_exhausted"]) for row in rows)),
        }

    cold = aggregate("cold")
    transfer = aggregate("transfer")
    cold_calls = max(cold["total_unique_fe_evaluations"], 1)
    return {
        "completed_cases": len(completed),
        "cold": cold,
        "transfer": transfer,
        "fe_call_reduction_fraction": 1.0
        - transfer["total_unique_fe_evaluations"] / cold_calls,
        "objective_change_fraction": (
            transfer["mean_best_objective"] - cold["mean_best_objective"]
        )
        / max(abs(cold["mean_best_objective"]), 1.0e-18),
        "guard_accepted_cases": int(
            sum(bool(case.get("transfer_guard_accepted")) for case in completed)
        ),
        "guard_fallback_cases": int(
            sum(case.get("transfer_guard_status") == "fallback_full_budget" for case in completed)
        ),
        "execution_order": [case["case_id"] for case in completed],
    }


def _write_outputs(
    request: FamilyRequest,
    cases: Sequence[Mapping[str, Any]],
    root: Path,
    elapsed: float,
) -> Dict[str, Any]:
    summary = _summary(cases)
    payload = {
        "request_id": request.request_id,
        "source": dict(request.source),
        "material": asdict(request.material),
        "mesh": asdict(request.mesh),
        "pso": asdict(request.pso),
        "transfer": asdict(request.transfer),
        "review_gate": dict(request.review_gate),
        "elapsed_seconds": elapsed,
        "summary": summary,
        "cases": list(cases),
    }
    (root / "tqz_batch_results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    fields = [
        "execution_index",
        "case_id",
        "bearing_model",
        "method",
        "transfer_source_case",
        "transfer_source_distance",
        "transfer_guard_status",
        "best_objective",
        "best_relative_error",
        "best_element_count",
        "best_feasible",
        "unique_fe_evaluations",
        "duplicate_refills",
        "wall_time_seconds",
    ]
    with open(root / "tqz_batch_metrics.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            if case.get("status") != "completed":
                continue
            for method in ("cold", "transfer"):
                row = case[method]
                writer.writerow(
                    {
                        "execution_index": case["execution_index"],
                        "case_id": case["case_id"],
                        "bearing_model": case["bearing_model"],
                        "method": method,
                        "transfer_source_case": case.get("transfer_source_case"),
                        "transfer_source_distance": case.get("transfer_source_distance"),
                        "transfer_guard_status": case.get("transfer_guard_status"),
                        "best_objective": row["best_objective"],
                        "best_relative_error": row["best_relative_error"],
                        "best_element_count": row["best_element_count"],
                        "best_feasible": row["best_feasible"],
                        "unique_fe_evaluations": row["unique_fe_evaluations"],
                        "duplicate_refills": row["duplicate_refills"],
                        "wall_time_seconds": row["wall_time_seconds"],
                    }
                )
    lines = [
        "# MeshPilot TQZ(XII) support-family VM report",
        "",
        f"- Request: `{request.request_id}`",
        f"- Drawing: `{request.source.get('drawing_number', '')}`",
        f"- Completed cases: {summary.get('completed_cases', 0)}",
        f"- Total wall time: {elapsed:.3f} s",
        "- Interpretation: algorithm benchmark only; engineer review remains required.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Case-level comparison",
        "",
        "| Order | Case | Source | Guard | Cold FE | Transfer FE | Cold error | Transfer error |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for case in cases:
        if case.get("status") != "completed":
            continue
        lines.append(
            "| {order} | {case_id} | {source} | {guard} | {cold_calls} | {transfer_calls} | "
            "{cold_error:.6f} | {transfer_error:.6f} |".format(
                order=case["execution_index"],
                case_id=case["case_id"],
                source=case.get("transfer_source_case") or "cold start",
                guard=case.get("transfer_guard_status"),
                cold_calls=case["cold"]["unique_fe_evaluations"],
                transfer_calls=case["transfer"]["unique_fe_evaluations"],
                cold_error=case["cold"]["best_relative_error"],
                transfer_error=case["transfer"]["best_relative_error"],
            )
        )
    (root / "tqz_batch_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def run_batch(
    request: FamilyRequest,
    output_root: str | Path,
    gmsh_cmd: str = "gmsh",
    ccx_cmd: str = "ccx",
) -> Dict[str, Any]:
    if not command_available(gmsh_cmd):
        raise FileNotFoundError(f"Gmsh command is unavailable: {gmsh_cmd}")
    if not command_available(ccx_cmd):
        raise FileNotFoundError(f"CalculiX command is unavailable: {ccx_cmd}")
    root = Path(output_root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    descriptors = normalized_descriptors(request.cases)
    ordered = order_cases(request.cases, request.batch_order)
    archive = TransferArchive()
    results: List[Mapping[str, Any]] = []
    started = time.perf_counter()
    for execution_index, case in enumerate(ordered, start=1):
        print(
            f"[TQZ] {execution_index}/{len(ordered)} case={case.case_id} "
            f"capacity={case.nominal_vertical_capacity_kN:.0f}kN Ag={case.ag:.2f}"
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
            result["status"] = "completed"
            print(
                "[TQZ RESULT] {case}: cold={cold} transfer={transfer} "
                "cold_error={cold_error:.6f} transfer_error={transfer_error:.6f}".format(
                    case=case.case_id,
                    cold=result["cold"]["unique_fe_evaluations"],
                    transfer=result["transfer"]["unique_fe_evaluations"],
                    cold_error=result["cold"]["best_relative_error"],
                    transfer_error=result["transfer"]["best_relative_error"],
                )
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
            print(f"[TQZ FAILURE] {case.case_id}: {exc!r}")
        results.append(result)
    elapsed = time.perf_counter() - started
    return _write_outputs(request, results, root, elapsed)


def manifest(request: FamilyRequest) -> Dict[str, Any]:
    descriptors = normalized_descriptors(request.cases)
    ordered = order_cases(request.cases, request.batch_order)
    return {
        "request_id": request.request_id,
        "batch_order": request.batch_order,
        "source": dict(request.source),
        "material": asdict(request.material),
        "mesh": asdict(request.mesh),
        "pso": asdict(request.pso),
        "transfer": asdict(request.transfer),
        "execution_order": [case.case_id for case in ordered],
        "cases": [
            {
                **asdict(case),
                "descriptor": list(descriptors[case.case_id]),
                "horizontal_ratio": case.horizontal_ratio,
                "review_required": True,
            }
            for case in ordered
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the TQZ support-family 3-D batch PSO benchmark")
    parser.add_argument("--request", required=True)
    parser.add_argument("--output", default="meshpilot_tqz_vm_results")
    parser.add_argument("--gmsh-cmd", default="gmsh")
    parser.add_argument("--ccx-cmd", default="ccx")
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request = FamilyRequest.from_json(args.request)
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

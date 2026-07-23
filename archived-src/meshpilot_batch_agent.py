"""Guarded batch expansion and Batch-Transfer PSO for the CalculiX demo."""
from __future__ import annotations

import argparse, csv, hashlib, itertools, json, math, shutil, time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from calculix_backend import PlateConfig, StateAwareCalculixEnv
from meshpilot_pso import (
    DiscretePSO, ObjectiveValue, PSOConfig, PSOResult, Position,
    project_level_field, similarity_adaptive_config,
)

SUPPORTED_CASE_FIELDS = set(PlateConfig.__dataclass_fields__)
DESCRIPTOR_FIELDS = ("hole_radius", "hole_center_x", "hole_center_y", "thickness", "load_x", "load_y")
BATCH_ORDER_MODES = {"input", "nearest_path"}


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
        def parse(raw: Mapping[str, Any]) -> Dict[str, RangeSpec]:
            result = {}
            for name, bounds in raw.items():
                if name not in SUPPORTED_CASE_FIELDS:
                    raise ValueError(f"Unknown scope field: {name}")
                if not isinstance(bounds, Sequence) or len(bounds) != 2:
                    raise ValueError(f"Scope for {name} must be [min, max]")
                lo, hi = map(float, bounds)
                if lo > hi:
                    raise ValueError(f"Scope for {name} has min greater than max")
                result[name] = RangeSpec(lo, hi)
            return result
        return cls(parse(value.get("calibration", {})), parse(value.get("hard", {})))


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
        levels = tuple(map(float, value.get("level_sizes", [])))
        if len(levels) < 2 or any(x <= 0 for x in levels):
            raise ValueError("mesh.level_sizes must contain at least two positive values")
        if any(levels[i + 1] >= levels[i] for i in range(len(levels) - 1)):
            raise ValueError("level_sizes must decrease from L0 to the finest level")
        count, budget = int(value.get("hotspot_count", 6)), int(value["element_budget"])
        if count < 1 or budget < 1:
            raise ValueError("hotspot_count and element_budget must be positive")
        return cls(
            float(value.get("global_size", levels[0])), float(value["reference_size"]),
            levels, count, budget,
            tuple(map(int, value.get("mandatory_cells", []))),
            float(value.get("resource_weight", 0.03)), float(value.get("budget_penalty", 5.0)),
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
    batch_order: str = "nearest_path"
    transfer_distance_scale: float = 0.75
    transfer_to_extrapolation: bool = True
    transfer_min_evaluations: int = 8
    transfer_guard_ratio: float = 1.05

    @classmethod
    def from_json(cls, filepath: str | Path) -> "BatchRequest":
        data = json.loads(Path(filepath).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise ValueError("Request JSON must contain an object")
        base = dict(data.get("base_case", {}))
        unknown = set(base) - SUPPORTED_CASE_FIELDS
        if unknown:
            raise ValueError(f"Unknown base_case fields: {sorted(unknown)}")
        raw_sweep = data.get("sweep", {})
        if not isinstance(raw_sweep, Mapping):
            raise ValueError("sweep must be an object")
        sweep = {}
        for name, values in raw_sweep.items():
            if name not in SUPPORTED_CASE_FIELDS:
                raise ValueError(f"Unknown sweep field: {name}")
            if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
                raise ValueError(f"Sweep values for {name} must be a non-empty list")
            sweep[name] = tuple(values)
        raw = data.get("pso", {})
        cap = raw.get("max_unique_evaluations")
        pso = PSOConfig(
            particles=int(raw.get("particles", 8)), iterations=int(raw.get("iterations", 8)),
            max_level=int(raw.get("max_level", 3)), inertia_start=float(raw.get("inertia_start", .85)),
            inertia_end=float(raw.get("inertia_end", .35)), cognitive=float(raw.get("cognitive", 1.55)),
            social=float(raw.get("social", 1.55)), velocity_limit=float(raw.get("velocity_limit", 2.0)),
            stagnation_iterations=int(raw.get("stagnation_iterations", 4)),
            target_objective=None if raw.get("target_objective") is None else float(raw["target_objective"]),
            transfer_fraction=float(raw.get("transfer_fraction", .5)),
            max_unique_evaluations=None if cap is None else int(cap),
            refill_repeats=bool(raw.get("refill_repeats", True)), seed=int(raw.get("seed", 7)),
        ).validated()
        order = str(data.get("batch_order", "nearest_path"))
        if order not in BATCH_ORDER_MODES:
            raise ValueError(f"batch_order must be one of {sorted(BATCH_ORDER_MODES)}")
        minimum, guard = int(data.get("transfer_min_evaluations", 8)), float(data.get("transfer_guard_ratio", 1.05))
        if minimum < 2 or guard < 1.0:
            raise ValueError("transfer_min_evaluations >= 2 and transfer_guard_ratio >= 1 are required")
        return cls(
            str(data.get("request_id", "meshpilot_batch")), str(data.get("user_role", "adjacent_engineer")),
            str(data.get("intent", "batch local mesh optimization")), base, sweep,
            str(data.get("sweep_mode", "zip")), bool(data.get("allow_extrapolation", False)),
            ScopeEnvelope.from_mapping(data.get("scope", {})), MeshSearchSpec.from_mapping(data.get("mesh", {})), pso,
            order, float(data.get("transfer_distance_scale", .75)),
            bool(data.get("transfer_to_extrapolation", True)), minimum, guard,
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
        vector = np.asarray(descriptor, dtype=float)
        pairs = [(float(np.linalg.norm(vector - np.asarray(r.descriptor))), r) for r in self.records]
        distance, record = min(pairs, key=lambda item: item[0])
        return record, distance


def _expand_sweep(request: BatchRequest) -> List[Dict[str, Any]]:
    if not request.sweep:
        return [dict(request.base_case)]
    names, values = list(request.sweep), list(request.sweep.values())
    if request.sweep_mode == "zip":
        if len({len(x) for x in values}) != 1:
            raise ValueError("zip sweep requires equal list lengths")
        combinations = zip(*values)
    elif request.sweep_mode == "cartesian":
        combinations = itertools.product(*values)
    else:
        raise ValueError("sweep_mode must be 'zip' or 'cartesian'")
    result = []
    for combination in combinations:
        row = dict(request.base_case); row.update(dict(zip(names, combination))); result.append(row)
    return result


def _scope_status(parameters: Mapping[str, Any], scope: ScopeEnvelope):
    hard = sorted(name for name, bounds in scope.hard.items() if name in parameters and not bounds.contains(parameters[name]))
    calibration = sorted(name for name, bounds in scope.calibration.items() if name in parameters and not bounds.contains(parameters[name]))
    if hard: return "unsupported", tuple(calibration), tuple(hard)
    if calibration: return "extrapolation", tuple(calibration), ()
    return "interpolation", (), ()


def _descriptor(parameters: Mapping[str, Any], scope: ScopeEnvelope) -> Tuple[float, ...]:
    defaults, values = asdict(PlateConfig()), []
    for name in DESCRIPTOR_FIELDS:
        raw, bounds = float(parameters.get(name, defaults[name])), scope.calibration.get(name)
        values.append(raw if bounds is None else (raw - bounds.minimum) / bounds.span)
    return tuple(values)


def expand_cases(request: BatchRequest) -> List[CaseSpec]:
    cases = []
    for index, params in enumerate(_expand_sweep(request), 1):
        plate = PlateConfig(**params).validated(); status, out_cal, out_hard = _scope_status(params, request.scope)
        cases.append(CaseSpec(f"case_{index:03d}", index, plate, status, out_cal, out_hard, status != "interpolation", _descriptor(params, request.scope)))
    return cases


def order_cases(cases: Sequence[CaseSpec], mode: str) -> List[CaseSpec]:
    cases = list(cases)
    if mode == "input" or len(cases) < 2: return cases
    if mode != "nearest_path": raise ValueError(f"Unknown batch order mode: {mode}")
    active = [c for c in cases if c.scope_status != "unsupported"]
    rejected = sorted((c for c in cases if c.scope_status == "unsupported"), key=lambda c: c.index)
    if len(active) < 2: return active + rejected
    centre = np.median(np.asarray([c.descriptor for c in active], dtype=float), axis=0)
    first = min(active, key=lambda c: (float(np.linalg.norm(np.asarray(c.descriptor) - centre)), c.index))
    ordered, pending = [first], [c for c in active if c is not first]
    while pending:
        nxt = min(pending, key=lambda c: (min(float(np.linalg.norm(np.asarray(c.descriptor) - np.asarray(done.descriptor))) for done in ordered), c.index))
        ordered.append(nxt); pending.remove(nxt)
    return ordered + rejected


def hotspot_candidates(result: Any, hotspot_count: int, mandatory_cells: Iterable[int]) -> Tuple[int, ...]:
    mandatory = tuple(map(int, mandatory_cells))
    index = StateAwareCalculixEnv.BASE_CELL_FEATURE_NAMES.index("max_mises")
    scored = sorted(((float(f[index]), int(cid)) for cid, f in result.cell_features.items() if result.cell_to_elements.get(int(cid), [])), reverse=True)
    selected = []
    for cid in mandatory:
        if cid in result.cell_features and cid not in selected: selected.append(cid)
    for _, cid in scored:
        if cid not in selected: selected.append(cid)
        if len(selected) >= hotspot_count: break
    if not selected: raise RuntimeError("Hotspot screening produced no active candidate cells")
    return tuple(selected[:max(hotspot_count, len(mandatory))])


def _case_hash(case: CaseSpec) -> str:
    text = json.dumps(asdict(case.plate), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _position_to_mesh_sizes(env, candidates, levels, position):
    if len(position) != len(candidates): raise ValueError("Position dimension does not match hotspot candidates")
    sizes = {cid: float(levels[0]) for cid in env.virtual_cells}
    for cid, level in zip(candidates, position): sizes[int(cid)] = float(levels[int(level)])
    return sizes


def _full_level_field(candidates, position):
    return {int(cid): int(level) for cid, level in zip(candidates, position)}


def _result_to_dict(result: PSOResult) -> Dict[str, Any]:
    return {
        "best_position": list(result.best.position), "best_objective": result.best.objective,
        "best_relative_error": result.best.relative_error, "best_element_count": result.best.element_count,
        "best_feasible": result.best.feasible, "best_constraint_violation": result.best.constraint_violation,
        "best_metadata": dict(result.best.metadata), "unique_fe_evaluations": result.unique_evaluations,
        "cache_hits": result.cache_hits, "duplicate_refills": result.duplicate_refills,
        "iterations_completed": result.iterations_completed, "wall_time_seconds": result.wall_time_seconds,
        "used_warm_start": result.used_warm_start, "budget_exhausted": result.budget_exhausted,
        "search_space_exhausted": result.search_space_exhausted, "config": asdict(result.config),
        "history": [asdict(x) for x in result.history],
    }


def _objective_from_fe_result(position, result, mesh, baseline_qoi, workdir=None):
    error = abs(float(result.qoi) - baseline_qoi) / (abs(baseline_qoi) + 1e-12)
    ratio = result.element_count / max(float(mesh.element_budget), 1.0); excess = max(0.0, ratio - 1.0)
    objective = error + mesh.resource_weight * ratio + mesh.budget_penalty * excess * excess
    metadata = {"workdir": workdir or str(getattr(result, "workdir", "")), "qoi": result.qoi, "resource_ratio": ratio,
                "mesh_signature": [result.mesh_signature[0], [list(x) for x in result.mesh_signature[1]]]}
    return ObjectiveValue(tuple(position), float(objective), float(error), int(result.element_count), result.element_count <= mesh.element_budget, float(excess), metadata)


def _make_evaluator(env, case_dir, candidates, mesh, baseline_qoi, method_name):
    counter = 0
    def evaluate(position):
        nonlocal counter; counter += 1
        workdir = case_dir / method_name / f"eval_{counter:04d}_{'-'.join(map(str, position))}"
        try:
            result = env._run_analysis(workdir, _position_to_mesh_sizes(env, candidates, mesh.level_sizes, position))
            return _objective_from_fe_result(position, result, mesh, baseline_qoi, str(workdir))
        except Exception as exc:
            return ObjectiveValue(tuple(position), 1e6, 1e6, 0, False, math.inf, {"workdir": str(workdir), "error": repr(exc)})
    return evaluate


def warm_start_is_acceptable(warm: ObjectiveValue, coarse: ObjectiveValue, ratio: float) -> bool:
    ratio = max(1.0, float(ratio))
    if warm.feasible != coarse.feasible: return bool(warm.feasible)
    field = "objective" if warm.feasible else "constraint_violation"
    warm_value, coarse_value = float(getattr(warm, field)), float(getattr(coarse, field))
    return warm_value <= coarse_value + (ratio - 1.0) * max(abs(coarse_value), 1e-12)


def _run_one_case(request, case, output_root, gmsh_cmd, ccx_cmd, archive):
    case_dir = output_root / case.case_id; case_dir.mkdir(parents=True, exist_ok=True)
    if case.scope_status == "unsupported":
        return {"case_id": case.case_id, "scope_status": case.scope_status, "review_required": True, "status": "rejected", "out_of_hard_scope": list(case.out_of_hard_scope)}
    if case.scope_status == "extrapolation" and not request.allow_extrapolation:
        return {"case_id": case.case_id, "scope_status": case.scope_status, "review_required": True, "status": "held_for_expert_review", "out_of_calibration": list(case.out_of_calibration)}
    env = StateAwareCalculixEnv(plate=case.plate, simulations_root=str(case_dir / "solver"), gmsh_cmd=gmsh_cmd, ccx_cmd=ccx_cmd,
        global_mesh_size=request.mesh.global_size, cell_min_mesh_size=min(request.mesh.level_sizes), cell_max_mesh_size=max(request.mesh.level_sizes),
        max_elements=max(request.mesh.element_budget * 3, request.mesh.element_budget + 100), min_elements=1, solver_timeout_seconds=300)
    case_hash = _case_hash(case)
    baseline = env._run_analysis(case_dir / "preprocess" / f"reference_{case_hash}", {cid: request.mesh.reference_size for cid in env.virtual_cells})
    coarse = env._run_analysis(case_dir / "preprocess" / f"coarse_{case_hash}", {cid: request.mesh.global_size for cid in env.virtual_cells})
    candidates = hotspot_candidates(coarse, request.mesh.hotspot_count, request.mesh.mandatory_cells)
    zero = tuple(0 for _ in candidates)
    coarse_value = _objective_from_fe_result(zero, coarse, request.mesh, baseline.qoi)
    seed = request.pso.seed + case.index * 101
    cold_eval = _make_evaluator(env, case_dir, candidates, request.mesh, baseline.qoi, "cold_pso")
    cold = DiscretePSO(replace(request.pso, seed=seed)).optimize(cold_eval, len(candidates), initial_values=(coarse_value,))

    source, distance = archive.nearest(case.descriptor)
    allowed = source is not None and (case.scope_status != "extrapolation" or request.transfer_to_extrapolation)
    warm = project_level_field(source.full_level_field, candidates) if allowed and source else None
    source_id = source.case_id if warm is not None else None
    transfer_eval = _make_evaluator(env, case_dir, candidates, request.mesh, baseline.qoi, "transfer_pso")
    initial, charged, probe, accepted, guard = [coarse_value], 0, None, False, "not_applicable"
    if warm is not None:
        if warm == zero:
            probe, accepted, guard = coarse_value, True, "accepted_same_as_coarse"
        else:
            probe = transfer_eval(warm); initial.append(probe); charged = 1
            accepted = warm_start_is_acceptable(probe, coarse_value, request.transfer_guard_ratio)
            guard = "accepted" if accepted else "fallback_full_budget"
    budget = similarity_adaptive_config(request.pso, distance if warm is not None and accepted else None,
        min_unique_evaluations=request.transfer_min_evaluations, distance_scale=request.transfer_distance_scale)
    transfer = DiscretePSO(replace(budget.config, seed=seed)).optimize(transfer_eval, len(candidates), warm,
        initial_values=tuple(initial), charged_initial_evaluations=charged)
    archive.add(TransferRecord(case.case_id, case.descriptor, _full_level_field(candidates, transfer.best.position), transfer.best.objective))
    return {
        "case_id": case.case_id, "status": "completed", "plate": asdict(case.plate), "scope_status": case.scope_status,
        "out_of_calibration": list(case.out_of_calibration), "review_required": case.review_required, "descriptor": list(case.descriptor),
        "reference_qoi": baseline.qoi, "reference_elements": baseline.element_count, "coarse_qoi": coarse.qoi,
        "coarse_elements": coarse.element_count, "coarse_relative_error": abs(coarse.qoi - baseline.qoi) / (abs(baseline.qoi) + 1e-12),
        "coarse_objective": coarse_value.objective, "hotspot_candidates": list(candidates), "transfer_source_case": source_id,
        "transfer_source_distance": distance, "transfer_similarity": budget.similarity, "transfer_guard_status": guard,
        "transfer_guard_accepted": accepted, "transfer_guard_ratio": request.transfer_guard_ratio,
        "transfer_warm_probe_objective": None if probe is None else probe.objective,
        "transfer_evaluation_budget": budget.evaluation_budget, "cold": _result_to_dict(cold), "transfer": _result_to_dict(transfer),
    }


def _summary(cases):
    completed = [c for c in cases if c.get("status") == "completed"]
    if not completed: return {"completed_cases": 0}
    def aggregate(method):
        rows = [c[method] for c in completed]
        return {"mean_best_objective": float(np.mean([r["best_objective"] for r in rows])),
            "mean_best_relative_error": float(np.mean([r["best_relative_error"] for r in rows])),
            "total_unique_fe_evaluations": sum(r["unique_fe_evaluations"] for r in rows),
            "total_cache_hits": sum(r["cache_hits"] for r in rows), "total_duplicate_refills": sum(r["duplicate_refills"] for r in rows),
            "mean_wall_time_seconds": float(np.mean([r["wall_time_seconds"] for r in rows])),
            "feasible_cases": sum(bool(r["best_feasible"]) for r in rows), "budget_exhausted_cases": sum(bool(r["budget_exhausted"]) for r in rows)}
    cold, transfer = aggregate("cold"), aggregate("transfer"); cold_calls = max(cold["total_unique_fe_evaluations"], 1)
    return {"completed_cases": len(completed), "interpolation_cases": sum(c["scope_status"] == "interpolation" for c in completed),
        "extrapolation_cases": sum(c["scope_status"] == "extrapolation" for c in completed),
        "guard_accepted_cases": sum(bool(c.get("transfer_guard_accepted")) for c in completed),
        "guard_fallback_cases": sum(c.get("transfer_guard_status") == "fallback_full_budget" for c in completed),
        "cold": cold, "transfer": transfer, "fe_call_reduction_fraction": 1 - transfer["total_unique_fe_evaluations"] / cold_calls,
        "objective_change_fraction": (transfer["mean_best_objective"] - cold["mean_best_objective"]) / max(abs(cold["mean_best_objective"]), 1e-12)}


def _write_outputs(request, cases, root, elapsed):
    summary = _summary(cases)
    payload = {"request": {"request_id": request.request_id, "user_role": request.user_role, "intent": request.intent,
        "allow_extrapolation": request.allow_extrapolation, "batch_order": request.batch_order,
        "transfer_min_evaluations": request.transfer_min_evaluations, "transfer_guard_ratio": request.transfer_guard_ratio,
        "mesh": asdict(request.mesh), "pso": asdict(request.pso)}, "elapsed_seconds": elapsed, "summary": summary, "cases": list(cases)}
    (root / "batch_results.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = ["execution_index", "case_id", "scope_status", "review_required", "method", "best_objective", "best_relative_error",
        "best_element_count", "unique_fe_evaluations", "duplicate_refills", "evaluation_budget", "iterations_completed", "wall_time_seconds",
        "transfer_source_case", "transfer_source_distance", "transfer_guard_status"]
    with open(root / "batch_metrics.csv", "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for case in cases:
            if case.get("status") != "completed": continue
            for method in ("cold", "transfer"):
                row = case[method]
                writer.writerow({"execution_index": case.get("execution_index"), "case_id": case["case_id"], "scope_status": case["scope_status"],
                    "review_required": case["review_required"], "method": method, "best_objective": row["best_objective"],
                    "best_relative_error": row["best_relative_error"], "best_element_count": row["best_element_count"],
                    "unique_fe_evaluations": row["unique_fe_evaluations"], "duplicate_refills": row["duplicate_refills"],
                    "evaluation_budget": row["config"].get("max_unique_evaluations"), "iterations_completed": row["iterations_completed"],
                    "wall_time_seconds": row["wall_time_seconds"], "transfer_source_case": case.get("transfer_source_case"),
                    "transfer_source_distance": case.get("transfer_source_distance"), "transfer_guard_status": case.get("transfer_guard_status")})
    lines = ["# MeshPilot batch-transfer PSO VM report", "", f"- Request: `{request.request_id}`", f"- Batch order: `{request.batch_order}`",
        f"- Total wall time: {elapsed:.3f} s", "", "## Batch summary", "", "```json", json.dumps(summary, indent=2), "```", "",
        "| Run | Case | Scope | Cold FE | Transfer FE | Warm source | Guard |", "|---:|---|---|---:|---:|---|---|"]
    for c in cases:
        if c.get("status") == "completed":
            lines.append(f"| {c['execution_index']} | {c['case_id']} | {c['scope_status']} | {c['cold']['unique_fe_evaluations']} | {c['transfer']['unique_fe_evaluations']} | {c.get('transfer_source_case') or 'cold start'} | {c.get('transfer_guard_status')} |")
    (root / "batch_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def run_batch(request: BatchRequest, output_root: str | Path, gmsh_cmd="gmsh", ccx_cmd="ccx"):
    root = Path(output_root)
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True)
    cases, archive, results, started = order_cases(expand_cases(request), request.batch_order), TransferArchive(), [], time.perf_counter()
    for run_index, case in enumerate(cases, 1):
        print(f"[MESHPILOT] run={run_index} {case.case_id} scope={case.scope_status} review_required={case.review_required}")
        result = dict(_run_one_case(request, case, root, gmsh_cmd, ccx_cmd, archive)); result["execution_index"] = run_index; results.append(result)
        if result.get("status") == "completed":
            print(f"[RESULT] {case.case_id}: cold_calls={result['cold']['unique_fe_evaluations']}, transfer_calls={result['transfer']['unique_fe_evaluations']}, guard={result['transfer_guard_status']}")
    return _write_outputs(request, results, root, time.perf_counter() - started)


def parse_args():
    parser = argparse.ArgumentParser(description="MeshPilot guarded batch expansion and Batch-Transfer PSO")
    parser.add_argument("--request", required=True); parser.add_argument("--output", default="meshpilot_batch_results")
    parser.add_argument("--gmsh-cmd", default="gmsh"); parser.add_argument("--ccx-cmd", default="ccx")
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def main():
    args, request = parse_args(), None
    request = BatchRequest.from_json(args.request)
    if args.manifest_only:
        rows = []
        for index, case in enumerate(order_cases(expand_cases(request), request.batch_order), 1):
            row = asdict(case); row["execution_index"] = index; rows.append(row)
        print(json.dumps(rows, indent=2, ensure_ascii=False, default=str)); return
    print(json.dumps(run_batch(request, args.output, args.gmsh_cmd, args.ccx_cmd)["summary"], indent=2))


if __name__ == "__main__":
    main()

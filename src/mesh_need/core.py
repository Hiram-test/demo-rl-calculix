from __future__ import annotations

import json
import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

EVIDENCE_STATES = {
    "supports_current_claim",
    "challenges_current_claim",
    "unresolved",
    "not_observed",
}

NEED_FAMILIES = {
    "resolution_convergence_budget",
    "result_validity_and_extraction",
    "topology_interface_load_transfer",
    "geometry_layout_and_generation",
    "global_local_and_model_fidelity",
    "automation_and_repeatability",
}

HOTSPOT_CLASSES = {
    "bounded_response_hotspot",
    "qoi_sensitivity_or_error_hotspot",
    "singular_or_artifact_hotspot",
    "topology_or_geometry_event",
    "none_or_unknown",
}

SKILLS = {
    "bounded_hotspot_refinement",
    "qoi_guided_refinement",
    "singularity_guard",
    "topology_alignment",
    "geometry_mesh_repair",
    "model_fidelity_switch",
    "mesh_replay_guard",
    "convergence_verifier",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(*parts: Any) -> str:
    return " ".join(str(part or "") for part in parts).lower()


def _contains(text: str, terms: Iterable[str]) -> bool:
    return any(term.lower() in text for term in terms)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _vector(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [item.strip() for item in value.split(",")]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return None
    converted = [_safe_float(item) for item in value]
    if not converted or any(item is None for item in converted):
        return None
    return [float(item) for item in converted if item is not None]


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def _vector_sum(*vectors: Sequence[float]) -> list[float]:
    width = max((len(vector) for vector in vectors), default=0)
    return [sum(vector[index] if index < len(vector) else 0.0 for vector in vectors) for index in range(width)]


def classify_question(question: str, context: str = "") -> dict[str, Any]:
    """Return a deterministic routing hypothesis; it is never a final verdict."""
    combined = _text(question, context)
    rules: list[tuple[set[str], str, str, str, float, str]] = [
        (
            {"裂纹", "裂尖", "奇异", "点载荷", "点支承", "尖角", "零圆角", "无限增大", "crack", "singular", "point load", "point support", "sharp corner"},
            "result_validity_and_extraction",
            "singular_or_artifact_hotspot",
            "singularity_guard",
            0.84,
            "change_qoi_or_extraction",
        ),
        (
            {"断开", "重合节点", "共同节点", "连接", "界面", "接触", "tie", "contact", "interface", "disconnected", "duplicate node", "shared node"},
            "topology_interface_load_transfer",
            "topology_or_geometry_event",
            "topology_alignment",
            0.78,
            "repair_topology_or_geometry",
        ),
        (
            {"短边", "压印", "碎面", "几何清理", "网格生成失败", "sliver", "short edge", "imprint", "geometry cleanup", "meshing failure"},
            "geometry_layout_and_generation",
            "topology_or_geometry_event",
            "geometry_mesh_repair",
            0.76,
            "repair_topology_or_geometry",
        ),
        (
            {"梁", "壳", "实体", "子模型", "全局局部", "beam", "shell", "solid", "submodel", "global-local", "model fidelity"},
            "global_local_and_model_fidelity",
            "qoi_sensitivity_or_error_hotspot",
            "model_fidelity_switch",
            0.72,
            "change_model_fidelity",
        ),
        (
            {"重网格", "集合漂移", "载荷丢失", "路径漂移", "remesh", "set drift", "load lost", "path drift", "replay"},
            "automation_and_repeatability",
            "none_or_unknown",
            "mesh_replay_guard",
            0.70,
            "preserve_remesh_mapping",
        ),
        (
            {"qoi", "收敛", "误差", "自适应", "目标导向", "convergence", "error estimate", "goal-oriented", "adaptive"},
            "resolution_convergence_budget",
            "qoi_sensitivity_or_error_hotspot",
            "qoi_guided_refinement",
            0.70,
            "verify_before_refinement",
        ),
        (
            {"有限圆角", "局部高应力", "有限热点", "bounded hotspot", "finite radius", "local hotspot"},
            "resolution_convergence_budget",
            "bounded_response_hotspot",
            "bounded_hotspot_refinement",
            0.72,
            "refine",
        ),
    ]
    for terms, family, hotspot, skill, confidence, action in rules:
        if _contains(combined, terms):
            return {
                "contract_version": "0.3",
                "need_family": family,
                "hotspot_class": hotspot,
                "selected_skill": skill,
                "action": action,
                "confidence": confidence,
                "hypothesis_source": "deterministic_question_router",
                "reasons": [f"问题文本命中 {skill} 的可复核规则"],
                "blocked_skills": [],
                "missing_evidence": [],
                "final_verdict": None,
            }
    return {
        "contract_version": "0.3",
        "need_family": "resolution_convergence_budget",
        "hotspot_class": "none_or_unknown",
        "selected_skill": "convergence_verifier",
        "action": "collect_discriminating_evidence",
        "confidence": 0.56,
        "hypothesis_source": "deterministic_question_router",
        "reasons": ["仅凭当前文字不能定义空间热点"],
        "blocked_skills": ["bounded_hotspot_refinement", "qoi_guided_refinement"],
        "missing_evidence": ["固定 QoI", "提取位置与协议", "受控网格序列"],
        "final_verdict": None,
    }


def analyze_mesh_series(series: Any, qoi_tolerance: float = 0.02) -> dict[str, Any]:
    if not isinstance(series, Sequence) or isinstance(series, (str, bytes, bytearray)):
        return {"status": "not_observed", "reasons": ["未提供受控网格序列"], "stop_allowed": False}
    rows: list[tuple[float, float, float]] = []
    for item in series:
        if not isinstance(item, Mapping):
            continue
        h = _safe_float(item.get("h") or item.get("mesh_size"))
        peak = _safe_float(item.get("peak") or item.get("raw_peak"))
        qoi = _safe_float(item.get("qoi") or item.get("fixed_qoi"))
        if h is not None and h > 0 and peak is not None and qoi is not None:
            rows.append((h, peak, qoi))
    if len(rows) < 2:
        return {"status": "insufficient", "reasons": ["至少需要两档含 h、peak、qoi 的网格结果"], "stop_allowed": False}
    rows.sort(key=lambda row: row[0], reverse=True)
    h0, peak0, _ = rows[0]
    h1, peak1, _ = rows[-1]
    qoi_prev, qoi_last = rows[-2][2], rows[-1][2]
    peak_prev, peak_last = rows[-2][1], rows[-1][1]
    qoi_rel = abs(qoi_last - qoi_prev) / max(abs(qoi_last), abs(qoi_prev), 1e-12)
    peak_rel = abs(peak_last - peak_prev) / max(abs(peak_last), abs(peak_prev), 1e-12)
    growth_ratio = abs(peak1) / max(abs(peak0), 1e-12)
    denominator = math.log(max(h0 / h1, 1.0 + 1e-12))
    slope = math.log(max(abs(peak1) / max(abs(peak0), 1e-12), 1e-12)) / denominator if denominator else 0.0
    qoi_converged = qoi_rel <= qoi_tolerance
    singular_like = qoi_converged and growth_ratio >= 1.25 and slope >= 0.15
    bounded_like = qoi_converged and peak_rel <= max(qoi_tolerance * 2.0, 0.03)
    if singular_like:
        status = "singular_peak_with_stable_qoi"
        reasons = ["固定工程 QoI 已稳定，但原始峰值随网格细化持续增长"]
    elif bounded_like:
        status = "qoi_and_peak_converged"
        reasons = ["固定工程 QoI 与原始峰值末级变化均满足当前阈值"]
    elif qoi_converged:
        status = "qoi_converged_peak_unresolved"
        reasons = ["固定工程 QoI 已稳定，但峰值语义仍未闭合"]
    else:
        status = "qoi_not_converged"
        reasons = ["固定工程 QoI 的末级变化尚未满足容差"]
    return {
        "status": status,
        "point_count": len(rows),
        "qoi_tolerance": qoi_tolerance,
        "qoi_last_relative_change": qoi_rel,
        "peak_last_relative_change": peak_rel,
        "peak_growth_ratio": growth_ratio,
        "peak_log_slope": slope,
        "stop_allowed": qoi_converged,
        "raw_peak_allowed_as_qoi": bounded_like and not singular_like,
        "reasons": reasons,
    }


def _parse_number_list(line: str) -> list[str]:
    return [part.strip() for part in line.split(",") if part.strip()]


def inspect_calculix_inp(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {"status": "not_observed", "reasons": ["未提供 CalculiX .inp 文件"]}
    inp = Path(path)
    if not inp.is_file():
        return {"status": "not_observed", "reasons": [f"无法读取输入文件：{inp}"]}
    nodes: dict[int, tuple[float, ...]] = {}
    elements: dict[int, list[int]] = {}
    referenced_nodes: set[int] = set()
    point_load_nodes: set[int] = set()
    boundary_nodes: set[int] = set()
    mode = ""
    for raw in inp.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("**"):
            continue
        if line.startswith("*"):
            mode = line.split(",", 1)[0].upper()
            continue
        values = _parse_number_list(line)
        try:
            if mode == "*NODE" and len(values) >= 3:
                nodes[int(values[0])] = tuple(float(value) for value in values[1:])
            elif mode == "*ELEMENT" and len(values) >= 3:
                element_id = int(values[0])
                connectivity = [int(value) for value in values[1:]]
                elements[element_id] = connectivity
                referenced_nodes.update(connectivity)
            elif mode == "*CLOAD" and values:
                point_load_nodes.add(int(values[0]))
            elif mode == "*BOUNDARY" and values:
                boundary_nodes.add(int(values[0]))
        except ValueError:
            continue
    undefined = sorted(referenced_nodes - nodes.keys())
    unused = sorted(nodes.keys() - referenced_nodes)
    coord_to_ids: dict[tuple[float, ...], list[int]] = defaultdict(list)
    for node_id, coordinate in nodes.items():
        coord_to_ids[tuple(round(value, 10) for value in coordinate)].append(node_id)
    duplicates = [ids for ids in coord_to_ids.values() if len(ids) > 1]
    adjacency: dict[int, set[int]] = defaultdict(set)
    for connectivity in elements.values():
        present = [node for node in connectivity if node in nodes]
        for node in present:
            adjacency[node].update(other for other in present if other != node)
    active = set(referenced_nodes & nodes.keys())
    components: list[list[int]] = []
    remaining = set(active)
    while remaining:
        start = remaining.pop()
        queue = deque([start])
        component = {start}
        while queue:
            node = queue.popleft()
            for neighbor in adjacency.get(node, set()):
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    risks: list[str] = []
    if len(components) > 1:
        risks.append(f"发现 {len(components)} 个断开的节点连通分量；接触体可能合法，但需显式证明载荷传递")
    if duplicates:
        risks.append(f"发现 {len(duplicates)} 组坐标重合节点")
    if undefined:
        risks.append(f"单元引用了 {len(undefined)} 个未定义节点")
    if unused:
        risks.append(f"发现 {len(unused)} 个未被单元使用的节点")
    if point_load_nodes:
        risks.append(f"发现作用于 {len(point_load_nodes)} 个节点的集中载荷；需评估局部奇异性")
    if boundary_nodes:
        risks.append(f"发现作用于 {len(boundary_nodes)} 个节点/节点集标识的边界条目；需核对作用域")
    return {
        "status": "risks_observed" if risks else "no_obvious_connectivity_risk",
        "node_count": len(nodes),
        "element_count": len(elements),
        "component_count": len(components),
        "duplicate_coordinate_groups": duplicates,
        "undefined_node_references": undefined,
        "unused_nodes": unused,
        "point_load_nodes": sorted(point_load_nodes),
        "boundary_nodes": sorted(boundary_nodes),
        "risks": risks,
    }


def evaluate_force_moment(case: Mapping[str, Any]) -> dict[str, Any]:
    provided_force = _safe_float(case.get("force_relative_residual"))
    provided_moment = _safe_float(case.get("moment_relative_residual"))
    tolerance = _safe_float(case.get("equilibrium_tolerance"), 0.02) or 0.02
    if provided_force is not None or provided_moment is not None:
        force_residual = provided_force if provided_force is not None else 0.0
        moment_residual = provided_moment if provided_moment is not None else 0.0
    else:
        external_force = _vector(case.get("external_force"))
        reaction_force = _vector(case.get("reaction_force"))
        external_moment = _vector(case.get("external_moment"))
        reaction_moment = _vector(case.get("reaction_moment"))
        if external_force is None or reaction_force is None:
            return {"status": "not_observed", "tolerance": tolerance}
        force_residual = _norm(_vector_sum(external_force, reaction_force)) / max(_norm(external_force), 1e-12)
        if external_moment is not None and reaction_moment is not None:
            moment_residual = _norm(_vector_sum(external_moment, reaction_moment)) / max(_norm(external_moment), 1e-12)
        else:
            moment_residual = 0.0
    status = "within_tolerance" if force_residual <= tolerance and moment_residual <= tolerance else "outside_tolerance"
    return {
        "status": status,
        "force_relative_residual": force_residual,
        "moment_relative_residual": moment_residual,
        "tolerance": tolerance,
    }


def evaluate_energy(case: Mapping[str, Any]) -> dict[str, Any]:
    history = case.get("energy_history")
    closure_tolerance = _safe_float(case.get("energy_closure_tolerance"), 0.03) or 0.03
    artificial_limit = _safe_float(case.get("artificial_energy_ratio_limit"), 0.05) or 0.05
    if not isinstance(history, Sequence) or isinstance(history, (str, bytes, bytearray)) or not history:
        return {"status": "not_observed", "closure_tolerance": closure_tolerance, "artificial_energy_ratio_limit": artificial_limit}
    closure_residuals: list[float] = []
    artificial_ratios: list[float] = []
    for row in history:
        if not isinstance(row, Mapping):
            continue
        external = _safe_float(row.get("external_work"))
        if external is None:
            continue
        internal = _safe_float(row.get("internal_energy"), 0.0) or 0.0
        kinetic = _safe_float(row.get("kinetic_energy"), 0.0) or 0.0
        damping = _safe_float(row.get("damping_dissipation"), 0.0) or 0.0
        plastic = _safe_float(row.get("plastic_dissipation"), 0.0) or 0.0
        contact = _safe_float(row.get("contact_dissipation"), 0.0) or 0.0
        artificial = abs(_safe_float(row.get("artificial_energy"), 0.0) or 0.0)
        accounted = internal + kinetic + damping + plastic + contact + artificial
        closure_residuals.append(abs(external - accounted) / max(abs(external), 1e-12))
        artificial_ratios.append(artificial / max(abs(internal), abs(external), 1e-12))
    if not closure_residuals:
        return {"status": "not_observed", "closure_tolerance": closure_tolerance, "artificial_energy_ratio_limit": artificial_limit}
    max_closure = max(closure_residuals)
    max_artificial = max(artificial_ratios)
    status = "within_tolerance" if max_closure <= closure_tolerance and max_artificial <= artificial_limit else "outside_tolerance"
    return {
        "status": status,
        "max_relative_closure_residual": max_closure,
        "max_artificial_energy_ratio": max_artificial,
        "closure_tolerance": closure_tolerance,
        "artificial_energy_ratio_limit": artificial_limit,
        "sign_convention": "positive user-supplied energy magnitudes; verify against solver definitions",
    }


def compare_replay_manifests(before: Any, after: Any) -> dict[str, Any]:
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return {"status": "not_observed", "drift": {}}
    keys = ("sets", "loads", "contacts", "paths", "supports", "qoi_locations")
    drift: dict[str, Any] = {}
    for key in keys:
        if before.get(key) != after.get(key):
            drift[key] = {"before": before.get(key), "after": after.get(key)}
    return {"status": "drift_observed" if drift else "stable", "drift": drift}


def apply_guards(diagnosis: dict[str, Any], case: Mapping[str, Any], topology: Mapping[str, Any], trend: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(diagnosis)
    reasons = list(result.get("reasons", []))
    blocked = set(result.get("blocked_skills", []))
    if trend.get("status") == "singular_peak_with_stable_qoi":
        result.update(
            need_family="result_validity_and_extraction",
            hotspot_class="singular_or_artifact_hotspot",
            selected_skill="singularity_guard",
            action="change_qoi_or_extraction",
            confidence=max(float(result.get("confidence", 0.0)), 0.90),
            hypothesis_source="numerical_guard_override",
        )
        blocked.update({"bounded_hotspot_refinement", "qoi_guided_refinement"})
        reasons.append("数值趋势覆盖了原始路由：禁止追逐发散峰值")
    topology_risks = topology.get("risks", []) if isinstance(topology, Mapping) else []
    serious_topology = any("断开" in risk or "未定义节点" in risk or "重合节点" in risk for risk in topology_risks)
    if serious_topology:
        result.update(
            need_family="topology_interface_load_transfer",
            hotspot_class="topology_or_geometry_event",
            selected_skill="topology_alignment",
            action="repair_topology_or_geometry",
            confidence=max(float(result.get("confidence", 0.0)), 0.88),
            hypothesis_source="model_connectivity_guard_override",
        )
        blocked.update({"bounded_hotspot_refinement", "qoi_guided_refinement"})
        reasons.append("模型连通证据优先于普通热点细化")
    qoi = case.get("qoi")
    if result.get("selected_skill") in {"bounded_hotspot_refinement", "qoi_guided_refinement", "convergence_verifier"}:
        if not isinstance(qoi, Mapping) or not all(qoi.get(key) not in (None, "") for key in ("name", "location", "extraction_method", "tolerance")):
            blocked.update({"bounded_hotspot_refinement", "qoi_guided_refinement"})
            missing = set(result.get("missing_evidence", []))
            missing.update({"固定 QoI 名称", "固定物理位置", "固定提取协议", "容差"})
            result["missing_evidence"] = sorted(missing)
            reasons.append("固定 QoI 合同不完整，细化结果不得进入停止判断")
    replay = compare_replay_manifests(case.get("manifest_before"), case.get("manifest_after"))
    if replay.get("status") == "drift_observed":
        result.update(
            need_family="automation_and_repeatability",
            hotspot_class="none_or_unknown",
            selected_skill="mesh_replay_guard",
            action="preserve_remesh_mapping",
            hypothesis_source="remesh_scope_guard_override",
        )
        blocked.update({"bounded_hotspot_refinement", "qoi_guided_refinement"})
        reasons.append("重网格作用域漂移阻止当前结果验收")
    result["blocked_skills"] = sorted(blocked)
    result["reasons"] = reasons
    result["final_verdict"] = None
    return result


def _evidence_item(
    evidence_id: str,
    question: str,
    state: str,
    observations: Sequence[str],
    limitations: Sequence[str],
    next_checks: Sequence[str],
    source: str,
    metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if state not in EVIDENCE_STATES:
        raise ValueError(f"invalid evidence state: {state}")
    return {
        "evidence_id": evidence_id,
        "question": question,
        "state": state,
        "observations": list(observations),
        "limitations": list(limitations),
        "next_checks": list(next_checks),
        "source": source,
        "metrics": dict(metrics or {}),
    }


def build_evidence_ledger(
    case: Mapping[str, Any],
    diagnosis: Mapping[str, Any],
    topology: Mapping[str, Any],
    trend: Mapping[str, Any],
    equilibrium: Mapping[str, Any],
    energy: Mapping[str, Any],
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    intended_use = str(case.get("intended_use") or "")
    current_claim = str(case.get("current_claim") or "")
    qoi = case.get("qoi")
    problem_complete = bool(intended_use and current_claim and isinstance(qoi, Mapping) and qoi.get("name"))
    items.append(_evidence_item(
        "problem_definition",
        "当前分析用途、工程结论与 QoI 是否被明确限定？",
        "supports_current_claim" if problem_complete else "unresolved",
        ["预定用途、当前结论和固定 QoI 均已显式记录"] if problem_complete else ["问题定义仍有缺项"],
        ["明确的问题定义不证明模型表达或数值结果正确"],
        ["确认工况、结果提取位置和容差在所有模型版本中保持一致"],
        "runtime",
    ))
    items.append(_evidence_item(
        "ai_or_rule_hypothesis",
        "AI/规则当前提出了什么建模与网格假设？",
        "unresolved",
        [
            f"当前诊断热点类别：{diagnosis.get('hotspot_class', 'unknown')}",
            f"当前候选 Skill：{diagnosis.get('selected_skill', 'unknown')}",
        ],
        ["诊断本身是待检验假设，不是支持它自己的独立证据", "置信度表示路由确定性，不等于工程结论正确概率"],
        ["执行所选 Skill 并用独立数值证据更新台账"],
        str(diagnosis.get("hypothesis_source") or "runtime"),
        {"selected_skill": diagnosis.get("selected_skill"), "action": diagnosis.get("action"), "confidence": diagnosis.get("confidence"), "blocked_skills": diagnosis.get("blocked_skills", [])},
    ))
    topology_status = topology.get("status")
    if topology_status == "not_observed":
        topology_state = "not_observed"
        topology_obs = list(topology.get("reasons", ["未提供模型文件"]))
    elif topology_status == "risks_observed":
        topology_state = "challenges_current_claim"
        topology_obs = list(topology.get("risks", []))
    else:
        topology_state = "supports_current_claim"
        topology_obs = ["未发现明显的节点连通或引用风险"]
    items.append(_evidence_item(
        "topology_and_point_actions",
        "网格连接、点荷载和点约束是否暴露了建模风险？",
        topology_state,
        topology_obs,
        ["输入文件静态检查不能证明接触、约束方程或真实载荷路径正确"],
        ["结合求解器反力、接触状态和变形模式复核载荷传递"],
        "runtime",
        topology,
    ))
    trend_status = trend.get("status")
    if trend_status in {"qoi_and_peak_converged", "qoi_converged_peak_unresolved"}:
        trend_state = "supports_current_claim"
    elif trend_status in {"singular_peak_with_stable_qoi", "qoi_not_converged"}:
        trend_state = "challenges_current_claim"
    elif trend_status == "not_observed":
        trend_state = "not_observed"
    else:
        trend_state = "unresolved"
    items.append(_evidence_item(
        "mesh_response_trend",
        "固定 QoI 与原始峰值随网格变化的趋势支持当前解释吗？",
        trend_state,
        list(trend.get("reasons", [])),
        ["网格趋势只检验当前模型和结果定义的离散敏感性", "QoI 趋稳不能证明材料、载荷、约束或模型形式代表现实"],
        ["结合平衡、能量和模型形式对照继续积累证据"],
        "runtime",
        trend,
    ))
    equilibrium_status = equilibrium.get("status")
    equilibrium_state = "supports_current_claim" if equilibrium_status == "within_tolerance" else "challenges_current_claim" if equilibrium_status == "outside_tolerance" else "not_observed"
    items.append(_evidence_item(
        "force_moment_equilibrium",
        "外载荷与反力/反力矩是否在当前提取口径下闭合？",
        equilibrium_state,
        [f"力平衡状态：{equilibrium_status}"],
        ["平衡只说明当前数学模型收支自洽，不证明载荷和约束代表现实", "遗漏惯性、接触或约束反力会造成虚假的不平衡"],
        ["检查全部反力来源、变形模式和模型形式"],
        "runtime",
        equilibrium,
    ))
    energy_status = energy.get("status")
    energy_state = "supports_current_claim" if energy_status == "within_tolerance" else "challenges_current_claim" if energy_status == "outside_tolerance" else "not_observed"
    items.append(_evidence_item(
        "energy_consistency",
        "在明确的符号约定下，外功与内部/动能/耗散等能量项是否闭合？",
        energy_state,
        [f"能量检查状态：{energy_status}"],
        ["必须与具体求解器输出定义核对符号约定", "错误模型也可能能量闭合", "人工能量阈值是可配置警戒值"],
        ["结合载荷路径、变形模式和独立模型对照解释能量证据"],
        "runtime",
        energy,
    ))
    counts = {state: sum(item["state"] == state for item in items) for state in sorted(EVIDENCE_STATES)}
    return {
        "contract_version": "0.1",
        "case_id": str(case.get("case_id") or "mesh-need-case"),
        "intended_use": intended_use,
        "current_claim": current_claim,
        "principle": "No final judge: every entry is limited evidence for a stated use and may be revised by later evidence.",
        "generated_at": _utc_now(),
        "state_counts": counts,
        "items": items,
        "final_verdict": None,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _gmsh_bounded_field(hotspots: Sequence[Mapping[str, Any]]) -> str:
    lines = ["// Generated by mesh-need MVP; merge after reviewing geometry tags."]
    fields: list[int] = []
    field_id = 1
    for hotspot in hotspots:
        shape = str(hotspot.get("shape") or "box").lower()
        if shape == "ball":
            lines.extend([
                f"Field[{field_id}] = Ball;",
                f"Field[{field_id}].XCenter = {float(hotspot.get('x', 0.0))};",
                f"Field[{field_id}].YCenter = {float(hotspot.get('y', 0.0))};",
                f"Field[{field_id}].ZCenter = {float(hotspot.get('z', 0.0))};",
                f"Field[{field_id}].Radius = {float(hotspot.get('radius', 1.0))};",
                f"Field[{field_id}].VIn = {float(hotspot.get('size_inside', 0.5))};",
                f"Field[{field_id}].VOut = {float(hotspot.get('size_outside', 2.0))};",
            ])
        else:
            lines.extend([
                f"Field[{field_id}] = Box;",
                f"Field[{field_id}].XMin = {float(hotspot.get('xmin', 0.0))};",
                f"Field[{field_id}].XMax = {float(hotspot.get('xmax', 1.0))};",
                f"Field[{field_id}].YMin = {float(hotspot.get('ymin', 0.0))};",
                f"Field[{field_id}].YMax = {float(hotspot.get('ymax', 1.0))};",
                f"Field[{field_id}].ZMin = {float(hotspot.get('zmin', 0.0))};",
                f"Field[{field_id}].ZMax = {float(hotspot.get('zmax', 1.0))};",
                f"Field[{field_id}].VIn = {float(hotspot.get('size_inside', 0.5))};",
                f"Field[{field_id}].VOut = {float(hotspot.get('size_outside', 2.0))};",
            ])
        fields.append(field_id)
        field_id += 1
    if fields:
        if len(fields) == 1:
            lines.append(f"Background Field = {fields[0]};")
        else:
            lines.append(f"Field[{field_id}] = Min;")
            lines.append(f"Field[{field_id}].FieldsList = {{{', '.join(map(str, fields))}}};")
            lines.append(f"Background Field = {field_id};")
    return "\n".join(lines) + "\n"


def _gmsh_qoi_files(points: Sequence[Mapping[str, Any]]) -> tuple[str, str]:
    pos_lines = ['View "qoi_indicator" {']
    for point in points:
        x = float(point.get("x", 0.0)); y = float(point.get("y", 0.0)); z = float(point.get("z", 0.0))
        indicator = max(float(point.get("indicator", 0.0)), 1e-12)
        size = float(point.get("target_size", 1.0 / math.sqrt(indicator)))
        pos_lines.append(f"SP({x},{y},{z}){{{size}}};")
    pos_lines.append("};")
    geo = 'Merge "qoi_target_size.pos";\nField[1] = PostView;\nField[1].ViewIndex = 0;\nBackground Field = 1;\n'
    return "\n".join(pos_lines) + "\n", geo


def execute_skill(diagnosis: Mapping[str, Any], case: Mapping[str, Any], output_dir: Path, trend: Mapping[str, Any], topology: Mapping[str, Any]) -> dict[str, Any]:
    skill = str(diagnosis.get("selected_skill") or "convergence_verifier")
    blocked = set(diagnosis.get("blocked_skills", []))
    generated: list[str] = []
    result: dict[str, Any] = {"skill": skill, "status": "completed", "generated_files": generated}
    if skill in blocked:
        return {"skill": skill, "status": "blocked", "reasons": diagnosis.get("reasons", []), "generated_files": []}
    if skill == "bounded_hotspot_refinement":
        hotspots = case.get("hotspots")
        if not isinstance(hotspots, Sequence) or isinstance(hotspots, (str, bytes, bytearray)) or not hotspots:
            return {"skill": skill, "status": "needs_input", "required": ["hotspots"], "generated_files": []}
        path = output_dir / "bounded_hotspot_field.geo"
        path.write_text(_gmsh_bounded_field([item for item in hotspots if isinstance(item, Mapping)]), encoding="utf-8")
        generated.append(path.name)
    elif skill == "qoi_guided_refinement":
        points = case.get("qoi_indicator_points")
        if isinstance(points, Sequence) and not isinstance(points, (str, bytes, bytearray)) and points:
            pos, geo = _gmsh_qoi_files([item for item in points if isinstance(item, Mapping)])
            pos_path = output_dir / "qoi_target_size.pos"; geo_path = output_dir / "qoi_postview_field.geo"
            pos_path.write_text(pos, encoding="utf-8"); geo_path.write_text(geo, encoding="utf-8")
            generated.extend([pos_path.name, geo_path.name])
        else:
            result.update(status="needs_input", required=["qoi_indicator_points"])
    elif skill == "singularity_guard":
        result["decision"] = "do_not_optimize_raw_peak"
        result["trend"] = dict(trend)
    elif skill == "topology_alignment":
        result["inspection"] = dict(topology)
    elif skill == "geometry_mesh_repair":
        result["repair_sequence"] = ["定位失败几何实体", "清理短边/碎面/残留压印", "保持构造事件点与命名作用域", "重网格并复核 QoI"]
    elif skill == "model_fidelity_switch":
        result["comparison_plan"] = ["固定工程 QoI 与提取协议", "建立梁/壳/实体或全局—局部对照", "检查边界距离与载荷传递", "用独立证据选择最小充分模型"]
    elif skill == "mesh_replay_guard":
        result["replay"] = compare_replay_manifests(case.get("manifest_before"), case.get("manifest_after"))
    else:
        result["trend"] = dict(trend)
    hotspots = case.get("hotspots")
    qoi = case.get("qoi")
    if (
        isinstance(hotspots, Sequence) and not isinstance(hotspots, (str, bytes, bytearray)) and len(hotspots) > 1
        and bool(case.get("budget_conflict"))
        and str(case.get("region_interaction") or "").lower() in {"strong", "measured_strong"}
        and isinstance(qoi, Mapping) and all(qoi.get(key) not in (None, "") for key in ("name", "location", "extraction_method", "tolerance"))
        and diagnosis.get("hotspot_class") not in {"singular_or_artifact_hotspot", "topology_or_geometry_event"}
    ):
        pso_job = {
            "job_version": "0.1",
            "optimizer": "external_multi_hotspot_pso",
            "case_id": case.get("case_id", "mesh-need-case"),
            "preconditions": {"hotspot_validity": diagnosis.get("hotspot_class"), "region_interaction": "measured_strong", "fixed_qoi": True, "singular_peak_target": False},
            "qoi": qoi,
            "candidate_regions": list(hotspots),
            "mesh_levels": case.get("mesh_levels", [0, 1, 2, 3]),
            "budget": case.get("budget", {"max_solver_runs": 32}),
            "objective_policy": {"primary": "meet_fixed_qoi_tolerance", "secondary": "minimize_mesh_or_solver_cost", "forbidden": ["optimize_raw_peak_at_singular_location", "change_qoi_location_between_candidates"]},
            "verification": ["same_qoi_location_and_extraction_for_every_particle", "mesh_quality_constraints", "reaction_or_energy_balance", "independent_final_mesh_check"],
        }
        path = output_dir / "external_hotspot_pso_job.json"
        _write_json(path, pso_job)
        generated.append(path.name)
        result["optimizer_export"] = path.name
    return result


def make_ai_prompt(case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prompt_version": "0.3",
        "role": "Propose a falsifiable FEA mesh/model diagnosis. Do not issue a final correctness verdict.",
        "required_output_schema": "schemas/mesh_need_diagnosis.schema.json",
        "case": dict(case),
        "instructions": [
            "Bind every mesh recommendation to a fixed engineering QoI and extraction protocol.",
            "Separate bounded response hotspots, QoI/error hotspots, singular/artifact peaks, and topology/geometry events.",
            "Treat force/moment balance, energy consistency, mesh trends, and model comparisons as limited evidence, not a final judge.",
            "State missing evidence and blocked methods explicitly.",
        ],
    }


def _normalise_ai_proposal(proposal: Any) -> dict[str, Any] | None:
    if not isinstance(proposal, Mapping):
        return None
    required = {"need_family", "hotspot_class", "selected_skill", "action"}
    if not required.issubset(proposal):
        return None
    result = dict(proposal)
    if result.get("need_family") not in NEED_FAMILIES or result.get("hotspot_class") not in HOTSPOT_CLASSES or result.get("selected_skill") not in SKILLS:
        return None
    result.setdefault("contract_version", "0.3")
    result.setdefault("confidence", 0.5)
    result.setdefault("hypothesis_source", "provider_neutral_ai_proposal")
    result.setdefault("reasons", ["外部模型提出候选诊断；等待保护层复核"])
    result.setdefault("blocked_skills", [])
    result.setdefault("missing_evidence", [])
    result["final_verdict"] = None
    return result


def run_pipeline(case: Mapping[str, Any], output_dir: str | Path, ai_proposal: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    case_data = dict(case)
    case_data.setdefault("case_id", f"mesh-need-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
    question = str(case_data.get("question") or "")
    base = _normalise_ai_proposal(ai_proposal) or classify_question(question, str(case_data.get("context") or ""))
    qoi = case_data.get("qoi") if isinstance(case_data.get("qoi"), Mapping) else {}
    tolerance = _safe_float(qoi.get("tolerance"), 0.02) or 0.02
    trend = analyze_mesh_series(case_data.get("mesh_series"), tolerance)
    topology = inspect_calculix_inp(case_data.get("calculix_inp"))
    diagnosis = apply_guards(base, case_data, topology, trend)
    equilibrium = evaluate_force_moment(case_data)
    energy = evaluate_energy(case_data)
    ledger = build_evidence_ledger(case_data, diagnosis, topology, trend, equilibrium, energy)
    skill_result = execute_skill(diagnosis, case_data, out, trend, topology)
    prompt = make_ai_prompt(case_data)
    _write_json(out / "case.json", case_data)
    _write_json(out / "diagnosis.json", diagnosis)
    _write_json(out / "ai_prompt.json", prompt)
    _write_json(out / "skill_result.json", skill_result)
    _write_json(out / "evidence_ledger.json", ledger)
    summary = {
        "case_id": case_data["case_id"],
        "selected_skill": diagnosis.get("selected_skill"),
        "skill_status": skill_result.get("status"),
        "evidence_state_counts": ledger["state_counts"],
        "generated_files": sorted(path.name for path in out.iterdir() if path.is_file()),
        "final_verdict": None,
    }
    _write_json(out / "pipeline-summary.json", summary)
    return summary

from __future__ import annotations  # 启用现代类型注解并保持运行环境兼容。

import json  # 读取前轮公开证据并保存隐藏映射和执行审计。
import re  # 从中英文自然语言提案中提取物理距离和公式意图。
from math import pi, sqrt  # 执行模型提出的线弹性裂尖派生量计算。
from pathlib import Path  # 管理冻结轮次目录和前轮公开反馈文件。
from typing import Any  # 表示动态提案、映射和有限元结果结构。

import numpy as np  # 处理裂纹面节点坐标、位移和对称平均。

from . import executor_adapter_v5 as previous_adapter  # 复用任务级路线控制和既有隐藏执行器栈。
from .contracts import canonical_json  # 规范序列化提案用于完整性检查。
from .contracts import sha256_text  # 计算冻结提案摘要以验证执行器没有改写内容。
from .contracts import verify_frozen_proposal  # 读取并验证已经冻结的模型提案。
from .executor import _load_backend  # 复用只读有限元后端、网格和真实位移解。

DISPLACEMENT_PROBE_OPERATION = "crack_face_displacement_probe"  # 定义只存在于隐藏审计层的裂纹面位移提取操作。
DISPLACEMENT_K_OPERATION = "crack_face_displacement_k_sequence"  # 定义裂纹面位移提取与K换算联合操作。
DISPLACEMENT_MATERIAL_OPERATION = "crack_face_displacement_with_material_request"  # 定义可计算位移证据与外部材料请求联合操作。
STRESS_POSTPROCESS_OPERATION = "stress_asymptotic_postprocess"  # 定义只基于既有公开应力数据的纯后处理操作。

_DISPLACEMENT_TERMS = ("裂纹面位移", "节点位移", "位移分量", "张开位移", "开口位移", "u_y", "uy", "crack-face displacement", "crack face displacement", "nodal displacement", "opening displacement", "displacement component")  # 定义中英文位移提取语义。
_FRACTURE_TERMS = ("应力强度因子", "k_i", "ki", "stress intensity factor", "断裂参数", "fracture parameter")  # 定义中英文断裂评价量语义。
_MATERIAL_TERMS = ("材料", "塑性", "屈服", "硬化", "本构", "断裂韧性", "material", "plastic", "yield", "hardening", "constitutive", "toughness")  # 定义外部材料事实语义。
_STRESS_FORMULA_TERMS = ("σ_yy", "sigma_yy", "sigma yy", "法向应力", "应力公式", "stress formula", "√(2πr)", "sqrt(2*pi*r)", "sqrt(2πr)")  # 定义基于固定距离应力反算K的公式语义。
_NO_NEW_SOLVE_TERMS = ("不进行新的有限元", "不进行任何新的有限元", "不进行新的仿真", "不进行任何新的仿真", "仅基于已有", "基于已有数据", "existing data", "existing results", "without a new", "no new simulation", "post-process", "postprocess")  # 定义纯后处理和禁止新求解语义。
_REFINE_TERMS = ("加密网格", "细化网格", "减小单元", "增加网格", "refine the mesh", "mesh refinement", "reduce the element size", "increase mesh resolution", "目标尺寸", "target mesh size")  # 定义真正改变网格分辨率的语义。


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:  # 检查文本是否包含任一中英文语义短语。
    return any(term in text for term in terms)  # 任一短语出现时返回真。


def _proposal_text(proposal: dict[str, Any]) -> str:  # 把冻结提案转换为统一小写检索文本。
    return json.dumps(proposal, ensure_ascii=False, sort_keys=True).lower()  # 保留全部公开自然语言字段用于忠实分类。


def _write_mapping(round_dir: Path, proposal_hash: str, operation: str, reason: str) -> dict[str, Any]:  # 统一保存模型不可见的隐藏映射收据。
    mapping = {"proposal_sha256": proposal_hash, "operation": operation, "mapping_reason": reason, "proposal_unchanged": True}  # 组织完整映射记录。
    (round_dir / "mapping_receipt.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存隐藏映射收据。
    return mapping  # 返回后续执行阶段使用的映射对象。


def _write_execution(round_dir: Path, proposal_hash: str, operation: str, raw_result: dict[str, Any], actual_parameters: dict[str, Any], feedback: dict[str, Any]) -> dict[str, Any]:  # 保存专用隐藏操作的审计和公开反馈。
    audit = {"proposal_sha256": proposal_hash, "internal_operation": operation, "raw_result": raw_result, "actual_parameters": actual_parameters, "public_feedback": feedback}  # 组织完整模型不可见执行审计。
    (round_dir / "execution_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存内部操作和原始结果。
    (round_dir / "public_feedback.json").write_text(json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存下一轮唯一允许看到的公开反馈。
    return feedback  # 返回公开物理证据。


def _extract_distances(text: str, default: list[float] | None = None) -> list[float]:  # 从提案中提取距裂尖的目标物理距离。
    patterns = (r"r\s*[=:：]\s*(\d+(?:\.\d+)?)\s*(?:mm|毫米)", r"(?:距离|距)裂尖[^\d]{0,18}(\d+(?:\.\d+)?)\s*(?:mm|毫米)", r"(?:distance|behind the crack tip|from the crack tip)[^\d]{0,18}(\d+(?:\.\d+)?)\s*(?:mm|millimeter)")  # 定义中英文裂尖距离表达式。
    values: list[float] = []  # 初始化距离数组。
    for pattern in patterns:  # 依次检查候选表达式。
        for match in re.findall(pattern, text, flags=re.IGNORECASE):  # 提取当前表达式下的全部数值。
            value = float(match)  # 把捕获值转换为毫米浮点数。
            if 0.0 < value <= 30.0 and value not in values:  # 只保留后端允许范围内的唯一距离。
                values.append(value)  # 保存模型明确请求的距离。
    if values:  # 检查是否提取到显式距离。
        return values[:6]  # 限制单轮距离数量以控制结果规模。
    return list(default or [2.5])  # 在没有明确距离时使用当前问题的最小公开距离。


def _crack_face_nodes(mesh: Any, edge_set: str) -> list[int]:  # 获取指定裂纹面的全部节点编号。
    return sorted({int(node) for edge in mesh.edge_sets.get(edge_set, []) for node in edge})  # 从裂纹边集合去重并排序节点。


def _nearest_interior_face_node(mesh: Any, node_ids: list[int], target_x: float, tip_sign: int, half_crack: float) -> int:  # 在指定裂尖后方选择最近的真实裂纹面节点。
    eps = 1.0e-9 * max(abs(half_crack), 1.0)  # 定义排除共享裂尖节点的坐标容差。
    candidates: list[int] = []  # 初始化当前裂尖一侧的候选节点。
    for node_id in node_ids:  # 遍历当前裂纹面全部节点。
        x_value = float(mesh.nodes[node_id, 0])  # 读取节点横坐标。
        if tip_sign > 0 and 0.0 <= x_value < half_crack - eps:  # 选择右裂尖后方且排除共享裂尖的节点。
            candidates.append(node_id)  # 保存右侧裂纹面候选节点。
        if tip_sign < 0 and -half_crack + eps < x_value <= 0.0:  # 选择左裂尖后方且排除共享裂尖的节点。
            candidates.append(node_id)  # 保存左侧裂纹面候选节点。
    if not candidates:  # 检查网格是否具有可用裂纹面内部节点。
        raise RuntimeError("no interior crack-face nodes are available for displacement extraction")  # 在拓扑不支持时明确失败。
    return min(candidates, key=lambda node_id: abs(float(mesh.nodes[node_id, 0]) - target_x))  # 返回横坐标最接近目标位置的裂纹面节点。


def _displacement_rows(backend: Any, distances: list[float], include_k: bool) -> list[dict[str, Any]]:  # 对全部初始网格提取裂纹面张开位移并可选换算K。
    rows: list[dict[str, Any]] = []  # 初始化公开位移证据列表。
    kappa = (3.0 - backend.POISSON) / (1.0 + backend.POISSON)  # 计算二维平面应力裂尖位移参数κ。
    modulus_factor = backend.YOUNG / ((1.0 + backend.POISSON) * (kappa + 1.0))  # 计算由半张开位移换算K的材料系数。
    for nx in backend.INITIAL_LEVELS:  # 遍历用户已看到的三档初始网格。
        result = backend._solve(int(nx))  # 读取或执行当前真实有限元求解。
        mesh = result["mesh"]  # 读取当前网格对象。
        solution = result["solution"]  # 读取包含真实节点位移的解对象。
        upper_nodes = _crack_face_nodes(mesh, "crack_upper")  # 获取上裂纹面节点集合。
        lower_nodes = _crack_face_nodes(mesh, "crack_lower")  # 获取下裂纹面节点集合。
        for requested_distance in distances:  # 逐个处理模型请求的物理距离。
            tip_samples: list[dict[str, Any]] = []  # 初始化左右裂尖的位移样本。
            half_openings: list[float] = []  # 初始化两个裂尖的半张开位移。
            actual_distances: list[float] = []  # 初始化网格实际可用距离。
            for tip_name, tip_sign in (("right", 1), ("left", -1)):  # 对两个对称裂尖分别提取裂纹面位移。
                target_x = tip_sign * (backend.HALF_CRACK - requested_distance)  # 计算裂尖后方目标横坐标。
                upper_id = _nearest_interior_face_node(mesh, upper_nodes, target_x, tip_sign, backend.HALF_CRACK)  # 选择上裂纹面最近内部节点。
                lower_id = _nearest_interior_face_node(mesh, lower_nodes, target_x, tip_sign, backend.HALF_CRACK)  # 选择下裂纹面最近内部节点。
                upper_x = float(mesh.nodes[upper_id, 0])  # 读取上裂纹面实际横坐标。
                lower_x = float(mesh.nodes[lower_id, 0])  # 读取下裂纹面实际横坐标。
                actual_x = 0.5 * (upper_x + lower_x)  # 对上下表面坐标取平均以处理浮点误差。
                actual_distance = backend.HALF_CRACK - actual_x if tip_sign > 0 else actual_x + backend.HALF_CRACK  # 计算实际距当前裂尖的距离。
                upper_uy = float(solution.displacements[upper_id, 1])  # 读取上裂纹面竖向位移。
                lower_uy = float(solution.displacements[lower_id, 1])  # 读取下裂纹面竖向位移。
                opening = upper_uy - lower_uy  # 计算上下裂纹面的完整张开位移差。
                half_opening = 0.5 * abs(opening)  # 计算用于对称裂尖场换算的半张开位移。
                half_openings.append(half_opening)  # 保存当前裂尖半张开位移。
                actual_distances.append(actual_distance)  # 保存当前裂尖实际取样距离。
                tip_samples.append({"tip": tip_name, "target_x_mm": float(target_x), "actual_x_mm": float(actual_x), "actual_distance_from_tip_mm": float(actual_distance), "upper_node": int(upper_id), "lower_node": int(lower_id), "upper_uy_mm": upper_uy, "lower_uy_mm": lower_uy, "opening_displacement_mm": float(opening), "half_opening_mm": float(half_opening)})  # 记录完整可审计节点和位移数据。
            mean_half_opening = float(np.mean(half_openings))  # 计算左右裂尖平均半张开位移。
            mean_distance = float(np.mean(actual_distances))  # 计算左右裂尖平均实际取样距离。
            row: dict[str, Any] = {"nx": int(nx), "h_local_mm": float(backend.WIDTH / nx), "requested_distance_from_tip_mm": float(requested_distance), "actual_distance_from_tip_mm": mean_distance, "mean_half_opening_mm": mean_half_opening, "tip_samples": tip_samples}  # 组织当前网格公开位移证据。
            if include_k:  # 检查模型是否要求把位移场换算为应力强度因子。
                k_value = modulus_factor * mean_half_opening * sqrt(2.0 * pi / mean_distance)  # 使用标准平面应力Mode-I裂纹面位移渐近式换算K。
                row["stress_intensity_from_opening_mpa_sqrt_mm"] = float(k_value)  # 保存当前网格位移法K估计值。
            rows.append(row)  # 追加当前网格和距离的公开结果。
    return rows  # 返回全部三档网格的真实位移证据。


def _relative_spread(values: list[float]) -> float | None:  # 计算一组派生评价量的最大最小相对离散度。
    if not values:  # 检查输入是否为空。
        return None  # 在没有数值时返回空值。
    mean_value = float(np.mean(values))  # 计算平均值作为相对离散分母。
    if abs(mean_value) <= 1.0e-30:  # 防止零均值导致除零。
        return None  # 在无有效尺度时返回空值。
    return 100.0 * (max(values) - min(values)) / abs(mean_value)  # 返回百分比形式的相对离散度。


def _execute_displacement(round_dir: Path, proposal_hash: str, operation: str) -> dict[str, Any]:  # 执行裂纹面位移提取、位移法K换算或材料联合请求。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 在执行前验证冻结提案内容完整。
    before_hash = sha256_text(canonical_json(proposal))  # 保存执行前提案摘要。
    text = _proposal_text(proposal)  # 组合提案自然语言用于参数提取。
    distances = _extract_distances(text)  # 提取模型请求的裂尖后方物理距离。
    include_k = operation == DISPLACEMENT_K_OPERATION  # 判断是否需要按位移渐近式换算K。
    backend = _load_backend()  # 加载只读有限元后端和真实解对象。
    rows = _displacement_rows(backend, distances, include_k)  # 提取真实裂纹面位移及可选K值。
    observations: dict[str, Any] = {"method": "从每档真实有限元解的上下裂纹面重复节点提取竖向位移差，并报告实际取样坐标与距离", "rows": rows}  # 组织公开位移方法和数值结果。
    limitations = ["目标物理距离若不落在网格节点上，结果使用最近的裂纹面内部节点并明确报告实际距离", "当前网格使用常规四节点四边形单元，未引入四分之一节点奇异单元"]  # 说明位移提取和网格表示边界。
    if include_k:  # 检查是否执行了位移法K换算。
        k_values = [float(row["stress_intensity_from_opening_mpa_sqrt_mm"]) for row in rows]  # 收集全部网格和距离的K估计值。
        observations["k_conversion"] = "平面应力Mode-I裂纹面位移渐近式：K = u_y_half·E/[(1+ν)(κ+1)]·sqrt(2π/r)，κ=(3-ν)/(1+ν)"  # 向模型公开实际换算公式。
        observations["relative_spread_percent"] = _relative_spread(k_values)  # 报告位移法K的网格离散度。
        limitations.append("位移渐近式只有在取样点位于K主导区且小范围屈服成立时具有工程意义")  # 说明位移法K的适用范围。
    requested_information: list[str] = []  # 初始化仍需用户补充的外部事实。
    status = "completed"  # 默认位移提取已经完整执行。
    if operation == DISPLACEMENT_MATERIAL_OPERATION:  # 检查提案是否同时请求材料适用性事实。
        requested_information = ["屈服强度", "真实应力-塑性应变曲线", "硬化模型", "断裂韧性或损伤参数"]  # 列出隐藏执行器无法自行生成的外部材料数据。
        observations["requested_information"] = requested_information  # 把真实外部阻塞和可计算位移证据同时反馈给模型。
        status = "information_required"  # 标记仍需用户提供外部事实但保留已完成数值证据。
    actual_parameters = {"requested_distances_mm": distances, "mesh_levels": [int(value) for value in backend.INITIAL_LEVELS], "plane_condition": "plane_stress"}  # 记录模型请求和实际分析条件。
    feedback = {"status": status, "executed_change": "不改变有限元模型，从三档已求解网格的上下裂纹面节点提取真实张开位移" + ("并按同一位移渐近式换算K" if include_k else ""), "actual_parameters": actual_parameters, "observations": observations, "limitations": limitations, "proposal_sha256": proposal_hash}  # 组织不含内部操作名称的公开反馈。
    after_hash = sha256_text(canonical_json(proposal))  # 重新计算执行后提案摘要。
    if after_hash != before_hash:  # 检查执行过程中是否改写冻结提案。
        raise RuntimeError("executor mutated the frozen proposal")  # 在内容改变时拒绝结果。
    return _write_execution(round_dir, proposal_hash, operation, observations, actual_parameters, feedback)  # 保存并返回真实位移证据。


def _prior_stress_records(round_dir: Path) -> list[dict[str, float]]:  # 从当前实验目录读取前轮固定距离应力公开证据。
    records: dict[tuple[int, float], dict[str, float]] = {}  # 初始化按网格和距离去重的记录映射。
    for sibling in sorted(round_dir.parent.glob("round_*")):  # 按轮次顺序遍历同一实验的冻结目录。
        if sibling.name >= round_dir.name:  # 跳过当前轮和后续轮次。
            continue  # 只使用模型已经看到的历史公开证据。
        feedback_path = sibling / "public_feedback.json"  # 定位前轮公开反馈文件。
        if not feedback_path.is_file():  # 检查前轮是否具有公开结果。
            continue  # 跳过没有公开反馈的轮次。
        feedback = json.loads(feedback_path.read_text(encoding="utf-8"))  # 读取前轮模型可见反馈。
        rows = feedback.get("observations", {}).get("rows", [])  # 读取可能存在的网格结果数组。
        if not isinstance(rows, list):  # 检查结果结构。
            continue  # 跳过非数组结果。
        for row in rows:  # 遍历当前前轮的全部网格结果。
            if not isinstance(row, dict) or "samples" not in row:  # 只处理固定距离应力采样结果。
                continue  # 跳过其他隐藏工具的结果行。
            nx = int(row.get("nx", 0))  # 读取网格划分数。
            samples = row.get("samples", {})  # 读取固定距离应力样本对象。
            if not isinstance(samples, dict):  # 检查样本结构。
                continue  # 跳过无效样本。
            for key, value in samples.items():  # 遍历距离字段和应力值。
                match = re.match(r"distance_(\d+(?:\.\d+)?)_mm_mean_sigma_y_mpa", str(key))  # 从字段名提取实际物理距离。
                if match is None:  # 检查字段是否符合固定距离应力合同。
                    continue  # 跳过其他样本字段。
                distance = float(match.group(1))  # 读取样本距离。
                records[(nx, distance)] = {"nx": float(nx), "h_local_mm": float(row.get("h_local_mm", 0.0)), "distance_from_tip_mm": distance, "mean_sigma_y_mpa": float(value)}  # 保存最新的公开应力记录。
    return [records[key] for key in sorted(records)]  # 返回按网格和距离排序的唯一记录。


def _execute_stress_postprocess(round_dir: Path, proposal_hash: str) -> dict[str, Any]:  # 仅基于前轮公开应力数据执行模型明确提出的K后处理。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 在后处理前验证冻结提案完整。
    before_hash = sha256_text(canonical_json(proposal))  # 保存后处理前提案摘要。
    text = _proposal_text(proposal)  # 组合提案文本用于距离选择。
    requested_distances = _extract_distances(text)  # 读取模型公式中使用的物理距离。
    requested_distance = float(requested_distances[0])  # 当前公式只使用一个明确固定距离。
    records = _prior_stress_records(round_dir)  # 读取模型已经看到的前轮固定距离应力证据。
    if not records:  # 检查是否存在可用于纯后处理的公开数据。
        raise RuntimeError("no prior fixed-distance stress evidence is available for the requested post-processing")  # 禁止偷偷运行新有限元求解替代已有数据。
    available_distances = sorted({float(row["distance_from_tip_mm"]) for row in records})  # 收集前轮实际可用距离。
    used_distance = min(available_distances, key=lambda value: abs(value - requested_distance))  # 选择最接近模型请求的既有公开距离。
    used_records = [row for row in records if abs(float(row["distance_from_tip_mm"]) - used_distance) <= 1.0e-12]  # 筛选同一距离的各网格应力值。
    result_rows: list[dict[str, float]] = []  # 初始化纯后处理结果数组。
    for row in used_records:  # 遍历各网格已有应力证据。
        sigma = float(row["mean_sigma_y_mpa"])  # 读取前轮公开法向应力。
        k_value = sigma * sqrt(2.0 * pi * used_distance)  # 严格按模型提出的裂尖前方渐近应力公式换算K。
        result_rows.append({"nx": float(row["nx"]), "h_local_mm": float(row["h_local_mm"]), "distance_from_tip_mm": used_distance, "mean_sigma_y_mpa": sigma, "stress_intensity_estimate_mpa_sqrt_mm": float(k_value)})  # 保存当前网格纯后处理结果。
    k_values = [float(row["stress_intensity_estimate_mpa_sqrt_mm"]) for row in result_rows]  # 收集K估计值用于收敛比较。
    observations = {"method": "仅使用前轮已经公开的固定距离法向应力，按提案公式K=σ_yy·sqrt(2πr)进行代数后处理；未运行新的有限元求解", "rows": result_rows, "relative_spread_percent": _relative_spread(k_values)}  # 组织可审计派生结果。
    actual_parameters = {"requested_distance_mm": requested_distance, "used_existing_distance_mm": used_distance, "source": "prior public_feedback only"}  # 记录请求距离和实际既有证据距离。
    limitations = ["固定距离应力换算K只有在取样点位于K主导区时有效", "该后处理不能替代路径无关积分、位移外推或解析标准算例校核"]  # 明确纯后处理的物理适用边界。
    feedback = {"status": "completed", "executed_change": "未运行新模型，仅对前轮公开的固定距离应力进行K换算和网格离散度比较", "actual_parameters": actual_parameters, "observations": observations, "limitations": limitations, "proposal_sha256": proposal_hash}  # 组织公开后处理反馈。
    after_hash = sha256_text(canonical_json(proposal))  # 重新计算后处理后提案摘要。
    if after_hash != before_hash:  # 检查后处理是否改写冻结提案。
        raise RuntimeError("executor mutated the frozen proposal")  # 在内容改变时拒绝结果。
    return _write_execution(round_dir, proposal_hash, STRESS_POSTPROCESS_OPERATION, observations, actual_parameters, feedback)  # 保存并返回纯后处理结果。


def map_frozen_proposal(round_dir: Path, proposal_hash: str) -> dict[str, Any]:  # 优先识别位移提取和纯后处理并收紧普通加密映射。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 在任何分类前验证冻结提案。
    before_hash = sha256_text(canonical_json(proposal))  # 保存分类前提案摘要。
    proposal_type = str(proposal.get("proposal_type", ""))  # 读取模型选择的任务级提案类型。
    text = _proposal_text(proposal)  # 组合全部自然语言字段用于忠实语义识别。
    asks_displacement = _contains_any(text, _DISPLACEMENT_TERMS)  # 检查是否明确要求裂纹面或节点位移。
    asks_fracture = _contains_any(text, _FRACTURE_TERMS)  # 检查是否要求位移法断裂评价量。
    asks_material = _contains_any(text, _MATERIAL_TERMS)  # 检查是否同时请求外部材料事实。
    if proposal_type in {"experiment", "request_information"} and asks_displacement:  # 在既有固定位置映射前优先保护位移场目的。
        after_hash = sha256_text(canonical_json(proposal))  # 重新计算分类后提案摘要。
        if after_hash != before_hash:  # 检查分类是否改写冻结提案。
            raise RuntimeError("executor mutated the frozen proposal")  # 在内容变化时拒绝执行。
        if asks_material:  # 检查是否属于可计算位移与外部材料事实联合请求。
            return _write_mapping(round_dir, proposal_hash, DISPLACEMENT_MATERIAL_OPERATION, "proposal requests computable crack-face displacement evidence together with external material facts")  # 保留两部分目的而不丢失任一项。
        if asks_fracture:  # 检查是否要求由裂纹面位移换算K。
            return _write_mapping(round_dir, proposal_hash, DISPLACEMENT_K_OPERATION, "proposal requests crack-face displacement extraction and a displacement-derived fracture quantity")  # 映射为位移提取与K换算联合操作。
        return _write_mapping(round_dir, proposal_hash, DISPLACEMENT_PROBE_OPERATION, "proposal requests crack-face or nodal displacement data from existing numerical solutions")  # 映射为真实裂纹面位移提取。
    if proposal_type == "experiment" and _contains_any(text, _STRESS_FORMULA_TERMS) and _contains_any(text, _NO_NEW_SOLVE_TERMS) and asks_fracture:  # 识别只基于已有应力数据的明确代数后处理实验。
        if _prior_stress_records(round_dir):  # 检查模型已经看到所需公开应力证据。
            return _write_mapping(round_dir, proposal_hash, STRESS_POSTPROCESS_OPERATION, "proposal performs a stated fracture-quantity formula on prior public stress evidence without a new solve")  # 映射为纯后处理而非网格加密。
        return _write_mapping(round_dir, proposal_hash, "unsupported", "proposal requires prior fixed-distance stress evidence that is not available in the public history")  # 在缺少输入时诚实拒绝而不新跑模型。
    mapping = previous_adapter.map_frozen_proposal(round_dir, proposal_hash)  # 对其他任务控制和真实实验复用v5栈。
    if proposal_type == "experiment" and mapping.get("operation") == "refine":  # 检查既有映射是否因为文本中的nx引用误判为加密。
        experiment = proposal.get("experiment", {})  # 读取冻结实验设计对象。
        change_text = str(experiment.get("change", "")).lower()  # 只读取模型声称要改变的变量。
        explicit_refinement = _contains_any(change_text, _REFINE_TERMS) or re.search(r"(?:from|从)[^。；,，]{0,40}\d+(?:\.\d+)?\s*(?:mm|毫米)[^。；,，]{0,30}(?:to|到|至|减小到|细化到)[^\d]{0,12}\d+(?:\.\d+)?\s*(?:mm|毫米)", change_text, flags=re.IGNORECASE) is not None  # 检查change字段是否真的改变网格分辨率。
        if not explicit_refinement:  # 检查当前提案只是引用网格编号或执行代数后处理。
            return _write_mapping(round_dir, proposal_hash, "unsupported", "proposal does not request a mesh-resolution change; references to existing nx values cannot be executed as refinement")  # 阻止把已有数据后处理偷换成重新求解。
    return mapping  # 返回最终忠实隐藏映射。


def execute_mapping(round_dir: Path, proposal_hash: str, mapping: dict[str, Any]) -> dict[str, Any]:  # 执行位移、纯后处理或委托任务级隐藏执行器。
    operation = str(mapping.get("operation", ""))  # 读取模型不可见的内部操作标识。
    if operation in {DISPLACEMENT_PROBE_OPERATION, DISPLACEMENT_K_OPERATION, DISPLACEMENT_MATERIAL_OPERATION}:  # 检查是否需要执行真实裂纹面位移路径。
        return _execute_displacement(round_dir, proposal_hash, operation)  # 返回忠实位移证据和可选材料阻塞。
    if operation == STRESS_POSTPROCESS_OPERATION:  # 检查是否需要执行纯代数后处理。
        return _execute_stress_postprocess(round_dir, proposal_hash)  # 使用前轮公开数据计算K而不重新求解。
    return previous_adapter.execute_mapping(round_dir, proposal_hash, mapping)  # 对其他路线控制和隐藏实验复用v5执行层。

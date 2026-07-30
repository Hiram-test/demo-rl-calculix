from __future__ import annotations  # 启用现代类型注解并保持运行环境兼容。

import json  # 更新隐藏映射收据和脱敏反馈文件。
import re  # 从中英文自然语言提案中提取数值参数。
from pathlib import Path  # 管理冻结轮次目录。
from typing import Any  # 表示动态提案、映射和反馈结构。

from .contracts import canonical_json  # 规范序列化冻结提案用于完整性检查。
from .contracts import sha256_text  # 计算冻结提案摘要以验证执行器没有改写内容。
from .contracts import verify_frozen_proposal  # 读取并验证冻结提案内容。
from .executor import _load_backend  # 复用既有只读有限元后端加载器。
from .executor import execute_mapping as _base_execute_mapping  # 复用已经实现的真实执行逻辑。
from .executor import map_frozen_proposal as _base_map_frozen_proposal  # 复用基础确定性映射逻辑。

FRACTURE_SEQUENCE_OPERATION = "fracture_parameter_sequence"  # 定义只存在于隐藏审计层的断裂参数序列操作标识。


def _write_mapping(round_dir: Path, proposal_hash: str, operation: str, reason: str) -> dict[str, Any]:  # 统一保存隐藏映射收据。
    mapping = {"proposal_sha256": proposal_hash, "operation": operation, "mapping_reason": reason, "proposal_unchanged": True}  # 组织完整内部映射记录。
    (round_dir / "mapping_receipt.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 把映射收据写入模型不可见文件。
    return mapping  # 返回后续执行阶段使用的映射对象。


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:  # 对中英文关键词执行确定性包含检查。
    return any(term in text for term in terms)  # 只要任一物理语义短语出现就返回真。


def map_frozen_proposal(round_dir: Path, proposal_hash: str) -> dict[str, Any]:  # 对中英文自然语言和派生计算请求增加保守适配层。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 在任何映射前重新验证冻结提案。
    before_hash = sha256_text(canonical_json(proposal))  # 保存映射前提案摘要。
    text = json.dumps(proposal, ensure_ascii=False).lower()  # 组合提案全部自然语言字段并统一小写。
    proposal_type = str(proposal.get("proposal_type", ""))  # 读取模型自己选择的提案类型。
    asks_fracture_quantity = _contains_any(text, ("应力强度因子", "j积分", "j 积分", "能量释放率", "裂纹驱动力", "stress intensity factor", "k_i", "ki ", "j-integral", "j integral", "energy release rate", "fracture driving force"))  # 检查模型是否明确请求断裂参量。
    asks_existing_computation = _contains_any(text, ("已完成", "已有", "现有求解", "分别报告", "计算方法", "current model", "existing result", "completed mesh", "provide the value", "method used", "directly output"))  # 检查请求是否针对现有数值模型的派生计算。
    if proposal_type == "request_information" and asks_fracture_quantity and asks_existing_computation:  # 只在请求明确可由现有求解派生时拦截外部信息映射。
        after_hash = sha256_text(canonical_json(proposal))  # 再次计算提案摘要。
        if after_hash != before_hash:  # 检查分类过程中是否改写了冻结提案。
            raise RuntimeError("executor mutated the frozen proposal")  # 发现任何改写时拒绝继续。
        return _write_mapping(round_dir, proposal_hash, FRACTURE_SEQUENCE_OPERATION, "proposal requests fracture quantities derived from the existing numerical model")  # 映射为隐藏断裂参量序列计算。
    if proposal_type == "experiment":  # 只对模型明确提出的控制实验执行物理语义映射。
        experiment = proposal.get("experiment", {})  # 读取冻结实验设计对象。
        change_text = str(experiment.get("change", "")).lower()  # 单独读取模型准备改变的变量。
        measure_text = " ".join(str(item) for item in experiment.get("measure", [])).lower()  # 单独读取模型准备测量的输出。
        has_crack_subject = _contains_any(change_text + " " + text, ("裂纹", "crack"))  # 检查实验对象是否明确是裂纹。
        has_extension_change = _contains_any(change_text, ("延长", "扩展", "增长", "增量", "变长", "increase the half-crack", "increase half-crack", "extend", "extension", "crack length", "virtual crack"))  # 检查是否明确改变裂纹长度。
        has_energy_measure = _contains_any(measure_text + " " + text, ("能量", "势能", "应变能", "释放", "energy", "strain energy", "energy release"))  # 检查是否明确比较能量量。
        if has_crack_subject and has_extension_change and has_energy_measure:  # 只在裂纹长度改变和能量测量同时出现时映射。
            return _write_mapping(round_dir, proposal_hash, "geometry_energy", "proposal changes crack length and compares an energy quantity in engineering language")  # 生成忠实中英文几何扰动映射。
        if _contains_any(text, ("固定物理位置", "固定距离", "距裂尖", "测点", "取样位置", "路径点", "fixed physical location", "fixed physical distance", "ahead of the crack tip", "sample stress at")):  # 识别中英文固定物理位置采样提案。
            return _write_mapping(round_dir, proposal_hash, "fixed_probe", "proposal compares field values at fixed physical locations")  # 映射到固定位置场量提取。
        if _contains_any(text, ("区域平均", "范围平均", "面积平均", "固定半径", "局部区域", "region average", "area average", "fixed radius", "average over a region")):  # 识别中英文固定物理区域聚合提案。
            return _write_mapping(round_dir, proposal_hash, "region_average", "proposal compares an aggregate over a fixed physical region")  # 映射到区域平均提取。
        if _contains_any(text, ("解析解", "理论解", "闭式解", "理论参照", "理论校核", "analytical solution", "closed-form", "closed form", "theoretical reference", "analytical reference")):  # 识别中英文理论参照提案。
            return _write_mapping(round_dir, proposal_hash, "closed_form", "proposal requests an analytical reference under stated assumptions")  # 映射到受限理论参照计算。
        if _contains_any(change_text, ("加密网格", "细化网格", "减小单元", "增加网格", "refine the mesh", "mesh refinement", "reduce the element size", "increase mesh resolution")) or re.search(r"nx\s*[=:]\s*\d+", change_text):  # 识别仅改变网格分辨率的中英文提案。
            return _write_mapping(round_dir, proposal_hash, "refine", "proposal changes only mesh resolution while holding the model fixed")  # 映射到新增真实网格求解。
    mapping = _base_map_frozen_proposal(round_dir, proposal_hash)  # 对未被多语言适配层识别的提案运行基础映射。
    return mapping  # 返回最终隐藏映射结果。


def _extract_geometry_request(text: str, default_nx: int, width: float) -> tuple[int, float]:  # 从中英文裂纹微增实验提案提取网格和请求增量。
    nx_match = re.search(r"nx\s*[=:：]?\s*(\d+)", text, flags=re.IGNORECASE)  # 搜索显式网格划分数。
    nx = int(nx_match.group(1)) if nx_match is not None else int(default_nx)  # 使用提案值或当前最细默认网格。
    from_to_match = re.search(r"(?:from|从)\s*(\d+(?:\.\d+)?)\s*(?:mm|毫米)?\s*(?:to|到|至)\s*(\d+(?:\.\d+)?)\s*(?:mm|毫米)", text, flags=re.IGNORECASE)  # 搜索从一个裂纹长度增加到另一个长度的表达。
    if from_to_match is not None:  # 在模型给出起止长度时计算差值。
        requested = abs(float(from_to_match.group(2)) - float(from_to_match.group(1)))  # 计算模型实际请求的裂纹增量。
    else:  # 在没有起止长度时搜索显式增量。
        patterns = (r"(?:extension|extend(?:ed)?(?: by)?|increase(?:d)?(?: by)?|增量|延长|扩展)[^\d]{0,20}(\d+(?:\.\d+)?)\s*(?:mm|毫米)", r"(\d+(?:\.\d+)?)\s*(?:mm|毫米)[^。；,，.]{0,20}(?:extension|increase|延长|扩展|增量)")  # 定义中英文增量表达式。
        requested = width / nx  # 默认使用一个当前网格步长。
        for pattern in patterns:  # 依次检查常见表达方式。
            match = re.search(pattern, text, flags=re.IGNORECASE)  # 搜索当前增量模式。
            if match is not None:  # 在找到明确数值时停止搜索。
                requested = float(match.group(1))  # 保存模型请求的连续裂纹增量。
                break  # 避免后续模式覆盖第一个明确表达。
    return nx, requested  # 返回网格划分和模型原始请求增量。


def _execute_geometry_energy(round_dir: Path, proposal_hash: str) -> dict[str, Any]:  # 执行中英文提案描述的裂纹微增能量比较。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 在执行前验证冻结提案完整性。
    text = json.dumps(proposal, ensure_ascii=False).lower()  # 组合提案自然语言用于参数提取。
    backend = _load_backend()  # 加载只读有限元后端和真实求解缓存。
    nx, requested_extension = _extract_geometry_request(text, max(backend.INITIAL_LEVELS), backend.WIDTH)  # 提取模型请求的网格和裂纹增量。
    nx = min(160, max(20, int(round(nx / 20.0) * 20)))  # 把网格级别修复为后端允许的二十整数倍。
    grid_step = backend.WIDTH / nx  # 计算当前结构网格节点间距。
    step_count = max(1, int(round(requested_extension / grid_step)))  # 把连续增量吸附为至少一个完整网格步长。
    used_extension = min(5.0, step_count * grid_step)  # 保证实际裂纹扰动不超过后端安全上限。
    if used_extension < grid_step:  # 处理网格步长大于上限的极端情况。
        used_extension = grid_step  # 至少保持一个完整节点步长。
    base = backend._solve(nx, backend.HALF_CRACK)  # 读取或求解原裂纹模型。
    extended = backend._solve(nx, backend.HALF_CRACK + used_extension)  # 求解节点对齐的延长裂纹模型。
    energy_change = float(extended["strain_energy_n_mm"] - base["strain_energy_n_mm"])  # 计算总应变能差。
    added_surface = 2.0 * used_extension * backend.THICKNESS  # 计算两个对称裂尖新增裂纹表面积。
    energy_release = energy_change / added_surface  # 用有限裂纹增量近似能量释放率。
    stress_intensity = (max(energy_release, 0.0) * backend.YOUNG) ** 0.5  # 依据平面应力线弹性关系换算 K。
    observations = {"base_strain_energy_n_mm": float(base["strain_energy_n_mm"]), "extended_strain_energy_n_mm": float(extended["strain_energy_n_mm"]), "energy_change_n_mm": energy_change, "added_crack_surface_mm2": added_surface, "energy_release_rate_n_per_mm": float(energy_release), "stress_intensity_mpa_sqrt_mm": float(stress_intensity)}  # 组织公开数值观测。
    parameters = {"nx": nx, "requested_extension_mm": requested_extension, "used_extension_mm": used_extension, "grid_step_mm": grid_step, "parameter_repair": "snapped_to_integer_mesh_steps"}  # 记录模型请求和实际执行参数。
    limitations = ["裂纹增量必须与结构网格节点对齐，因此连续请求值可能被吸附到完整网格步长", "该方法使用有限裂纹增量能量差，并非轮廓积分", "换算采用二维平面应力线弹性关系，尚未包含材料塑性"]  # 明确方法和参数修复边界。
    feedback = {"status": "completed", "executed_change": f"保持外形、材料、载荷和边界不变，把两侧裂尖各延长 {used_extension:g} mm 并比较总应变能", "actual_parameters": parameters, "observations": observations, "limitations": limitations, "proposal_sha256": proposal_hash}  # 组织不泄露内部操作名称的公开反馈。
    audit = {"proposal_sha256": proposal_hash, "internal_operation": "geometry_energy", "raw_result": observations, "actual_parameters": parameters, "public_feedback": feedback}  # 组织模型不可见执行审计。
    (round_dir / "execution_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存内部操作和原始结果。
    (round_dir / "public_feedback.json").write_text(json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存下一轮唯一可见的物理反馈。
    return feedback  # 返回真实数值证据。


def _execute_fracture_parameter_sequence(round_dir: Path, proposal_hash: str) -> dict[str, Any]:  # 从现有三档网格计算线弹性断裂参量序列。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 在执行前验证冻结提案仍未改变。
    proposal_hash_before = sha256_text(canonical_json(proposal))  # 保存执行前提案摘要。
    backend = _load_backend()  # 加载只读有限元后端和真实求解缓存。
    rows: list[dict[str, Any]] = []  # 初始化各网格断裂参量结果列表。
    for nx in backend.INITIAL_LEVELS:  # 遍历模型请求比较的三档既有网格。
        extension = backend.WIDTH / nx  # 使用当前网格一个完整节点步长作为裂纹微增量。
        base = backend._solve(nx, backend.HALF_CRACK)  # 读取或求解原裂纹模型。
        extended = backend._solve(nx, backend.HALF_CRACK + extension)  # 求解与当前结构网格节点对齐的延长裂纹模型。
        energy_change = float(extended["strain_energy_n_mm"] - base["strain_energy_n_mm"])  # 计算裂纹微增前后的总应变能差。
        added_surface = 2.0 * extension * backend.THICKNESS  # 计算两个对称裂尖新增裂纹表面积。
        energy_release = energy_change / added_surface  # 用有限裂纹增量近似平面应力能量释放率。
        stress_intensity = (max(energy_release, 0.0) * backend.YOUNG) ** 0.5  # 依据平面应力关系 K=sqrt(EG) 换算应力强度因子。
        rows.append({"nx": int(nx), "h_local_mm": float(backend.WIDTH / nx), "crack_extension_mm": float(extension), "base_strain_energy_n_mm": float(base["strain_energy_n_mm"]), "extended_strain_energy_n_mm": float(extended["strain_energy_n_mm"]), "energy_release_rate_n_per_mm": float(energy_release), "stress_intensity_mpa_sqrt_mm": float(stress_intensity)})  # 保存当前网格的可审计计算量。
    finest_value = float(rows[-1]["stress_intensity_mpa_sqrt_mm"])  # 读取最密网格应力强度因子作为请求中的比较基准。
    for row in rows:  # 为每档网格计算相对最密网格的变化百分比。
        current_value = float(row["stress_intensity_mpa_sqrt_mm"])  # 读取当前网格应力强度因子。
        row["difference_from_finest_percent"] = 100.0 * (current_value - finest_value) / finest_value if finest_value else 0.0  # 保存有符号相对差异。
    proposal_hash_after = sha256_text(canonical_json(proposal))  # 重新计算执行后提案摘要。
    if proposal_hash_after != proposal_hash_before:  # 检查数值执行过程中是否改写提案。
        raise RuntimeError("executor mutated the frozen proposal")  # 在提案发生变化时拒绝结果。
    observations = {"method": "对每档网格将两侧裂尖各延长本档一个网格步长，用总应变能差除以新增裂纹表面积得到 G，并在二维平面应力线弹性假设下用 K=sqrt(EG) 换算", "rows": rows}  # 组织模型可见的物理方法和数值结果。
    limitations = ["该结果使用有限裂纹增量能量差，并非轮廓积分或裂纹面位移外推", "每档裂纹增量等于该档结构网格步长，因此不同网格的差分步长不同", "换算采用二维平面应力线弹性关系，尚未包含裂尖塑性区或真实材料曲线"]  # 明确方法边界和未解决问题。
    feedback = {"status": "completed", "executed_change": "保持外形、材料弹性参数、远场载荷和边界不变，对三档网格分别计算裂纹微增前后的能量差并换算 G 与 K", "actual_parameters": {"mesh_levels": [int(value) for value in backend.INITIAL_LEVELS], "extension_rule": "one grid step at each mesh level", "plane_condition": "plane_stress", "young_mpa": float(backend.YOUNG)}, "observations": observations, "limitations": limitations, "proposal_sha256": proposal_hash}  # 组织不含内部操作名称的公开反馈。
    audit = {"proposal_sha256": proposal_hash, "internal_operation": FRACTURE_SEQUENCE_OPERATION, "raw_result": observations, "actual_parameters": feedback["actual_parameters"], "public_feedback": feedback}  # 组织模型不可见的完整执行审计。
    (round_dir / "execution_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存内部操作和原始数值结果。
    (round_dir / "public_feedback.json").write_text(json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存下一轮唯一允许回传的物理反馈。
    return feedback  # 返回脱敏数值证据供下一轮模型决策。


def execute_mapping(round_dir: Path, proposal_hash: str, mapping: dict[str, Any]) -> dict[str, Any]:  # 对多语言几何实验、派生断裂量和公开结束状态进行适配处理。
    operation = mapping.get("operation")  # 读取模型不可见的内部操作标识。
    if operation == FRACTURE_SEQUENCE_OPERATION:  # 检查是否需要执行三档网格断裂参量计算。
        return _execute_fracture_parameter_sequence(round_dir, proposal_hash)  # 执行忠实派生计算并返回公开反馈。
    if operation == "geometry_energy":  # 检查是否需要执行中英文裂纹微增能量比较。
        return _execute_geometry_energy(round_dir, proposal_hash)  # 使用多语言参数提取和透明网格吸附执行真实求解。
    feedback = _base_execute_mapping(round_dir, proposal_hash, mapping)  # 对其他操作执行基础真实求解或停止逻辑。
    if operation != "finish":  # 在非结束操作时保持原始物理反馈。
        return feedback  # 返回基础脱敏反馈。
    feedback = dict(feedback)  # 复制反馈以避免修改调用方持有的对象。
    feedback["status"] = "analysis_complete"  # 使用不含内部操作词的中性公开状态。
    feedback["observations"] = {"analysis_complete": True}  # 使用中性布尔观测替代内部状态字符串。
    (round_dir / "public_feedback.json").write_text(json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 重写下一轮模型可见反馈。
    audit_path = round_dir / "execution_audit.json"  # 定位完整内部审计文件。
    audit = json.loads(audit_path.read_text(encoding="utf-8"))  # 读取基础执行审计。
    audit["public_feedback"] = feedback  # 让审计文件引用最终中性公开反馈。
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存更新后的完整审计记录。
    return feedback  # 返回中性化的公开结束反馈。

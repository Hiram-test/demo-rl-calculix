from __future__ import annotations  # 启用现代类型注解并保持运行环境兼容。

import json  # 更新隐藏映射收据和脱敏反馈文件。
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


def map_frozen_proposal(round_dir: Path, proposal_hash: str) -> dict[str, Any]:  # 对自然语言语序和派生计算请求增加保守兼容层。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 在任何映射前重新验证冻结提案。
    before_hash = sha256_text(canonical_json(proposal))  # 保存映射前提案摘要。
    text = json.dumps(proposal, ensure_ascii=False).lower()  # 组合提案全部自然语言字段。
    proposal_type = str(proposal.get("proposal_type", ""))  # 读取模型自己选择的提案类型。
    asks_fracture_quantity = any(token in text for token in ("应力强度因子", "j积分", "j 积分", "能量释放率", "裂纹驱动力"))  # 检查模型是否明确请求断裂参量。
    asks_mesh_comparison = any(token in text for token in ("三个网格", "三档网格", "各网格", "网格变化", "收敛", "变化百分比"))  # 检查模型是否要求跨网格比较。
    asks_existing_results = any(token in text for token in ("已完成", "已有", "现有求解", "分别报告", "计算方法"))  # 检查请求是否针对现有数值模型的派生计算。
    if proposal_type == "request_information" and asks_fracture_quantity and asks_mesh_comparison and asks_existing_results:  # 只在请求明确可由现有求解派生时拦截外部信息映射。
        after_hash = sha256_text(canonical_json(proposal))  # 再次计算提案摘要。
        if after_hash != before_hash:  # 检查分类过程中是否改写了冻结提案。
            raise RuntimeError("executor mutated the frozen proposal")  # 发现任何改写时拒绝继续。
        return _write_mapping(round_dir, proposal_hash, FRACTURE_SEQUENCE_OPERATION, "proposal requests fracture quantities derived across the already solved mesh sequence")  # 映射为隐藏断裂参量序列计算。
    mapping = _base_map_frozen_proposal(round_dir, proposal_hash)  # 对其他提案运行基础隐藏映射。
    if mapping.get("operation") != "unsupported":  # 在基础映射已经明确时保持原结果。
        return mapping  # 返回基础忠实映射。
    has_crack_subject = "裂纹" in text  # 检查实验对象是否明确是裂纹。
    has_extension_change = any(token in text for token in ("延长", "扩展", "增长", "增量", "变长"))  # 检查是否明确改变裂纹长度。
    has_energy_measure = any(token in text for token in ("能量", "势能", "应变能", "释放"))  # 检查是否明确比较能量量。
    if has_crack_subject and has_extension_change and has_energy_measure:  # 只在三个物理条件同时满足时补充语序无关映射。
        return _write_mapping(round_dir, proposal_hash, "geometry_energy", "proposal changes crack length and compares an energy quantity regardless of word order")  # 生成忠实内部映射收据。
    return mapping  # 返回最终隐藏映射结果。


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


def execute_mapping(round_dir: Path, proposal_hash: str, mapping: dict[str, Any]) -> dict[str, Any]:  # 对派生断裂量和公开结束状态进行适配处理。
    operation = mapping.get("operation")  # 读取模型不可见的内部操作标识。
    if operation == FRACTURE_SEQUENCE_OPERATION:  # 检查是否需要执行三档网格断裂参量计算。
        return _execute_fracture_parameter_sequence(round_dir, proposal_hash)  # 执行忠实派生计算并返回公开反馈。
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

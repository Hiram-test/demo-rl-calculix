from __future__ import annotations  # 启用现代类型注解并保持运行环境兼容。

import json  # 保存组合实验的隐藏审计和公开反馈。
import re  # 从中英文冻结提案中提取目标网格尺寸。
from pathlib import Path  # 管理冻结轮次目录。
from typing import Any  # 表示动态提案、映射和反馈结构。

from . import executor_adapter as previous_adapter  # 复用已验证的多语言隐藏执行器适配层。
from .contracts import canonical_json  # 规范序列化提案用于完整性检查。
from .contracts import sha256_text  # 计算提案摘要以验证执行器没有改写内容。
from .contracts import verify_frozen_proposal  # 读取并验证冻结提案。
from .executor import _load_backend  # 复用只读有限元后端和真实求解缓存。

REFINE_FRACTURE_OPERATION = "refine_and_fracture_parameter"  # 定义只存在于模型不可见审计层的组合操作。


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:  # 对中英文自然语言执行确定性包含检查。
    return any(term in text for term in terms)  # 任一物理语义短语出现时返回真。


def map_frozen_proposal(round_dir: Path, proposal_hash: str) -> dict[str, Any]:  # 在既有适配层之后识别网格细化与断裂量联合实验。
    mapping = previous_adapter.map_frozen_proposal(round_dir, proposal_hash)  # 首先调用已验证的多语言映射器。
    if mapping.get("operation") != "unsupported":  # 在既有映射已经能忠实执行时保持原结果。
        return mapping  # 返回既有隐藏映射。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 重新读取并验证冻结提案。
    before_hash = sha256_text(canonical_json(proposal))  # 保存映射前提案摘要。
    if proposal.get("proposal_type") != "experiment":  # 只处理模型明确提出的控制实验。
        return mapping  # 对其他提案保持诚实不支持结果。
    experiment = proposal.get("experiment", {})  # 读取实验设计对象。
    change_text = str(experiment.get("change", "")).lower()  # 读取准备改变的变量。
    measure_text = " ".join(str(item) for item in experiment.get("measure", [])).lower()  # 读取准备比较的输出。
    requests_refinement = (_contains_any(change_text, ("加密", "细化", "减小单元", "refine", "refinement", "reduce the element size", "smaller element")) and _contains_any(change_text, ("网格", "单元", "mesh", "element")))  # 检查实验是否明确改变网格分辨率。
    requests_fracture_quantity = _contains_any(measure_text, ("应力强度因子", "能量释放率", "j积分", "j 积分", "stress intensity factor", "k_i", "ki", "energy release rate", "j-integral", "j integral"))  # 检查测量项是否明确包含断裂参量。
    requests_energy_method = _contains_any(measure_text, ("总应变能", "能量差", "新增裂纹", "strain energy", "energy difference", "newly created crack", "k =", "sqrt(e", "√(e"))  # 检查模型是否指定当前后端可以实现的能量差方法。
    if not (requests_refinement and requests_fracture_quantity and requests_energy_method):  # 检查三个物理条件是否全部满足。
        return mapping  # 在无法保持测量目的时继续返回不支持。
    after_hash = sha256_text(canonical_json(proposal))  # 再次计算冻结提案摘要。
    if after_hash != before_hash:  # 检查分类过程中是否改写提案。
        raise RuntimeError("executor mutated the frozen proposal")  # 发现改写时立即拒绝执行。
    mapped = {"proposal_sha256": proposal_hash, "operation": REFINE_FRACTURE_OPERATION, "mapping_reason": "proposal refines the mesh and recomputes an energy-derived fracture parameter", "proposal_unchanged": True}  # 组织模型不可见的组合映射收据。
    (round_dir / "mapping_receipt.json").write_text(json.dumps(mapped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存覆盖后的忠实映射记录。
    return mapped  # 返回组合实验内部操作。


def _extract_target_size(text: str, default_size: float) -> float:  # 从冻结提案中提取目标裂尖单元尺寸。
    patterns = (r"(?:to|至|到|为)\s*(\d+(?:\.\d+)?)\s*(?:mm|毫米)", r"(?:element size|local mesh size|局部单元尺寸|裂尖尺寸)[^\d]{0,24}(\d+(?:\.\d+)?)\s*(?:mm|毫米)")  # 定义中英文目标尺寸表达式。
    for pattern in patterns:  # 依次检查候选表达式。
        matches = re.findall(pattern, text, flags=re.IGNORECASE)  # 提取当前模式下的全部数值。
        if matches:  # 在找到明确目标尺寸时停止搜索。
            value = matches[-1] if isinstance(matches[-1], str) else matches[-1][0]  # 读取最后一个目标值以处理从粗到细表达。
            return float(value)  # 返回模型提出的目标毫米尺寸。
    return float(default_size)  # 在未明确数值时使用受控默认细化尺寸。


def _fracture_value(backend: Any, nx: int) -> dict[str, float]:  # 在指定网格上以节点对齐裂纹微增计算 G 与 K。
    extension = backend.WIDTH / nx  # 使用当前网格一个完整节点步长作为裂纹增量。
    base = backend._solve(nx, backend.HALF_CRACK)  # 读取或求解原裂纹模型。
    extended = backend._solve(nx, backend.HALF_CRACK + extension)  # 求解与结构节点对齐的延长裂纹模型。
    energy_change = float(extended["strain_energy_n_mm"] - base["strain_energy_n_mm"])  # 计算裂纹微增前后的总应变能差。
    added_surface = 2.0 * extension * backend.THICKNESS  # 计算两个对称裂尖新增表面积。
    energy_release = energy_change / added_surface  # 用有限裂纹增量近似能量释放率。
    stress_intensity = (max(energy_release, 0.0) * backend.YOUNG) ** 0.5  # 按平面应力线弹性关系换算应力强度因子。
    return {"nx": float(nx), "h_local_mm": float(backend.WIDTH / nx), "crack_extension_mm": float(extension), "energy_release_rate_n_per_mm": float(energy_release), "stress_intensity_mpa_sqrt_mm": float(stress_intensity), "base_strain_energy_n_mm": float(base["strain_energy_n_mm"]), "extended_strain_energy_n_mm": float(extended["strain_energy_n_mm"])}  # 返回完整可审计数值。


def _execute_refine_fracture(round_dir: Path, proposal_hash: str) -> dict[str, Any]:  # 执行模型要求的网格细化与断裂参量复算联合实验。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 在执行前验证冻结提案完整性。
    text = json.dumps(proposal, ensure_ascii=False).lower()  # 组合提案文本用于目标尺寸提取。
    backend = _load_backend()  # 加载只读有限元后端和求解缓存。
    target_size = _extract_target_size(text, backend.WIDTH / 160.0)  # 读取模型提出的目标裂尖尺寸。
    target_nx = int(round((backend.WIDTH / target_size) / 20.0) * 20) if target_size > 0.0 else 160  # 把目标尺寸转换为后端允许的二十整数倍网格。
    target_nx = min(160, max(20, target_nx))  # 把网格参数限制在已验证预算范围。
    baseline_nx = int(max(backend.INITIAL_LEVELS))  # 使用初始最密网格作为模型要求的前后比较基准。
    baseline = _fracture_value(backend, baseline_nx)  # 计算基准网格的 G 与 K。
    refined = _fracture_value(backend, target_nx)  # 计算细化网格的 G 与 K。
    baseline_k = float(baseline["stress_intensity_mpa_sqrt_mm"])  # 读取基准应力强度因子。
    refined_k = float(refined["stress_intensity_mpa_sqrt_mm"])  # 读取细化应力强度因子。
    relative_change = 100.0 * (refined_k - baseline_k) / baseline_k if baseline_k else 0.0  # 计算有符号相对变化百分比。
    observations = {"method": "两档网格均将两侧裂尖各延长本档一个网格步长，以总应变能差计算 G，再用二维平面应力关系 K=sqrt(EG) 换算", "baseline": baseline, "refined": refined, "relative_change_percent": float(relative_change)}  # 组织公开物理方法和数值结果。
    actual_parameters = {"requested_target_h_mm": float(target_size), "used_target_nx": int(target_nx), "used_target_h_mm": float(backend.WIDTH / target_nx), "baseline_nx": baseline_nx, "extension_rule": "one grid step at each compared mesh"}  # 记录模型请求和实际网格参数。
    limitations = ["裂纹增量随各档网格步长变化，比较中仍含有限差分步长效应", "该计算使用能量差换算，并非奇异单元应力外推或轮廓J积分", "模型保持二维平面应力线弹性，未检验真实材料塑性"]  # 说明组合实验的适用边界。
    feedback = {"status": "completed", "executed_change": f"保持几何、材料、载荷和边界不变，把裂尖网格从 {baseline['h_local_mm']:g} mm 细化到 {refined['h_local_mm']:g} mm，并在两档网格上按相同能量差协议复算 G 与 K", "actual_parameters": actual_parameters, "observations": observations, "limitations": limitations, "proposal_sha256": proposal_hash}  # 组织不含内部操作名称的公开反馈。
    audit = {"proposal_sha256": proposal_hash, "internal_operation": REFINE_FRACTURE_OPERATION, "raw_result": observations, "actual_parameters": actual_parameters, "public_feedback": feedback}  # 组织模型不可见的完整执行审计。
    (round_dir / "execution_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存组合实验内部审计。
    (round_dir / "public_feedback.json").write_text(json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存下一轮唯一可见的物理反馈。
    return feedback  # 返回真实组合实验结果。


def execute_mapping(round_dir: Path, proposal_hash: str, mapping: dict[str, Any]) -> dict[str, Any]:  # 执行组合实验或委托既有适配层。
    if mapping.get("operation") == REFINE_FRACTURE_OPERATION:  # 检查是否需要执行网格细化与断裂参量联合实验。
        return _execute_refine_fracture(round_dir, proposal_hash)  # 执行联合实验并返回公开反馈。
    return previous_adapter.execute_mapping(round_dir, proposal_hash, mapping)  # 对其他提案复用已验证的多语言执行器。

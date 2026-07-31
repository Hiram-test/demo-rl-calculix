from __future__ import annotations  # 启用现代类型注解并保持运行环境兼容。

import json  # 保存通用目标尺寸执行审计和净化后的公开反馈。
import re  # 从中英文冻结提案中识别显式目标网格尺寸。
from pathlib import Path  # 管理冻结轮次目录。
from typing import Any  # 表示动态提案、映射和反馈结构。

from . import executor_adapter_v2 as previous_adapter  # 复用已经支持多语言和断裂联合实验的适配层。
from .contracts import canonical_json  # 规范序列化提案用于完整性检查。
from .contracts import sha256_text  # 计算提案摘要以验证执行器没有改写内容。
from .contracts import verify_frozen_proposal  # 读取并验证冻结提案。
from .executor import _load_backend  # 复用只读有限元后端和真实求解缓存。

REFINE_TARGET_OPERATION = "refine_to_explicit_target_size"  # 定义只存在于模型不可见审计层的显式尺寸加密操作。
INTERNAL_PUBLIC_KEYS = {"tool", "operation", "internal_operation", "mapping_reason"}  # 定义不得出现在下一轮模型证据中的内部字段名。


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:  # 对中英文自然语言执行确定性包含检查。
    return any(term in text for term in terms)  # 任一语义短语出现时返回真。


def _extract_target_size(text: str) -> float | None:  # 从冻结提案中提取明确的目标网格尺寸。
    patterns = (r"(?:from|从)[^。；,，]{0,40}(\d+(?:\.\d+)?)\s*(?:mm|毫米)[^。；,，]{0,30}(?:to|至|到|减小到|细化到)[^\d]{0,12}(\d+(?:\.\d+)?)\s*(?:mm|毫米)", r"(?:target size|target element size|local element size|目标尺寸|目标单元尺寸|裂尖单元尺寸)[^\d]{0,30}(\d+(?:\.\d+)?)\s*(?:mm|毫米)", r"(?:to|至|到|减小到|细化到)\s*(\d+(?:\.\d+)?)\s*(?:mm|毫米)")  # 定义从粗到细和直接目标值表达式。
    for pattern in patterns:  # 依次搜索候选表达式。
        match = re.search(pattern, text, flags=re.IGNORECASE)  # 搜索当前中英文模式。
        if match is None:  # 在当前模式没有匹配时继续。
            continue  # 检查下一种表达方式。
        groups = [value for value in match.groups() if value is not None]  # 收集当前模式捕获的全部数值。
        if not groups:  # 防御性检查模式是否真的捕获数值。
            continue  # 在没有数值时继续搜索。
        return float(groups[-1])  # 对从粗到细表达使用最后一个数值作为目标尺寸。
    return None  # 在模型没有明确尺寸时返回空值并交给既有映射器。


def map_frozen_proposal(round_dir: Path, proposal_hash: str) -> dict[str, Any]:  # 在既有映射前优先保护模型明确给出的目标网格尺寸。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 在任何分类前验证冻结提案。
    before_hash = sha256_text(canonical_json(proposal))  # 保存分类前提案摘要。
    if proposal.get("proposal_type") == "experiment":  # 只对模型明确提出的控制实验检查目标尺寸。
        experiment = proposal.get("experiment", {})  # 读取冻结实验设计对象。
        change_text = str(experiment.get("change", "")).lower()  # 读取模型准备改变的变量。
        measure_text = " ".join(str(item) for item in experiment.get("measure", [])).lower()  # 读取模型要求比较的输出。
        requests_refinement = _contains_any(change_text, ("加密", "细化", "减小", "refine", "refinement", "reduce", "smaller")) and _contains_any(change_text, ("网格", "单元", "mesh", "element"))  # 检查实验是否明确改变网格分辨率。
        target_size = _extract_target_size(change_text)  # 提取模型给出的目标毫米尺寸。
        requests_fracture_recompute = _contains_any(measure_text, ("应力强度因子", "能量释放率", "stress intensity factor", "energy release rate", "k =", "sqrt(e", "√(e"))  # 检查是否属于v2已经处理的断裂联合实验。
        if requests_refinement and target_size is not None and not requests_fracture_recompute:  # 只拦截普通加密且具有明确目标尺寸的实验。
            after_hash = sha256_text(canonical_json(proposal))  # 重新计算分类后提案摘要。
            if after_hash != before_hash:  # 检查分类过程是否改写了冻结提案。
                raise RuntimeError("executor mutated the frozen proposal")  # 发现改写时拒绝执行。
            mapping = {"proposal_sha256": proposal_hash, "operation": REFINE_TARGET_OPERATION, "mapping_reason": "proposal explicitly specifies a target mesh size", "proposal_unchanged": True, "target_size_mm": target_size}  # 组织模型不可见的显式尺寸映射收据。
            (round_dir / "mapping_receipt.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存隐藏映射记录。
            return mapping  # 返回显式目标尺寸内部操作。
    return previous_adapter.map_frozen_proposal(round_dir, proposal_hash)  # 对其他提案复用v2忠实映射。


def _execute_target_refinement(round_dir: Path, proposal_hash: str, mapping: dict[str, Any]) -> dict[str, Any]:  # 执行模型明确给出的普通网格目标尺寸。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 在执行前验证冻结提案完整性。
    before_hash = sha256_text(canonical_json(proposal))  # 保存执行前提案摘要。
    backend = _load_backend()  # 加载只读有限元后端和求解缓存。
    requested_size = float(mapping["target_size_mm"])  # 读取隐藏映射中保存的模型目标尺寸。
    if requested_size <= 0.0:  # 检查目标尺寸是否物理有效。
        raise ValueError("target mesh size must be positive")  # 拒绝非正尺寸。
    requested_divisions = backend.WIDTH / requested_size  # 把目标尺寸转换为全局结构网格划分数。
    used_nx = int(round(requested_divisions / 20.0) * 20)  # 吸附到后端允许的二十整数倍网格。
    used_nx = min(160, max(20, used_nx))  # 把网格限制在已验证计算预算范围。
    result = backend._public_result(backend._solve(used_nx))  # 执行真实有限元求解并提取公开结果。
    used_size = float(backend.WIDTH / used_nx)  # 计算实际使用的网格尺寸。
    after_hash = sha256_text(canonical_json(proposal))  # 重新计算执行后提案摘要。
    if after_hash != before_hash:  # 检查数值执行是否改写冻结提案。
        raise RuntimeError("executor mutated the frozen proposal")  # 在内容改变时拒绝结果。
    limitations: list[str] = []  # 初始化参数修复和模型边界说明。
    if abs(used_size - requested_size) > 1.0e-9:  # 检查目标尺寸是否因离散网格合同发生吸附。
        limitations.append("目标尺寸已吸附到后端允许的结构网格划分，实际尺寸与请求值不同")  # 向模型透明报告参数修复。
    feedback = {"status": "completed", "executed_change": f"保持几何、材料、载荷和边界不变，把网格目标尺寸从现有最细档调整为 {used_size:g} mm", "actual_parameters": {"requested_target_h_mm": requested_size, "used_nx": used_nx, "used_target_h_mm": used_size}, "observations": result, "limitations": limitations, "proposal_sha256": proposal_hash}  # 组织不含内部操作名的公开物理反馈。
    audit = {"proposal_sha256": proposal_hash, "internal_operation": REFINE_TARGET_OPERATION, "raw_result": result, "actual_parameters": feedback["actual_parameters"], "public_feedback": feedback}  # 组织完整模型不可见执行审计。
    (round_dir / "execution_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存内部操作和真实数值结果。
    (round_dir / "public_feedback.json").write_text(json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存下一轮唯一可见的公开反馈。
    return feedback  # 返回忠实目标尺寸求解结果。


def _sanitize_public_value(value: Any) -> Any:  # 递归删除模型可见反馈中的内部执行字段。
    if isinstance(value, dict):  # 检查当前值是否为对象。
        return {key: _sanitize_public_value(item) for key, item in value.items() if key not in INTERNAL_PUBLIC_KEYS}  # 删除内部字段并递归处理子对象。
    if isinstance(value, list):  # 检查当前值是否为数组。
        return [_sanitize_public_value(item) for item in value]  # 递归净化数组元素。
    return value  # 对普通数值和自然语言保持原样。


def _persist_sanitized_feedback(round_dir: Path, feedback: dict[str, Any]) -> dict[str, Any]:  # 保存净化后的公开反馈并同步内部审计引用。
    sanitized = _sanitize_public_value(feedback)  # 递归移除全部内部字段。
    (round_dir / "public_feedback.json").write_text(json.dumps(sanitized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 覆盖下一轮模型可见反馈文件。
    audit_path = round_dir / "execution_audit.json"  # 定位完整内部审计文件。
    if audit_path.is_file():  # 检查当前操作是否已经生成内部审计。
        audit = json.loads(audit_path.read_text(encoding="utf-8"))  # 读取内部审计记录。
        audit["public_feedback"] = sanitized  # 让审计文件引用最终净化版本。
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存同步后的内部审计。
    return sanitized  # 返回最终模型可见反馈。


def execute_mapping(round_dir: Path, proposal_hash: str, mapping: dict[str, Any]) -> dict[str, Any]:  # 执行显式目标加密或委托v2并统一净化公开反馈。
    if mapping.get("operation") == REFINE_TARGET_OPERATION:  # 检查是否需要执行显式目标尺寸加密。
        feedback = _execute_target_refinement(round_dir, proposal_hash, mapping)  # 执行忠实网格尺寸求解。
    else:  # 对其他操作复用v2多语言执行器。
        feedback = previous_adapter.execute_mapping(round_dir, proposal_hash, mapping)  # 执行既有隐藏操作。
    return _persist_sanitized_feedback(round_dir, feedback)  # 对所有公开反馈应用统一内部字段隔离。

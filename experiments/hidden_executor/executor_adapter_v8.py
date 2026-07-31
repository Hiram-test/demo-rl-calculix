from __future__ import annotations  # 启用现代类型注解并保持运行环境兼容。

import json  # 保存材料联合请求的隐藏映射、审计和公开反馈。
from pathlib import Path  # 管理冻结轮次目录和审计文件。
from typing import Any  # 表示动态提案、映射和反馈结构。

from . import executor_adapter_v7 as previous_adapter  # 复用作用域严格的位移、后处理和任务控制执行层。
from .contracts import canonical_json  # 规范序列化冻结提案用于完整性检查。
from .contracts import sha256_text  # 计算提案摘要以验证映射过程没有改写内容。
from .contracts import verify_frozen_proposal  # 读取并验证已经冻结的模型提案。

DISPLACEMENT_PROBE_OPERATION = previous_adapter.DISPLACEMENT_PROBE_OPERATION  # 复用原始裂纹面位移操作标识。
DISPLACEMENT_K_OPERATION = previous_adapter.DISPLACEMENT_K_OPERATION  # 复用位移法K联合操作标识。
DISPLACEMENT_MATERIAL_OPERATION = previous_adapter.DISPLACEMENT_MATERIAL_OPERATION  # 复用位移和材料联合操作标识。
STRESS_POSTPROCESS_OPERATION = previous_adapter.STRESS_POSTPROCESS_OPERATION  # 复用纯应力后处理操作标识。
FRACTURE_MATERIAL_OPERATION = "fracture_parameter_with_material_request"  # 定义断裂参量计算与外部材料请求联合操作。

_MATERIAL_TERMS = ("屈服强度", "塑性曲线", "应力-塑性应变", "硬化模型", "材料牌号", "材料参数", "断裂韧性", "yield strength", "plastic curve", "plastic strain", "hardening", "material grade", "material property", "fracture toughness")  # 定义明确的外部材料事实请求语义。
_FRACTURE_TERMS = ("应力强度因子", "k_i", "ki", "j积分", "j 积分", "能量释放率", "stress intensity factor", "j-integral", "j integral", "energy release rate")  # 定义可由当前数值解派生的断裂参量语义。
_DISPLACEMENT_TERMS = ("裂纹面位移", "节点位移", "张开位移", "u_y", "uy", "crack-face displacement", "crack face displacement", "nodal displacement", "opening displacement")  # 定义需要交给v7位移执行层的语义。
_REQUESTED_MATERIAL = ["屈服强度", "真实应力-塑性应变曲线", "硬化模型", "断裂韧性或损伤参数"]  # 定义当前工程适用性判断所需外部事实。


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:  # 检查文本是否包含任一中英文语义短语。
    return any(term in text for term in terms)  # 任一短语出现时返回真。


def _question_text(proposal: dict[str, Any]) -> str:  # 只读取request_information本轮真正提出的问题。
    request = proposal.get("information_request", {})  # 读取信息请求对象。
    return str(request.get("question", "")).lower() if isinstance(request, dict) else ""  # 返回小写问题文本并排除证据和不确定性污染。


def _write_mapping(round_dir: Path, proposal_hash: str, operation: str, reason: str) -> dict[str, Any]:  # 保存v8专用模型不可见映射收据。
    mapping = {"proposal_sha256": proposal_hash, "operation": operation, "mapping_reason": reason, "proposal_unchanged": True}  # 组织完整隐藏映射记录。
    (round_dir / "mapping_receipt.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存映射收据。
    return mapping  # 返回后续执行阶段使用的映射对象。


def map_frozen_proposal(round_dir: Path, proposal_hash: str) -> dict[str, Any]:  # 优先保护材料请求并避免重复断裂参量计算覆盖外部阻塞。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 在分类前验证冻结提案完整性。
    before_hash = sha256_text(canonical_json(proposal))  # 保存分类前提案摘要。
    if proposal.get("proposal_type") == "request_information":  # 只处理模型明确提出的数据请求。
        question = _question_text(proposal)  # 读取本轮实际问题文本。
        asks_displacement = _contains_any(question, _DISPLACEMENT_TERMS)  # 检查是否请求真实节点或裂纹面位移。
        if asks_displacement:  # 检查请求是否应由v7位移执行层处理。
            return previous_adapter.map_frozen_proposal(round_dir, proposal_hash)  # 保留位移、位移法K和位移材料联合目的。
        asks_material = _contains_any(question, _MATERIAL_TERMS)  # 检查本轮是否明确请求外部材料事实。
        asks_fracture = _contains_any(question, _FRACTURE_TERMS)  # 检查本轮是否同时请求现有数值解的断裂参量。
        after_hash = sha256_text(canonical_json(proposal))  # 重新计算分类后提案摘要。
        if after_hash != before_hash:  # 检查分类过程是否改写冻结提案。
            raise RuntimeError("executor mutated the frozen proposal")  # 发现改写时拒绝执行。
        if asks_material and asks_fracture:  # 检查是否属于可计算证据与外部事实联合请求。
            return _write_mapping(round_dir, proposal_hash, FRACTURE_MATERIAL_OPERATION, "question requests computable fracture-parameter evidence together with external material facts")  # 保留两个目的并在同一轮返回。
        if asks_material:  # 检查是否是纯外部材料数据请求。
            return _write_mapping(round_dir, proposal_hash, "request_material", "question requests external material or constitutive facts rather than another numerical fracture calculation")  # 直接标记真实外部阻塞。
    return previous_adapter.map_frozen_proposal(round_dir, proposal_hash)  # 对其他提案复用v7忠实隐藏映射。


def _execute_fracture_material(round_dir: Path, proposal_hash: str, mapping: dict[str, Any]) -> dict[str, Any]:  # 执行可计算断裂参量并同时保存材料外部阻塞。
    fracture_mapping = {"proposal_sha256": proposal_hash, "operation": "fracture_parameter_sequence", "mapping_reason": mapping.get("mapping_reason"), "proposal_unchanged": True}  # 构造只供内部执行的断裂参量子操作。
    feedback = previous_adapter.execute_mapping(round_dir, proposal_hash, fracture_mapping)  # 复用真实三档网格能量差G/K计算。
    feedback = dict(feedback)  # 复制基础反馈以避免改写调用方持有对象。
    observations = dict(feedback.get("observations", {}))  # 复制已经计算的断裂参量数值。
    observations["requested_information"] = list(_REQUESTED_MATERIAL)  # 同时记录隐藏执行器无法生成的外部材料事实。
    feedback["status"] = "information_required"  # 标记数值部分已返回但模型适用性仍被外部事实阻塞。
    feedback["executed_change"] = str(feedback.get("executed_change", "")) + "；同时记录材料适用性判断所需的外部数据"  # 向模型说明两个请求均被保留。
    feedback["observations"] = observations  # 写入断裂参量证据和材料请求。
    feedback["limitations"] = list(feedback.get("limitations", [])) + ["断裂参量数值已经计算，但缺少材料屈服与韧性数据时不能完成工程适用性判断"]  # 明确外部阻塞的工程意义。
    audit_path = round_dir / "execution_audit.json"  # 定位基础执行审计文件。
    audit = json.loads(audit_path.read_text(encoding="utf-8"))  # 读取基础断裂参量执行审计。
    audit["internal_operation"] = FRACTURE_MATERIAL_OPERATION  # 把审计操作更新为完整联合目的。
    audit["public_feedback"] = feedback  # 保存最终公开反馈。
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 写回完整联合审计。
    (round_dir / "public_feedback.json").write_text(json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 写回下一轮唯一允许看到的联合反馈。
    return feedback  # 返回数值证据与外部材料阻塞。


def execute_mapping(round_dir: Path, proposal_hash: str, mapping: dict[str, Any]) -> dict[str, Any]:  # 执行材料联合请求或委托v7执行层。
    if mapping.get("operation") == FRACTURE_MATERIAL_OPERATION:  # 检查是否需要同时计算断裂参量和记录材料阻塞。
        return _execute_fracture_material(round_dir, proposal_hash, mapping)  # 返回完整联合结果。
    return previous_adapter.execute_mapping(round_dir, proposal_hash, mapping)  # 对纯材料、位移、实验和任务控制复用v7执行层。

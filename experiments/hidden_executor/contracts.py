from __future__ import annotations  # 启用现代类型注解并避免运行时前向引用问题。

import hashlib  # 为冻结提案和审计链生成稳定哈希。
import json  # 以规范 JSON 形式保存提案和执行收据。
from datetime import datetime, timezone  # 记录统一 UTC 冻结时间。
from pathlib import Path  # 安全管理隔离实验输出路径。
from typing import Any  # 表示模型生成的动态 JSON 结构。

REQUIRED_EXPERIMENT_FIELDS = ("purpose", "change", "hold_fixed", "measure", "decision_rule", "stop_condition")  # 定义实验提案必须包含的字段。
ALLOWED_PROPOSAL_TYPES = {"experiment", "request_information", "stop"}  # 限制模型只能提出实验、请求信息或停止。


def canonical_json(value: Any) -> str:  # 把任意 JSON 值序列化为稳定文本。
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))  # 固定键顺序和分隔符以保证哈希可复现。


def sha256_text(text: str) -> str:  # 计算 UTF-8 文本的 SHA256。
    return hashlib.sha256(text.encode("utf-8")).hexdigest()  # 返回十六进制摘要用于审计。


def validate_proposal(proposal: dict[str, Any]) -> list[str]:  # 校验模型在看不到工具时提出的实验方案。
    errors: list[str] = []  # 初始化错误列表以一次返回全部合同问题。
    hypotheses = proposal.get("competing_hypotheses")  # 读取竞争假设数组。
    if not isinstance(hypotheses, list) or len(hypotheses) < 2 or not all(isinstance(item, str) and item.strip() for item in hypotheses):  # 检查至少两个非空假设。
        errors.append("competing_hypotheses must contain at least two non-empty strings")  # 记录假设合同错误。
    evidence_refs = proposal.get("evidence_refs")  # 读取证据引用数组。
    if not isinstance(evidence_refs, list) or not evidence_refs or not all(isinstance(item, str) and item.strip() for item in evidence_refs):  # 检查引用必须存在且非空。
        errors.append("evidence_refs must contain non-empty strings")  # 记录证据引用错误。
    uncertainties = proposal.get("uncertainties")  # 读取未决事项数组。
    if not isinstance(uncertainties, list) or not all(isinstance(item, str) and item.strip() for item in uncertainties):  # 检查未知项结构。
        errors.append("uncertainties must be an array of non-empty strings")  # 记录未知项错误。
    proposal_type = proposal.get("proposal_type")  # 读取提案类型。
    if proposal_type not in ALLOWED_PROPOSAL_TYPES:  # 检查类型是否属于通用合同。
        errors.append("proposal_type must be experiment, request_information, or stop")  # 记录类型错误。
    if proposal_type == "experiment":  # 仅对实验提案执行完整实验设计校验。
        experiment = proposal.get("experiment")  # 读取实验设计对象。
        if not isinstance(experiment, dict):  # 检查实验设计必须是对象。
            errors.append("experiment must be an object")  # 记录对象缺失错误。
        else:  # 在对象存在时逐字段校验。
            for field in REQUIRED_EXPERIMENT_FIELDS:  # 遍历全部必需字段。
                value = experiment.get(field)  # 读取当前字段值。
                if field in {"hold_fixed", "measure"}:  # 对数组字段使用数组合同。
                    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):  # 检查数组内容。
                        errors.append(f"experiment.{field} must contain non-empty strings")  # 记录数组字段错误。
                elif not isinstance(value, str) or not value.strip():  # 对文本字段检查非空字符串。
                    errors.append(f"experiment.{field} must be a non-empty string")  # 记录文本字段错误。
    if proposal_type == "request_information":  # 对请求信息提案执行信息合同校验。
        request = proposal.get("information_request")  # 读取请求对象。
        if not isinstance(request, dict) or not isinstance(request.get("question"), str) or not request.get("question", "").strip():  # 检查问题文本。
            errors.append("information_request.question must be a non-empty string")  # 记录信息请求错误。
    provisional_answer = proposal.get("provisional_answer")  # 读取当前暂定答复。
    if not isinstance(provisional_answer, str) or not provisional_answer.strip():  # 检查暂定答复必须存在。
        errors.append("provisional_answer must be a non-empty string")  # 记录答复合同错误。
    return errors  # 返回全部校验错误供调用方决定是否重试。


def freeze_proposal(proposal: dict[str, Any], output_dir: Path, round_index: int, previous_chain_hash: str) -> dict[str, Any]:  # 在映射和执行前冻结模型提案。
    errors = validate_proposal(proposal)  # 首先验证模型输出满足无工具提案合同。
    if errors:  # 在合同不满足时拒绝继续执行。
        raise ValueError("invalid proposal: " + "; ".join(errors))  # 抛出包含全部错误的异常。
    round_dir = output_dir / f"round_{round_index:02d}"  # 为当前轮创建独立目录。
    round_dir.mkdir(parents=True, exist_ok=True)  # 确保目录存在且不影响其他实验结果。
    proposal_text = canonical_json(proposal)  # 生成稳定提案文本用于冻结。
    proposal_hash = sha256_text(proposal_text)  # 计算提案内容哈希。
    sealed_at = datetime.now(timezone.utc).isoformat()  # 记录冻结时刻。
    chain_payload = canonical_json({"previous_chain_hash": previous_chain_hash, "proposal_sha256": proposal_hash, "round": round_index, "sealed_at_utc": sealed_at})  # 组合审计链载荷。
    chain_hash = sha256_text(chain_payload)  # 计算当前链节点哈希。
    (round_dir / "proposal.json").write_text(json.dumps(proposal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 先把模型原始提案写入磁盘。
    seal = {"round": round_index, "proposal_sha256": proposal_hash, "previous_chain_hash": previous_chain_hash, "chain_sha256": chain_hash, "sealed_at_utc": sealed_at}  # 组织冻结收据。
    (round_dir / "proposal_seal.json").write_text(json.dumps(seal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 把冻结收据写入独立文件。
    return {"round_dir": round_dir, "seal": seal, "proposal": proposal}  # 返回后续隐藏执行器所需的只读对象。


def verify_frozen_proposal(round_dir: Path, expected_hash: str) -> dict[str, Any]:  # 在执行前重新验证冻结提案未被改写。
    proposal = json.loads((round_dir / "proposal.json").read_text(encoding="utf-8"))  # 从磁盘重新读取冻结提案。
    actual_hash = sha256_text(canonical_json(proposal))  # 重新计算当前文件哈希。
    if actual_hash != expected_hash:  # 检查执行前内容是否发生变化。
        raise RuntimeError("frozen proposal hash mismatch")  # 拒绝执行被改写的提案。
    return proposal  # 返回通过完整性验证的提案。

from __future__ import annotations  # 启用现代类型注解并保持运行环境兼容。

import json  # 保存路线切换和任务完成候选的审计文件。
from pathlib import Path  # 管理冻结轮次目录。
from typing import Any  # 表示动态提案、映射和公开反馈结构。

from . import executor_adapter_v4 as previous_adapter  # 复用禁止部分执行和真实隐藏工具执行层。
from .contracts import verify_frozen_proposal  # 读取并验证已经冻结的模型提案。


def _write_special_feedback(round_dir: Path, proposal_hash: str, operation: str, feedback: dict[str, Any]) -> dict[str, Any]:  # 保存不调用有限元后端的任务控制反馈。
    audit = {"proposal_sha256": proposal_hash, "internal_operation": operation, "raw_result": feedback.get("observations", {}), "actual_parameters": {}, "public_feedback": feedback}  # 组织模型不可见的完整审计记录。
    (round_dir / "execution_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存内部任务控制审计。
    (round_dir / "public_feedback.json").write_text(json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存下一轮唯一允许回传给模型的公开反馈。
    return feedback  # 返回公开任务控制反馈。


def map_frozen_proposal(round_dir: Path, proposal_hash: str) -> dict[str, Any]:  # 区分路线切换、任务完成候选和真实隐藏工具提案。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 读取并验证冻结提案未被改写。
    proposal_type = str(proposal.get("proposal_type", ""))  # 读取模型声明的提案作用域。
    if proposal_type in {"switch_route", "stop"}:  # 把新路线切换和遗留stop都限制为当前路线结束。
        mapping = {"proposal_sha256": proposal_hash, "operation": "route_transition", "mapping_reason": "proposal ends or changes only the current reasoning route; the global task remains under controller review", "proposal_unchanged": True}  # 生成路线级映射收据。
        (round_dir / "mapping_receipt.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存模型不可见的路线切换收据。
        return mapping  # 返回路线切换操作且不触发全局结束。
    if proposal_type == "resolve_task":  # 处理模型声称原始问题已经完整解决的提案。
        mapping = {"proposal_sha256": proposal_hash, "operation": "task_resolution_candidate", "mapping_reason": "proposal requests global task completion and must pass independent decision-gate review", "proposal_unchanged": True}  # 生成任务完成候选收据。
        (round_dir / "mapping_receipt.json").write_text(json.dumps(mapping, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存模型不可见的完成候选收据。
        return mapping  # 返回任务控制操作供外层独立裁决。
    return previous_adapter.map_frozen_proposal(round_dir, proposal_hash)  # 对实验和信息请求继续使用真实隐藏执行器。


def execute_mapping(round_dir: Path, proposal_hash: str, mapping: dict[str, Any]) -> dict[str, Any]:  # 执行路线控制或委托真实隐藏工具执行。
    proposal = verify_frozen_proposal(round_dir, proposal_hash)  # 再次验证执行前冻结提案完整性。
    operation = str(mapping.get("operation", ""))  # 读取模型不可见的任务控制操作。
    if operation == "route_transition":  # 处理停止当前方法或切换工程路线。
        transition = proposal.get("route_transition", {})  # 读取新合同中的路线切换说明。
        if not isinstance(transition, dict):  # 兼容旧stop提案没有结构化路线字段。
            transition = {}  # 使用空对象并从暂定答复构造公开说明。
        route_conclusion = str(transition.get("route_conclusion") or proposal.get("provisional_answer") or "当前方法已经达到停止条件，但原始任务仍需继续。")  # 保存当前路线结论。
        next_route = str(transition.get("next_route") or "根据总体任务的未决决策门提出新的工程实验或信息请求。")  # 保存下一条模型自主路线。
        feedback = {"status": "route_transition", "executed_change": "未运行新模型；当前方法结束，但原始工程任务继续。", "actual_parameters": {}, "observations": {"route_conclusion": route_conclusion, "next_route": next_route, "task_continues": True}, "limitations": [], "proposal_sha256": proposal_hash}  # 组织不泄露内部工具名称的路线切换反馈。
        return _write_special_feedback(round_dir, proposal_hash, operation, feedback)  # 保存并返回路线级公开反馈。
    if operation == "task_resolution_candidate":  # 处理等待独立任务门裁决的完成声明。
        feedback = {"status": "resolution_candidate", "executed_change": "未运行新模型；总体任务完成声明等待独立决策门检查。", "actual_parameters": {}, "observations": {"final_answer": proposal.get("final_answer", {}), "controller_review_required": True}, "limitations": [], "proposal_sha256": proposal_hash}  # 组织模型可见的候选状态。
        return _write_special_feedback(round_dir, proposal_hash, operation, feedback)  # 保存并返回候选反馈供外层裁决。
    return previous_adapter.execute_mapping(round_dir, proposal_hash, mapping)  # 对真实实验和信息请求调用既有隐藏执行器。

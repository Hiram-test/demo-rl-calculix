from __future__ import annotations  # 启用现代类型注解并保持Python 3.11兼容。

import json  # 同步标准化后的公开反馈和内部审计文件。
from pathlib import Path  # 管理当前轮冻结目录。
from typing import Any  # 表示动态Skill反馈和审计对象。

from experiments.skill_planner.executor import execute_frozen_plan as previous_execute_frozen_plan  # 复用已经验证的确定性Skill执行器。
from experiments.skill_planner.registry import SkillRegistry  # 保持执行入口类型合同一致。


def _collect_requested_information(value: Any) -> list[str]:  # 递归提取组合Skill反馈中的外部信息请求。
    found: list[str] = []  # 初始化去重前请求列表。
    if isinstance(value, dict):  # 检查当前值是否为JSON对象。
        requested = value.get("requested_information")  # 读取当前层标准材料请求字段。
        if isinstance(requested, list):  # 检查请求字段是否为数组。
            found.extend(str(item) for item in requested if str(item).strip())  # 保存全部非空请求项。
        for child in value.values():  # 遍历对象全部子值。
            found.extend(_collect_requested_information(child))  # 递归收集嵌套result_groups中的请求。
    if isinstance(value, list):  # 检查当前值是否为JSON数组。
        for child in value:  # 遍历数组元素。
            found.extend(_collect_requested_information(child))  # 递归收集数组中的请求。
    return list(dict.fromkeys(found))  # 保持顺序并删除重复请求。


def execute_frozen_plan(round_dir: Path, proposal_hash: str, plan_hash: str, proposal: dict[str, Any], evidence_packet: dict[str, Any], registry: SkillRegistry) -> tuple[dict[str, Any], dict[str, Any]]:  # 执行冻结Skill图并标准化任务控制器所需外部阻塞字段。
    feedback, audit = previous_execute_frozen_plan(round_dir, proposal_hash, plan_hash, proposal, evidence_packet, registry)  # 调用既有确定性执行路径。
    requested = _collect_requested_information(feedback.get("observations", {}))  # 从组合公开结果中提取全部外部信息请求。
    if feedback.get("status") == "information_required" and requested:  # 检查本轮是否真实形成外部数据阻塞。
        observations = dict(feedback.get("observations", {}))  # 复制公开观测避免修改冻结计划。
        observations["requested_information"] = requested  # 在顶层增加既有任务控制器可识别的标准字段。
        feedback = dict(feedback)  # 复制公开反馈对象。
        feedback["observations"] = observations  # 写入标准化观测。
        audit = dict(audit)  # 复制内部审计对象。
        audit["public_feedback"] = feedback  # 让内部审计引用最终公开版本。
        (round_dir / "public_feedback.json").write_text(json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 覆盖第一模型下一轮可见反馈。
        (round_dir / "skill_execution_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 同步更新完整内部审计。
    return feedback, audit  # 返回标准化公开反馈和内部审计。

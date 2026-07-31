from __future__ import annotations  # 启用现代类型注解并保持Python 3.11兼容。

import json  # 同步标准化后的公开反馈和内部审计文件。
import re  # 从冻结提案和证据字符串中提取真实数值字面量。
from pathlib import Path  # 管理当前轮冻结目录。
from typing import Any  # 表示动态Skill反馈和审计对象。

from experiments.skill_planner import executor as previous_executor  # 读取并扩展既有确定性参数来源门。
from experiments.skill_planner.registry import SkillRegistry  # 保持执行入口类型合同一致。


_ORIGINAL_NUMBERS = previous_executor._numbers  # 保存既有结构化数值提取函数供非字符串值复用。


def _numbers_with_text(value: Any) -> list[float]:  # 同时提取JSON数值和自然语言字段中的显式数值字面量。
    if isinstance(value, str):  # 检查当前值是否为冻结自然语言文本。
        matches = re.findall(r"(?<![A-Za-z_])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", value)  # 提取独立整数、小数和科学计数法且避免变量名内数字。
        return [float(match) for match in matches]  # 返回文本中真实出现的全部数值。
    if isinstance(value, dict):  # 检查当前值是否为JSON对象。
        values: list[float] = []  # 初始化对象数值集合。
        for child in value.values():  # 遍历对象全部子值。
            values.extend(_numbers_with_text(child))  # 递归提取结构化值和文本字面量。
        return values  # 返回对象全部真实数值。
    if isinstance(value, list):  # 检查当前值是否为JSON数组。
        values: list[float] = []  # 初始化数组数值集合。
        for child in value:  # 遍历数组元素。
            values.extend(_numbers_with_text(child))  # 递归提取数组中的全部数值。
        return values  # 返回数组全部真实数值。
    return _ORIGINAL_NUMBERS(value)  # 对布尔值和普通数值保持既有严格语义。


previous_executor._numbers = _numbers_with_text  # 让既有运行时来源门识别冻结文本中明确写出的参数而不允许无来源数值。
previous_execute_frozen_plan = previous_executor.execute_frozen_plan  # 在扩展来源门后保存既有确定性Skill执行入口。


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
    feedback, audit = previous_execute_frozen_plan(round_dir, proposal_hash, plan_hash, proposal, evidence_packet, registry)  # 调用扩展数值溯源后的确定性执行路径。
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

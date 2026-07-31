from __future__ import annotations  # 启用现代类型注解并保持Python 3.11兼容。

import json  # 把冻结提案、证据包和Skill目录发送给第二API。
from typing import Any  # 表示动态JSON提案、计划和调用元数据。

from openai import OpenAI  # 通过兼容接口调用独立Skill规划API。

from experiments.skill_planner.contracts import validate_skill_plan  # 在返回调用方前校验第二API结构。
from experiments.skill_planner.registry import SkillRegistry  # 读取只对第二API可见的隐藏Skill目录。

PLANNER_SYSTEM_PROMPT = (  # 定义第二API唯一职责和严格输出合同。
    "你是隐藏Skill执行编译器，不负责重新解决工程问题，也不能改写第一模型已经冻结的提案。"  # 限制第二API只能翻译而不能替代工程决策代理。
    "你可以看到Skill目录，但第一模型看不到。请先把冻结提案规范化为experiment_spec，再选择零个或多个Skill形成有向无环执行计划。"  # 定义实验规范化和Skill组合任务。
    "experiment_spec必须包含objective、scope、interventions、invariants、observables、derivations、external_dependencies、acceptance_rule、completion_scope。"  # 定义统一实验中间表示。
    "observables、derivations和external_dependencies的每一项必须包含唯一id与description，这些id构成必须逐项覆盖的requirements。"  # 定义可审计需求集合。
    "plan_type只能是execute或unsupported。execute时calls必须完整覆盖全部requirements，不能只执行容易的一部分；unsupported时calls必须为空并说明原因。"  # 禁止部分执行和偷偷替换。
    "每个call必须包含call_id、skill_id、arguments、argument_sources、covers、depends_on。argument_sources的键必须与arguments完全一致，值只能说明参数来自proposal、initial_evidence、previous_rounds或skill_output:<call_id>。"  # 要求参数来源可追溯。
    "不得发明未在冻结提案或证据包中出现的几何、材料、载荷、网格或结果数值；不得用目录中相近Skill替换提案指定的方法。"  # 禁止参数幻觉和方法偷换。
    "当提案同时包含可计算任务和外部信息请求时，必须分别安排相应Skill并在同一计划中完整保留。"  # 支持组合Skill而不丢失其中一部分。
    "只输出一个合法JSON对象，不输出解释文本。"  # 限制响应便于冻结和确定性验证。
)  # 完成第二API系统提示词。


def _usage_dict(usage: Any) -> dict[str, Any]:  # 把SDK使用量对象转换为可序列化字典。
    if usage is None:  # 检查响应是否缺少使用量。
        return {}  # 在缺失时返回空对象。
    if hasattr(usage, "model_dump"):  # 优先使用新版SDK序列化方法。
        return usage.model_dump()  # 返回完整使用量字段。
    return {name: getattr(usage, name) for name in ("prompt_tokens", "completion_tokens", "total_tokens") if getattr(usage, name, None) is not None}  # 兼容旧版SDK字段。


def request_skill_plan(client: OpenAI, model: str, proposal: dict[str, Any], evidence_packet: dict[str, Any], registry: SkillRegistry) -> tuple[dict[str, Any], dict[str, Any]]:  # 请求第二API把冻结提案编译为Skill调用计划。
    payload = {"frozen_proposal": proposal, "available_skills": registry.catalog(), "evidence_packet": evidence_packet}  # 组织第二API可见的提案、目录和证据。
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)  # 序列化为稳定可读文本。
    last_error: Exception | None = None  # 保存最近一次结构错误用于最终诊断。
    for attempt in range(1, 3):  # 最多允许一次只修复JSON或合同结构的重试。
        repair = "" if attempt == 1 else "上一次输出未满足Skill计划合同。保持实验目的和Skill选择意图不变，只修复JSON结构、覆盖声明和参数来源。\n\n"  # 防止重试时改写工程提案。
        response = client.chat.completions.create(  # 发起真实第二API调用。
            model=model,  # 使用独立环境变量配置的规划模型。
            messages=[{"role": "system", "content": PLANNER_SYSTEM_PROMPT}, {"role": "user", "content": repair + payload_text}],  # 只向第二API暴露隐藏Skill目录。
            response_format={"type": "json_object"},  # 要求API返回合法JSON对象。
            max_tokens=10000,  # 为实验规范和多Skill执行图保留足够输出预算。
            reasoning_effort="high",  # 请求高强度语义编译和保真检查。
            extra_body={"thinking": {"type": "enabled"}},  # 显式启用兼容API的思考模式但不保存隐藏推理。
            stream=False,  # 使用单次完整响应便于冻结计划。
        )  # 完成第二API请求。
        content = response.choices[0].message.content  # 读取公开Skill计划JSON文本。
        if not content or not content.strip():  # 检查第二API是否返回空计划。
            last_error = RuntimeError("Skill planner API returned empty content")  # 保存空响应错误。
            continue  # 进入结构修复重试。
        try:  # 尝试解析第二API JSON。
            plan = json.loads(content)  # 把响应文本转换为计划对象。
        except json.JSONDecodeError as exc:  # 捕获JSON语法错误。
            last_error = exc  # 保存解析错误。
            continue  # 允许一次格式修复重试。
        errors = validate_skill_plan(plan)  # 按冻结Skill计划合同检查结构和完整覆盖。
        if errors:  # 检查是否存在遗漏、部分执行或结构错误。
            last_error = RuntimeError("invalid skill plan: " + "; ".join(errors))  # 保存全部合同错误。
            continue  # 允许第二API仅修复合同一次。
        metadata = {"requested_model": model, "response_model": response.model, "finish_reason": response.choices[0].finish_reason, "usage": _usage_dict(response.usage)}  # 保存真实规划模型和令牌使用量。
        return plan, metadata  # 返回通过结构合同的第二API计划。
    raise RuntimeError(f"Skill planner API did not return a valid plan after two attempts: {last_error}")  # 在两次失败后诚实终止当前实验轮次。

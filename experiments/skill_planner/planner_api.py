from __future__ import annotations  # 启用现代类型注解并保持Python 3.11兼容。

import json  # 把冻结提案、证据包和Skill目录发送给第二API并保存尝试结果。
from pathlib import Path  # 管理每轮第二API原始响应审计文件。
from typing import Any  # 表示动态JSON提案、计划和调用元数据。

from openai import OpenAI  # 通过兼容接口调用独立Skill规划API。

from experiments.skill_planner.contracts import validate_skill_plan  # 在返回调用方前校验第二API结构。
from experiments.skill_planner.registry import SkillRegistry  # 读取只对第二API可见的隐藏Skill目录。

_PLAN_TEMPLATE = {"experiment_spec": {"objective": "非空字符串", "scope": "experiment或information_request", "interventions": ["字符串"], "invariants": ["字符串"], "observables": [{"id": "唯一id", "description": "非空字符串"}], "derivations": [{"id": "唯一id", "description": "非空字符串"}], "external_dependencies": [{"id": "唯一id", "description": "非空字符串"}], "acceptance_rule": "非空字符串", "completion_scope": "current_step、current_route或global_task"}, "plan_type": "execute或unsupported", "calls": [{"call_id": "call_1", "skill_id": "目录中的精确标识", "arguments": {}, "argument_sources": {}, "covers": ["requirement_id"], "depends_on": []}], "uncovered_requirements": [], "proposal_fully_preserved": True, "unsupported_reason": None}  # 定义第二API必须遵守的精确JSON骨架。

PLANNER_SYSTEM_PROMPT = (  # 定义第二API唯一职责和严格输出合同。
    "你是隐藏Skill执行编译器，不负责重新解决工程问题，也不能改写第一模型已经冻结的提案。"  # 限制第二API只能翻译而不能替代工程决策代理。
    "你可以看到Skill目录，但第一模型看不到。请先把冻结提案规范化为experiment_spec，再选择零个或多个Skill形成有向无环执行计划。"  # 定义实验规范化和Skill组合任务。
    "experiment_spec必须包含objective、scope、interventions、invariants、observables、derivations、external_dependencies、acceptance_rule、completion_scope。"  # 定义统一实验中间表示。
    "scope只能逐字使用experiment或information_request；completion_scope只能逐字使用current_step、current_route或global_task。"  # 固定枚举值避免自由翻译。
    "observables、derivations和external_dependencies的每一项必须包含唯一id与description，这些id构成必须逐项覆盖的requirements。"  # 定义可审计需求集合。
    "plan_type只能逐字使用execute或unsupported。execute时calls必须完整覆盖全部requirements，不能只执行容易的一部分；unsupported时calls必须为空并说明原因。"  # 禁止部分执行和偷偷替换。
    "每个call必须包含call_id、skill_id、arguments、argument_sources、covers、depends_on。argument_sources的键必须与arguments完全一致，值只能是proposal、initial_evidence、previous_rounds或skill_output:<call_id>。"  # 要求参数来源可追溯。
    "proposal_fully_preserved必须是JSON布尔值true或false，绝不能输出字符串、yes、no或中文。"  # 固定保真字段机器类型。
    "不得发明未在冻结提案或证据包中出现的几何、材料、载荷、网格或结果数值；不得用目录中相近Skill替换提案指定的方法。"  # 禁止参数幻觉和方法偷换。
    "当提案同时包含可计算任务和外部信息请求时，必须分别安排相应Skill并在同一计划中完整保留。"  # 支持组合Skill而不丢失其中一部分。
    "严格按照用户消息中给出的output_template输出一个合法JSON对象，不输出解释文本，也不要输出枚举占位符中的竖线。"  # 强制第二API参照精确骨架。
)  # 完成第二API系统提示词。

_SCOPE_ALIASES = {"experiment": "experiment", "实验": "experiment", "current_experiment": "experiment", "information_request": "information_request", "request_information": "information_request", "信息请求": "information_request", "request": "information_request"}  # 定义不改变提案语义的scope结构别名。
_COMPLETION_ALIASES = {"current_step": "current_step", "step": "current_step", "current_experiment": "current_step", "experiment": "current_step", "当前步骤": "current_step", "当前实验": "current_step", "current_route": "current_route", "route": "current_route", "当前路线": "current_route", "global_task": "global_task", "task": "global_task", "global": "global_task", "总体任务": "global_task"}  # 定义停止作用域的无语义归一化别名。


def _usage_dict(usage: Any) -> dict[str, Any]:  # 把SDK使用量对象转换为可序列化字典。
    if usage is None:  # 检查响应是否缺少使用量。
        return {}  # 在缺失时返回空对象。
    if hasattr(usage, "model_dump"):  # 优先使用新版SDK序列化方法。
        return usage.model_dump()  # 返回完整使用量字段。
    return {name: getattr(usage, name) for name in ("prompt_tokens", "completion_tokens", "total_tokens") if getattr(usage, name, None) is not None}  # 兼容旧版SDK字段。


def _normalize_structural_aliases(plan: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:  # 只修复枚举别名和JSON布尔字符串，不改变实验或Skill语义。
    normalized = json.loads(json.dumps(plan, ensure_ascii=False))  # 深拷贝第二API计划避免修改原始响应对象。
    changes: list[str] = []  # 初始化结构归一化审计列表。
    spec = normalized.get("experiment_spec")  # 读取实验规范对象。
    if isinstance(spec, dict):  # 检查规范对象是否存在。
        raw_scope = spec.get("scope")  # 读取第二API原始scope值。
        scope_key = str(raw_scope).strip().lower() if raw_scope is not None else ""  # 统一字符串形式用于别名查找。
        if scope_key in _SCOPE_ALIASES and raw_scope != _SCOPE_ALIASES[scope_key]:  # 检查是否为已知结构别名。
            spec["scope"] = _SCOPE_ALIASES[scope_key]  # 转换为合同精确枚举。
            changes.append(f"experiment_spec.scope:{raw_scope}->{spec['scope']}")  # 记录无语义结构修复。
        raw_completion = spec.get("completion_scope")  # 读取第二API原始停止作用域。
        completion_key = str(raw_completion).strip().lower() if raw_completion is not None else ""  # 统一字符串形式。
        if completion_key in _COMPLETION_ALIASES and raw_completion != _COMPLETION_ALIASES[completion_key]:  # 检查是否为已知别名。
            spec["completion_scope"] = _COMPLETION_ALIASES[completion_key]  # 转换为合同精确枚举。
            changes.append(f"experiment_spec.completion_scope:{raw_completion}->{spec['completion_scope']}")  # 记录结构修复。
    raw_preserved = normalized.get("proposal_fully_preserved")  # 读取完整保真字段。
    if isinstance(raw_preserved, str):  # 检查第二API是否错误输出字符串布尔值。
        lowered = raw_preserved.strip().lower()  # 统一字符串布尔表达。
        if lowered in {"true", "yes", "是", "完整", "fully_preserved"}:  # 检查明确真值别名。
            normalized["proposal_fully_preserved"] = True  # 转换为JSON布尔真。
            changes.append(f"proposal_fully_preserved:{raw_preserved}->true")  # 记录类型修复。
        if lowered in {"false", "no", "否", "不完整", "not_preserved"}:  # 检查明确假值别名。
            normalized["proposal_fully_preserved"] = False  # 转换为JSON布尔假。
            changes.append(f"proposal_fully_preserved:{raw_preserved}->false")  # 记录类型修复。
    return normalized, changes  # 返回只做结构别名修复的计划和审计记录。


def _write_attempt(audit_dir: Path | None, attempt: int, content: str, parsed: dict[str, Any] | None, normalized: dict[str, Any] | None, errors: list[str], changes: list[str]) -> None:  # 保存第二API每次原始和归一化响应。
    if audit_dir is None:  # 检查调用方是否提供独立轮次目录。
        return  # 在单元测试或其他调用中允许不写盘。
    audit = {"attempt": attempt, "raw_content": content, "parsed_plan": parsed, "normalized_plan": normalized, "normalization_changes": changes, "validation_errors": errors}  # 组织完整第二API尝试审计。
    (audit_dir / f"planner_attempt_{attempt:02d}.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存原始响应、结构修复和错误。


def request_skill_plan(client: OpenAI, model: str, proposal: dict[str, Any], evidence_packet: dict[str, Any], registry: SkillRegistry, audit_dir: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:  # 请求第二API把冻结提案编译为Skill调用计划。
    payload = {"frozen_proposal": proposal, "available_skills": registry.catalog(), "evidence_packet": evidence_packet, "output_template": _PLAN_TEMPLATE}  # 组织第二API可见提案、目录、证据和精确输出骨架。
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)  # 序列化为稳定可读文本。
    last_error: Exception | None = None  # 保存最近一次结构错误用于最终诊断。
    previous_content = ""  # 保存上一次无效原始响应供修复轮读取。
    previous_errors: list[str] = []  # 保存上一次合同错误供精确修复。
    attempt_metadata: list[dict[str, Any]] = []  # 保存两次规划调用的模型和使用量。
    for attempt in range(1, 3):  # 最多允许一次只修复JSON或合同结构的重试。
        repair = ""  # 初始化首轮无修复指令。
        if attempt > 1:  # 检查是否进入结构修复重试。
            repair = "上一次输出如下：\n" + previous_content + "\n\n合同错误如下：\n- " + "\n- ".join(previous_errors) + "\n\n保持experiment_spec的工程语义、Skill选择、参数和覆盖关系不变，只把字段改成output_template要求的精确枚举与JSON类型，然后重新输出完整JSON。\n\n"  # 把实际错误和原始输出反馈给第二API。
        response = client.chat.completions.create(  # 发起真实第二API调用。
            model=model,  # 使用独立环境变量配置的规划模型。
            messages=[{"role": "system", "content": PLANNER_SYSTEM_PROMPT}, {"role": "user", "content": repair + payload_text}],  # 只向第二API暴露隐藏Skill目录。
            response_format={"type": "json_object"},  # 要求API返回合法JSON对象。
            max_tokens=10000,  # 为实验规范和多Skill执行图保留足够输出预算。
            reasoning_effort="high",  # 请求高强度语义编译和保真检查。
            extra_body={"thinking": {"type": "enabled"}},  # 显式启用兼容API的思考模式但不保存隐藏推理。
            stream=False,  # 使用单次完整响应便于冻结计划。
        )  # 完成第二API请求。
        attempt_metadata.append({"attempt": attempt, "requested_model": model, "response_model": response.model, "finish_reason": response.choices[0].finish_reason, "usage": _usage_dict(response.usage)})  # 保存本次真实规划调用元数据。
        content = response.choices[0].message.content or ""  # 读取公开Skill计划JSON文本。
        previous_content = content  # 保存原始响应供下一次精确修复。
        if not content.strip():  # 检查第二API是否返回空计划。
            previous_errors = ["Skill planner API returned empty content"]  # 保存空响应错误。
            last_error = RuntimeError(previous_errors[0])  # 更新最终诊断。
            _write_attempt(audit_dir, attempt, content, None, None, previous_errors, [])  # 保存空响应审计。
            continue  # 进入结构修复重试。
        try:  # 尝试解析第二API JSON。
            parsed = json.loads(content)  # 把响应文本转换为原始计划对象。
        except json.JSONDecodeError as exc:  # 捕获JSON语法错误。
            previous_errors = [str(exc)]  # 保存解析错误文本。
            last_error = exc  # 更新最终诊断。
            _write_attempt(audit_dir, attempt, content, None, None, previous_errors, [])  # 保存无法解析的原始响应。
            continue  # 允许一次格式修复重试。
        normalized, changes = _normalize_structural_aliases(parsed)  # 只修复精确枚举别名和布尔类型。
        errors = validate_skill_plan(normalized)  # 按冻结Skill计划合同检查结构和完整覆盖。
        _write_attempt(audit_dir, attempt, content, parsed, normalized, errors, changes)  # 保存原始、归一化和合同错误。
        if errors:  # 检查是否仍存在遗漏、部分执行或结构错误。
            previous_errors = errors  # 把实际错误反馈给下一轮修复。
            last_error = RuntimeError("invalid skill plan: " + "; ".join(errors))  # 保存全部合同错误。
            continue  # 允许第二API仅修复合同一次。
        metadata = {"requested_model": model, "response_model": response.model, "finish_reason": response.choices[0].finish_reason, "usage": _usage_dict(response.usage), "attempts": attempt_metadata, "normalization_changes": changes}  # 保存最终规划模型、调用次数和结构修复记录。
        return normalized, metadata  # 返回通过结构合同的第二API计划。
    raise RuntimeError(f"Skill planner API did not return a valid plan after two attempts: {last_error}")  # 在两次失败后诚实终止当前实验轮次。

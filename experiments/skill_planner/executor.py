from __future__ import annotations  # 启用现代类型注解并保持Python 3.11兼容。

import json  # 保存计划校验收据、内部Skill审计和公开物理反馈。
from pathlib import Path  # 管理每轮冻结目录。
from typing import Any  # 表示动态计划、参数、证据和Skill结果。

from experiments.skill_planner.contracts import verify_skill_plan  # 在执行前验证第二API计划未被改写。
from experiments.skill_planner.registry import SkillContext  # 构造每个Skill可读取的真实证据上下文。
from experiments.skill_planner.registry import SkillRegistry  # 校验Skill标识、参数合同并调用处理函数。

_ALLOWED_SOURCE_PREFIXES = ("proposal", "initial_evidence", "previous_rounds", "skill_output:")  # 定义第二API参数来源的允许范围。


def _numbers(value: Any) -> list[float]:  # 递归提取JSON结构中的全部非布尔数值。
    values: list[float] = []  # 初始化数值列表。
    if isinstance(value, bool):  # 检查布尔值避免被Python视为整数。
        return values  # 布尔参数不参与数值来源校验。
    if isinstance(value, (int, float)):  # 检查当前值是否为普通数值。
        return [float(value)]  # 返回单个数值列表。
    if isinstance(value, dict):  # 检查当前值是否为对象。
        for child in value.values():  # 遍历对象全部子值。
            values.extend(_numbers(child))  # 递归收集数值。
    if isinstance(value, list):  # 检查当前值是否为数组。
        for child in value:  # 遍历数组元素。
            values.extend(_numbers(child))  # 递归收集数值。
    return values  # 返回全部数值。


def _numeric_values_grounded(arguments: dict[str, Any], argument_sources: dict[str, Any], proposal: dict[str, Any], evidence_packet: dict[str, Any]) -> list[str]:  # 检查第二API没有发明数值参数。
    errors: list[str] = []  # 初始化数值来源错误列表。
    grounded = _numbers({"proposal": proposal, "evidence_packet": evidence_packet})  # 收集第一模型提案和真实证据中的全部数值。
    for name, value in arguments.items():  # 遍历当前Skill全部参数。
        source = str(argument_sources.get(name, ""))  # 读取第二API声明的参数来源。
        if source.startswith("skill_output:"):  # 检查参数是否来自当前执行图前序Skill。
            continue  # 前序输出将在执行依赖阶段验证，不要求出现在初始证据中。
        for number in _numbers(value):  # 遍历当前参数中的全部数值。
            matched = any(abs(number - candidate) <= 1.0e-9 * max(1.0, abs(number), abs(candidate)) for candidate in grounded)  # 检查数值是否真实出现在冻结提案或证据中。
            if not matched:  # 检查第二API是否生成了无来源数值。
                errors.append(f"argument {name} contains ungrounded numeric value {number}")  # 记录参数幻觉错误。
    return errors  # 返回全部无来源数值错误。


def validate_runtime_plan(plan: dict[str, Any], proposal: dict[str, Any], evidence_packet: dict[str, Any], registry: SkillRegistry) -> list[str]:  # 对冻结计划实施Skill目录和参数来源的确定性校验。
    errors: list[str] = []  # 初始化运行时计划错误列表。
    if plan.get("plan_type") == "unsupported":  # 检查第二API是否诚实声明无法完整执行。
        return errors  # 结构合同已经校验，不支持计划无需Skill运行时检查。
    seen_calls: set[str] = set()  # 保存按照计划顺序已经出现的调用标识。
    requirement_ids = {str(item.get("id")) for field in ("observables", "derivations", "external_dependencies") for item in plan.get("experiment_spec", {}).get(field, []) if isinstance(item, dict)}  # 提取实验规范全部要求标识。
    for index, call in enumerate(plan.get("calls", [])):  # 按冻结顺序遍历Skill调用。
        call_id = str(call.get("call_id", ""))  # 读取当前调用标识。
        skill_id = str(call.get("skill_id", ""))  # 读取当前Skill标识。
        arguments = call.get("arguments", {})  # 读取当前Skill参数。
        argument_sources = call.get("argument_sources", {})  # 读取参数来源声明。
        dependencies = [str(value) for value in call.get("depends_on", [])]  # 读取有向执行依赖。
        if skill_id not in registry.ids():  # 检查第二API是否引用目录外能力。
            errors.append(f"calls[{index}] references unknown skill {skill_id}")  # 记录未知Skill错误。
            continue  # 跳过未知Skill的参数合同检查。
        errors.extend(registry.validate_arguments(skill_id, arguments))  # 按不可变Skill合同校验参数类型和必需性。
        for parameter, source in argument_sources.items():  # 遍历每个参数来源声明。
            source_text = str(source)  # 把动态来源值转换为文本。
            if not any(source_text == prefix or source_text.startswith(prefix) for prefix in _ALLOWED_SOURCE_PREFIXES):  # 检查来源是否属于允许范围。
                errors.append(f"calls[{index}].argument_sources[{parameter}] is invalid")  # 记录未授权来源。
        errors.extend(_numeric_values_grounded(arguments, argument_sources, proposal, evidence_packet))  # 检查所有非Skill输出数值均来自冻结证据。
        for dependency in dependencies:  # 遍历当前调用依赖。
            if dependency not in seen_calls:  # 检查依赖是否已经在计划前方完成。
                errors.append(f"calls[{index}] dependency {dependency} is not an earlier call")  # 拒绝环或未来依赖。
        covers = {str(value) for value in call.get("covers", [])}  # 读取当前调用覆盖的实验要求。
        if not covers or not covers.issubset(requirement_ids):  # 检查覆盖要求是否真实存在且非空。
            errors.append(f"calls[{index}] covers invalid requirements")  # 记录无效覆盖声明。
        seen_calls.add(call_id)  # 在完成当前检查后把调用加入已见集合。
    return errors  # 返回全部确定性Skill计划错误。


def _resolve_skill_output_arguments(arguments: dict[str, Any], sources: dict[str, Any], prior_outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:  # 解析显式引用前序Skill输出的参数。
    resolved = dict(arguments)  # 复制计划参数避免改写冻结对象。
    for name, source in sources.items():  # 遍历参数来源声明。
        source_text = str(source)  # 读取来源文本。
        if not source_text.startswith("skill_output:"):  # 检查是否需要从前序Skill取值。
            continue  # 对提案或证据来源参数保持冻结字面值。
        parts = source_text.split(":", 2)  # 拆分skill_output、调用标识和可选字段路径。
        if len(parts) < 2 or parts[1] not in prior_outputs:  # 检查前序调用是否真实完成。
            raise RuntimeError(f"missing prior skill output for argument {name}")  # 拒绝无法解析的执行依赖。
        value: Any = prior_outputs[parts[1]]  # 从前序Skill完整结果开始解析。
        if len(parts) == 3 and parts[2]:  # 检查是否指定点路径。
            for token in parts[2].split("."):  # 按点分隔逐层读取对象字段。
                if not isinstance(value, dict) or token not in value:  # 检查字段路径是否真实存在。
                    raise RuntimeError(f"invalid prior skill output path for argument {name}")  # 拒绝第二API编造的字段路径。
                value = value[token]  # 进入下一层真实输出字段。
        resolved[name] = value  # 用已验证前序输出覆盖计划占位值。
    return resolved  # 返回当前Skill最终执行参数。


def _unsupported_feedback(plan: dict[str, Any], proposal_hash: str) -> dict[str, Any]:  # 把第二API诚实不支持结果转换为第一模型可见反馈。
    return {"status": "unsupported", "executed_change": "第二API检查了冻结提案和隐藏Skill目录，但无法在不改变实验目的的前提下形成完整执行计划", "actual_parameters": {}, "observations": {"reason": str(plan.get("unsupported_reason", "完整Skill覆盖不可用")), "uncovered_requirements": list(plan.get("uncovered_requirements", []))}, "limitations": ["没有调用任何Skill，也没有用相近方法替换冻结提案"] , "proposal_sha256": proposal_hash}  # 返回不暴露Skill名称的诚实拒绝反馈。


def execute_frozen_plan(round_dir: Path, proposal_hash: str, plan_hash: str, proposal: dict[str, Any], evidence_packet: dict[str, Any], registry: SkillRegistry) -> tuple[dict[str, Any], dict[str, Any]]:  # 校验并执行第二API冻结的Skill调用图。
    plan = verify_skill_plan(round_dir, plan_hash)  # 从磁盘读取并验证计划内容未被改写。
    validation_errors = validate_runtime_plan(plan, proposal, evidence_packet, registry)  # 执行Skill目录、参数来源和依赖检查。
    validation_receipt = {"proposal_sha256": proposal_hash, "skill_plan_sha256": plan_hash, "skill_catalog_sha256": registry.catalog_hash(), "valid": not validation_errors, "errors": validation_errors}  # 组织确定性计划校验收据。
    (round_dir / "skill_plan_validation.json").write_text(json.dumps(validation_receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存独立校验结果。
    if validation_errors:  # 检查计划是否通过全部确定性门。
        raise ValueError("Skill plan failed deterministic validation: " + "; ".join(validation_errors))  # 拒绝执行错误或幻觉计划。
    if plan.get("plan_type") == "unsupported":  # 检查第二API是否没有完整保真的可执行计划。
        feedback = _unsupported_feedback(plan, proposal_hash)  # 生成不泄露目录的公开拒绝反馈。
        audit = {"proposal_sha256": proposal_hash, "skill_plan_sha256": plan_hash, "plan_type": "unsupported", "planner_plan": plan, "skill_calls": [], "public_feedback": feedback}  # 保存完整模型不可见审计。
        (round_dir / "skill_execution_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存内部计划和拒绝原因。
        (round_dir / "public_feedback.json").write_text(json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存第一模型下一轮唯一可见反馈。
        return feedback, audit  # 返回诚实不支持结果和内部审计。
    prior_outputs: dict[str, dict[str, Any]] = {}  # 初始化当前执行图前序Skill输出。
    call_audits: list[dict[str, Any]] = []  # 初始化模型不可见的逐Skill审计。
    public_results: list[dict[str, Any]] = []  # 初始化按实验要求组织的公开结果片段。
    executed_changes: list[str] = []  # 初始化公开执行变化摘要。
    limitations: list[str] = []  # 初始化组合Skill限制列表。
    statuses: list[str] = []  # 初始化逐Skill公开状态。
    for call in plan.get("calls", []):  # 按冻结有向图拓扑顺序执行Skill。
        call_id = str(call["call_id"])  # 读取当前调用标识供依赖和审计使用。
        skill_id = str(call["skill_id"])  # 读取当前隐藏Skill标识。
        skill = registry.get(skill_id)  # 从不可变注册表读取处理函数和合同。
        resolved_arguments = _resolve_skill_output_arguments(call.get("arguments", {}), call.get("argument_sources", {}), prior_outputs)  # 解析前序Skill输出引用。
        context = SkillContext(initial_evidence=evidence_packet["initial_evidence"], public_history=evidence_packet.get("previous_rounds", []), prior_skill_outputs=prior_outputs)  # 构造只含真实证据的Skill上下文。
        result = skill.handler(resolved_arguments, context)  # 调用真实Skill处理函数。
        prior_outputs[call_id] = result  # 保存完整结果供后续依赖引用。
        statuses.append(str(result.get("status", "completed")))  # 保存当前Skill状态。
        change = str(result.get("executed_change", "")).strip()  # 读取公开物理变化摘要。
        if change and change not in executed_changes:  # 检查是否需要追加新的变化描述。
            executed_changes.append(change)  # 保存去重后的执行变化。
        for limitation in result.get("limitations", []):  # 遍历当前Skill固有限制。
            text = str(limitation)  # 转换为公开文本。
            if text and text not in limitations:  # 检查是否已经记录相同限制。
                limitations.append(text)  # 保存去重后的限制。
        public_results.append({"requirements": list(call.get("covers", [])), "actual_parameters": result.get("actual_parameters", {}), "observations": result.get("observations", {}), "status": result.get("status", "completed")})  # 按实验规范要求公开结果但隐藏Skill标识。
        call_audits.append({"call_id": call_id, "skill_id": skill_id, "arguments": resolved_arguments, "argument_sources": call.get("argument_sources", {}), "covers": call.get("covers", []), "depends_on": call.get("depends_on", []), "raw_result": result})  # 保存完整内部Skill调用审计。
    overall_status = "information_required" if "information_required" in statuses else "completed"  # 只要任一Skill需要外部事实就保留阻塞状态。
    feedback = {"status": overall_status, "executed_change": "；".join(executed_changes) if executed_changes else "完成冻结实验规范对应的隐藏Skill执行图", "actual_parameters": {"result_groups": [{"requirements": item["requirements"], "actual_parameters": item["actual_parameters"]} for item in public_results]}, "observations": {"result_groups": [{"requirements": item["requirements"], "observations": item["observations"], "status": item["status"]} for item in public_results]}, "limitations": limitations, "proposal_sha256": proposal_hash}  # 组织第一模型可见的组合物理反馈。
    audit = {"proposal_sha256": proposal_hash, "skill_plan_sha256": plan_hash, "plan_type": "execute", "experiment_spec": plan.get("experiment_spec", {}), "skill_calls": call_audits, "public_feedback": feedback}  # 组织完整模型不可见Skill执行审计。
    (round_dir / "skill_execution_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存Skill标识、参数和原始结果。
    (round_dir / "public_feedback.json").write_text(json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存第一模型下一轮唯一可见物理反馈。
    return feedback, audit  # 返回公开反馈和完整内部审计。

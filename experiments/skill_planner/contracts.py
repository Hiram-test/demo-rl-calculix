from __future__ import annotations  # 启用现代类型注解并保持Python 3.11兼容。

import json  # 保存Skill计划、封条和校验收据。
from datetime import datetime, timezone  # 记录统一UTC冻结时间。
from pathlib import Path  # 管理每轮独立结果目录。
from typing import Any  # 表示第二API生成的动态JSON对象。

from experiments.hidden_executor.contracts import canonical_json  # 复用仓库既有稳定JSON序列化合同。
from experiments.hidden_executor.contracts import sha256_text  # 复用仓库既有SHA256实现。

ALLOWED_PLAN_TYPES = {"execute", "unsupported"}  # 限制第二API只能生成可执行计划或诚实不支持结果。
ALLOWED_SPEC_SCOPES = {"experiment", "information_request"}  # 限制Skill规划器只处理需要执行或取数的提案。
ALLOWED_COMPLETION_SCOPES = {"current_step", "current_route", "global_task"}  # 定义实验规范中的停止作用域。


def _non_empty_text(value: Any) -> bool:  # 检查任意值是否为非空文本。
    return isinstance(value, str) and bool(value.strip())  # 只接受去除空白后仍有内容的字符串。


def _validate_named_items(value: Any, field_name: str, errors: list[str]) -> list[str]:  # 校验带id和description的规范条目数组。
    ids: list[str] = []  # 初始化条目id列表用于后续覆盖检查。
    if not isinstance(value, list):  # 检查目标字段必须是数组。
        errors.append(f"experiment_spec.{field_name} must be an array")  # 记录字段类型错误。
        return ids  # 在类型错误时返回空列表。
    for index, item in enumerate(value):  # 遍历每个规范条目。
        if not isinstance(item, dict):  # 检查条目必须是对象。
            errors.append(f"experiment_spec.{field_name}[{index}] must be an object")  # 记录对象错误。
            continue  # 继续检查其他条目。
        item_id = item.get("id")  # 读取条目稳定标识。
        description = item.get("description")  # 读取条目自然语言说明。
        if not _non_empty_text(item_id):  # 检查条目标识不能为空。
            errors.append(f"experiment_spec.{field_name}[{index}].id must be non-empty")  # 记录id错误。
        else:  # 在id有效时保存用于覆盖校验。
            ids.append(str(item_id))  # 保存规范条目标识。
        if not _non_empty_text(description):  # 检查说明文本不能为空。
            errors.append(f"experiment_spec.{field_name}[{index}].description must be non-empty")  # 记录说明错误。
    if len(ids) != len(set(ids)):  # 检查同一字段中是否存在重复id。
        errors.append(f"experiment_spec.{field_name} contains duplicate ids")  # 记录重复标识错误。
    return ids  # 返回有效条目标识列表。


def _validate_string_array(value: Any, field_name: str, errors: list[str]) -> None:  # 校验普通非空字符串数组。
    if not isinstance(value, list) or not all(_non_empty_text(item) for item in value):  # 检查数组和每个文本元素。
        errors.append(f"experiment_spec.{field_name} must contain non-empty strings")  # 记录数组合同错误。


def validate_skill_plan(plan: dict[str, Any]) -> list[str]:  # 校验第二API输出的实验规范和Skill执行计划。
    errors: list[str] = []  # 初始化一次返回全部合同错误的列表。
    spec = plan.get("experiment_spec")  # 读取第二API规范化后的实验意图。
    requirement_ids: list[str] = []  # 初始化需要由Skill调用覆盖的规范条目标识。
    if not isinstance(spec, dict):  # 检查实验规范必须存在且为对象。
        errors.append("experiment_spec must be an object")  # 记录规范对象缺失错误。
    else:  # 在规范对象存在时逐项检查。
        if not _non_empty_text(spec.get("objective")):  # 检查本轮工程目标。
            errors.append("experiment_spec.objective must be non-empty")  # 记录目标错误。
        if spec.get("scope") not in ALLOWED_SPEC_SCOPES:  # 检查提案作用域必须对应实验或信息请求。
            errors.append("experiment_spec.scope must be experiment or information_request")  # 记录作用域错误。
        _validate_string_array(spec.get("interventions", []), "interventions", errors)  # 校验准备改变的物理或数值变量。
        _validate_string_array(spec.get("invariants", []), "invariants", errors)  # 校验必须保持不变的条件。
        requirement_ids.extend(_validate_named_items(spec.get("observables", []), "observables", errors))  # 收集直接观测要求。
        requirement_ids.extend(_validate_named_items(spec.get("derivations", []), "derivations", errors))  # 收集派生计算要求。
        requirement_ids.extend(_validate_named_items(spec.get("external_dependencies", []), "external_dependencies", errors))  # 收集外部事实要求。
        if not _non_empty_text(spec.get("acceptance_rule")):  # 检查结果判据。
            errors.append("experiment_spec.acceptance_rule must be non-empty")  # 记录判据错误。
        if spec.get("completion_scope") not in ALLOWED_COMPLETION_SCOPES:  # 检查停止作用域。
            errors.append("experiment_spec.completion_scope is invalid")  # 记录停止作用域错误。
    plan_type = plan.get("plan_type")  # 读取计划类型。
    if plan_type not in ALLOWED_PLAN_TYPES:  # 检查计划类型是否受合同允许。
        errors.append("plan_type must be execute or unsupported")  # 记录计划类型错误。
    calls = plan.get("calls", [])  # 读取有序Skill调用数组。
    if not isinstance(calls, list):  # 检查调用计划必须是数组。
        errors.append("calls must be an array")  # 记录调用数组错误。
        calls = []  # 使用空数组继续其他防御性校验。
    call_ids: list[str] = []  # 初始化调用标识列表。
    covered: list[str] = []  # 初始化全部调用声明覆盖的要求。
    for index, call in enumerate(calls):  # 遍历第二API生成的每个Skill调用。
        if not isinstance(call, dict):  # 检查调用必须是对象。
            errors.append(f"calls[{index}] must be an object")  # 记录调用对象错误。
            continue  # 继续检查其他调用。
        call_id = call.get("call_id")  # 读取调用稳定标识。
        skill_id = call.get("skill_id")  # 读取隐藏Skill注册标识。
        arguments = call.get("arguments")  # 读取调用参数。
        argument_sources = call.get("argument_sources")  # 读取每个参数的证据来源声明。
        covers = call.get("covers")  # 读取该调用覆盖的规范要求。
        depends_on = call.get("depends_on")  # 读取Skill调用依赖边。
        if not _non_empty_text(call_id):  # 检查调用标识不能为空。
            errors.append(f"calls[{index}].call_id must be non-empty")  # 记录调用标识错误。
        else:  # 在调用标识有效时保存。
            call_ids.append(str(call_id))  # 保存调用标识用于唯一性和依赖检查。
        if not _non_empty_text(skill_id):  # 检查Skill标识不能为空。
            errors.append(f"calls[{index}].skill_id must be non-empty")  # 记录Skill标识错误。
        if not isinstance(arguments, dict):  # 检查参数必须是对象。
            errors.append(f"calls[{index}].arguments must be an object")  # 记录参数对象错误。
        if not isinstance(argument_sources, dict):  # 检查参数来源必须是对象。
            errors.append(f"calls[{index}].argument_sources must be an object")  # 记录参数来源错误。
        elif isinstance(arguments, dict) and set(argument_sources) != set(arguments):  # 检查每个参数都有且只有一个来源声明。
            errors.append(f"calls[{index}].argument_sources must match argument keys")  # 记录参数来源键不匹配。
        if not isinstance(covers, list) or not all(_non_empty_text(item) for item in covers):  # 检查覆盖要求必须是非空文本数组。
            errors.append(f"calls[{index}].covers must contain non-empty requirement ids")  # 记录覆盖数组错误。
        else:  # 在覆盖数组有效时累计。
            covered.extend(str(item) for item in covers)  # 保存全部覆盖要求。
        if not isinstance(depends_on, list) or not all(_non_empty_text(item) for item in depends_on):  # 检查依赖数组。
            errors.append(f"calls[{index}].depends_on must contain call ids")  # 记录依赖数组错误。
    if len(call_ids) != len(set(call_ids)):  # 检查调用标识唯一性。
        errors.append("calls contain duplicate call_id values")  # 记录重复调用标识。
    call_id_set = set(call_ids)  # 构造依赖合法性检查所需集合。
    for index, call in enumerate(calls):  # 再次遍历调用检查依赖引用。
        if not isinstance(call, dict):  # 跳过无效调用对象。
            continue  # 继续检查下一项。
        for dependency in call.get("depends_on", []):  # 遍历当前调用依赖。
            if dependency not in call_id_set:  # 检查依赖是否指向实际调用。
                errors.append(f"calls[{index}] depends on unknown call {dependency}")  # 记录未知依赖错误。
            if dependency == call.get("call_id"):  # 检查直接自依赖。
                errors.append(f"calls[{index}] cannot depend on itself")  # 记录自依赖错误。
    uncovered = plan.get("uncovered_requirements", [])  # 读取第二API显式承认未覆盖的要求。
    if not isinstance(uncovered, list) or not all(_non_empty_text(item) for item in uncovered):  # 检查未覆盖要求数组。
        errors.append("uncovered_requirements must contain requirement ids")  # 记录未覆盖数组错误。
        uncovered = []  # 使用空数组继续集合校验。
    requirement_set = set(requirement_ids)  # 构造实验规范全部要求集合。
    declared_set = set(covered) | set(str(item) for item in uncovered)  # 构造已覆盖和未覆盖的完整声明集合。
    if requirement_set != declared_set:  # 检查第二API是否逐项解释全部规范要求。
        errors.append("coverage declarations must exactly match experiment_spec requirements")  # 记录遗漏或凭空添加要求。
    if any(item not in requirement_set for item in covered):  # 检查调用是否覆盖不存在的要求。
        errors.append("calls cover unknown requirement ids")  # 记录虚假覆盖错误。
    fully_preserved = plan.get("proposal_fully_preserved")  # 读取第二API完整保真声明。
    if not isinstance(fully_preserved, bool):  # 检查保真声明必须是布尔值。
        errors.append("proposal_fully_preserved must be boolean")  # 记录保真字段错误。
    if plan_type == "execute":  # 对可执行计划实施严格完成合同。
        if not calls:  # 检查可执行计划至少包含一个Skill调用。
            errors.append("execute plan must contain at least one call")  # 记录空执行计划错误。
        if uncovered:  # 检查可执行计划不能遗留未覆盖要求。
            errors.append("execute plan cannot contain uncovered requirements")  # 记录部分执行错误。
        if fully_preserved is not True:  # 检查可执行计划必须明确完整保留原提案。
            errors.append("execute plan must fully preserve the proposal")  # 记录保真声明错误。
    if plan_type == "unsupported":  # 对不支持结果实施诚实拒绝合同。
        if calls:  # 检查不支持计划不得包含准备执行的调用。
            errors.append("unsupported plan cannot contain calls")  # 记录拒绝与执行并存错误。
        if not _non_empty_text(plan.get("unsupported_reason")):  # 检查拒绝理由不能为空。
            errors.append("unsupported plan requires unsupported_reason")  # 记录拒绝理由错误。
        if fully_preserved is not False:  # 检查不支持时不得声称已完整执行。
            errors.append("unsupported plan must set proposal_fully_preserved to false")  # 记录错误保真声明。
    return errors  # 返回全部Skill计划合同错误。


def freeze_skill_plan(round_dir: Path, plan: dict[str, Any], proposal_hash: str, catalog_hash: str) -> dict[str, Any]:  # 在任何Skill执行前冻结第二API计划。
    errors = validate_skill_plan(plan)  # 按严格合同校验第二API输出。
    if errors:  # 检查计划是否可进入执行阶段。
        raise ValueError("invalid skill plan: " + "; ".join(errors))  # 拒绝结构错误或部分执行计划。
    plan_text = canonical_json(plan)  # 生成稳定计划文本用于哈希。
    plan_hash = sha256_text(plan_text)  # 计算Skill计划内容摘要。
    sealed_at = datetime.now(timezone.utc).isoformat()  # 记录计划冻结UTC时间。
    seal = {"proposal_sha256": proposal_hash, "skill_catalog_sha256": catalog_hash, "skill_plan_sha256": plan_hash, "sealed_at_utc": sealed_at}  # 组织提案、目录和计划绑定封条。
    (round_dir / "skill_plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存第二API原始计划。
    (round_dir / "skill_plan_seal.json").write_text(json.dumps(seal, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存计划完整性封条。
    return seal  # 返回后续执行器使用的冻结收据。


def verify_skill_plan(round_dir: Path, expected_hash: str) -> dict[str, Any]:  # 在Skill执行前重新验证冻结计划未被改写。
    plan = json.loads((round_dir / "skill_plan.json").read_text(encoding="utf-8"))  # 从磁盘重新读取第二API计划。
    actual_hash = sha256_text(canonical_json(plan))  # 重新计算当前计划文件摘要。
    if actual_hash != expected_hash:  # 检查计划冻结后是否被篡改。
        raise RuntimeError("frozen skill plan hash mismatch")  # 拒绝执行被改写的Skill计划。
    return plan  # 返回通过完整性验证的计划对象。

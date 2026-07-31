from __future__ import annotations  # 启用现代类型注解并保持运行环境兼容。

import json  # 把模型提案和公开反馈转换为可检索文本。
from typing import Any  # 表示动态证据、提案和任务状态结构。

GENERIC_DECISION_GATES = (  # 定义不泄露具体工程路线的顶层完成门。
    {"id": "mesh_strategy_evidence", "description": "已有证据足以判断是否还需要继续改变离散或网格策略。"},  # 要求回答是否继续加密及其依据。
    {"id": "decision_quantity_evidence", "description": "至少一个适合原始工程决策的评价量已经得到可比较的数值证据，不能只依赖局部峰值或数值平衡。"},  # 要求评价量真正支撑裂纹判断但不指定具体量。
    {"id": "model_scope_assessment", "description": "当前物理模型的适用范围、缺失机制和所需外部数据已经被明确评估。"},  # 要求判断线性或其他模型假设是否足够。
    {"id": "engineering_answer", "description": "最终答复分别说明是否继续细化、当前模型能否用于判断以及下一步具体行动。"},  # 要求闭合用户提出的三项工程问题。
)  # 完成通用任务门定义。

_MESH_EVIDENCE_OPERATIONS = {"refine", "refine_to_explicit_target_size", "refine_and_fracture_parameter"}  # 定义能够产生新增离散证据的内部操作。
_DECISION_QUANTITY_OPERATIONS = {"fracture_parameter_sequence", "refine_and_fracture_parameter", "fixed_probe", "region_average"}  # 定义能够验证替代评价量的内部操作。
_MATERIAL_TERMS = ("材料", "塑性", "屈服", "硬化", "本构", "断裂韧性", "material", "plastic", "yield", "hardening", "constitutive", "toughness")  # 定义模型适用性和材料外部数据语义。
_FINAL_ANSWER_FIELDS = ("continue_refinement", "model_usability", "next_action")  # 定义最终工程答复必须覆盖的三个字段。


def _text(value: Any) -> str:  # 把任意JSON结构转换为稳定小写检索文本。
    return json.dumps(value, ensure_ascii=False, sort_keys=True).lower()  # 使用完整自然语言内容进行通用语义检查。


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:  # 检查文本是否包含任一中英文语义词。
    return any(term in text for term in terms)  # 任一词出现时返回真。


def _completed_operations(audit_rounds: list[dict[str, Any]]) -> list[str]:  # 提取已经产生成功公开证据的内部操作。
    operations: list[str] = []  # 初始化操作序列。
    for item in audit_rounds:  # 逐轮读取审计摘要。
        feedback = item.get("public_feedback", {})  # 读取该轮公开执行结果。
        mapping = item.get("internal_mapping", {})  # 读取模型不可见的映射收据。
        if feedback.get("status") == "completed":  # 只接受实际完成的计算或后处理。
            operations.append(str(mapping.get("operation", "")))  # 保存内部操作标识供任务控制器判断证据类型。
    return operations  # 返回已完成操作列表。


def _material_facts_available(initial_evidence: dict[str, Any]) -> bool:  # 判断初始模型是否已经提供足够材料非线性事实。
    material = initial_evidence.get("model_facts", {}).get("material", {})  # 读取公开材料事实。
    plastic_curve = material.get("plastic_curve")  # 读取塑性曲线。
    yield_strength = material.get("yield_strength_mpa")  # 读取可选屈服强度。
    return bool(plastic_curve) and isinstance(yield_strength, (int, float))  # 同时具备屈服点和塑性曲线时认为材料范围可直接评估。


def _external_blockers(public_history: list[dict[str, Any]]) -> list[dict[str, Any]]:  # 收集隐藏执行器无法自行补齐的外部信息请求。
    blockers: list[dict[str, Any]] = []  # 初始化外部阻塞列表。
    for item in public_history:  # 逐轮检查公开历史。
        feedback = item.get("execution_feedback", {})  # 读取模型可见执行反馈。
        if feedback.get("status") != "information_required":  # 跳过已经执行或只是路线切换的轮次。
            continue  # 继续检查下一轮。
        observations = feedback.get("observations", {})  # 读取具体缺失信息。
        requested = observations.get("requested_information", [])  # 读取标准化请求数组。
        if not isinstance(requested, list):  # 兼容单个请求值。
            requested = [requested]  # 把单值转为数组。
        blockers.append({"round": item.get("round"), "requested_information": [str(value) for value in requested if str(value).strip()]})  # 保存脱敏轮次和外部事实请求。
    return blockers  # 返回全部真实阻塞项。


def _material_scope_requested(public_history: list[dict[str, Any]]) -> bool:  # 判断模型是否已经明确触发材料或模型范围评估。
    for item in public_history:  # 逐轮检查模型原始提案和公开反馈。
        proposal = item.get("proposal", {})  # 读取冻结后的模型原始提案。
        feedback = item.get("execution_feedback", {})  # 读取执行层反馈。
        if feedback.get("status") == "information_required" and _contains_any(_text(proposal), _MATERIAL_TERMS):  # 检查是否因材料或本构事实不足而阻塞。
            return True  # 认为模型已经识别并正式处理适用性边界。
    return False  # 未发现正式材料范围请求。


def _valid_final_answer(proposal: dict[str, Any] | None) -> bool:  # 检查任务完成提案是否覆盖用户要求的三项工程答复。
    if not isinstance(proposal, dict) or proposal.get("proposal_type") != "resolve_task":  # 只接受明确的任务级完成提案。
        return False  # 其他提案不能关闭总体任务。
    final_answer = proposal.get("final_answer")  # 读取结构化最终答复。
    if not isinstance(final_answer, dict):  # 检查最终答复必须是对象。
        return False  # 缺少结构化答复时拒绝完成。
    for field in _FINAL_ANSWER_FIELDS:  # 遍历三个必需答复字段。
        value = final_answer.get(field)  # 读取当前答复内容。
        if not isinstance(value, str) or not value.strip():  # 检查答复必须是非空文本。
            return False  # 任一问题未回答时拒绝完成。
    remaining = final_answer.get("remaining_uncertainties", [])  # 读取仍需保留的不确定性。
    return isinstance(remaining, list) and all(isinstance(item, str) and item.strip() for item in remaining)  # 只接受可审计的不确定性数组。


def assess_task_state(initial_evidence: dict[str, Any], public_history: list[dict[str, Any]], audit_rounds: list[dict[str, Any]], resolution_proposal: dict[str, Any] | None = None) -> dict[str, Any]:  # 根据已冻结证据评估总体任务完成度。
    operations = _completed_operations(audit_rounds)  # 获取已经真实完成的隐藏执行操作。
    mesh_supported = any(operation in _MESH_EVIDENCE_OPERATIONS for operation in operations)  # 检查是否已经产生新增网格或离散证据。
    decision_quantity_supported = any(operation in _DECISION_QUANTITY_OPERATIONS for operation in operations)  # 检查是否已经验证适合工程决策的评价量。
    blockers = _external_blockers(public_history)  # 收集当前真实外部数据阻塞。
    model_scope_assessed = _material_facts_available(initial_evidence) or _material_scope_requested(public_history) or _valid_final_answer(resolution_proposal)  # 接受已有材料事实、正式外部请求或结构化条件性答复作为模型范围评估。
    engineering_answer_complete = _valid_final_answer(resolution_proposal)  # 检查是否已经提交完整任务级答复。
    gate_values = {  # 组织四个顶层决策门状态。
        "mesh_strategy_evidence": mesh_supported,  # 保存网格策略证据状态。
        "decision_quantity_evidence": decision_quantity_supported,  # 保存工程评价量证据状态。
        "model_scope_assessment": model_scope_assessed,  # 保存模型适用性评估状态。
        "engineering_answer": engineering_answer_complete,  # 保存最终工程答复状态。
    }  # 完成任务门状态对象。
    resolved = [gate["id"] for gate in GENERIC_DECISION_GATES if gate_values[gate["id"]]]  # 生成已完成决策门列表。
    unresolved = [gate["id"] for gate in GENERIC_DECISION_GATES if not gate_values[gate["id"]]]  # 生成仍需继续处理的决策门列表。
    return {"objective": initial_evidence.get("user_question", "解决原始工程问题"), "gates": gate_values, "resolved_decision_gates": resolved, "unresolved_decision_gates": unresolved, "external_blockers": blockers}  # 返回不含工具目录的总体任务状态。


def adjudicate_resolution(initial_evidence: dict[str, Any], public_history: list[dict[str, Any]], audit_rounds: list[dict[str, Any]], proposal: dict[str, Any]) -> dict[str, Any]:  # 判断模型的任务完成声明是否满足真实证据门。
    state = assess_task_state(initial_evidence, public_history, audit_rounds, proposal)  # 把当前完成提案纳入任务门评估。
    unresolved = list(state["unresolved_decision_gates"])  # 读取仍未完成的决策门。
    if unresolved:  # 检查任务是否仍有未决门。
        return {"status": "resolution_rejected", "task_state": state, "reason": "原始工程问题仍有未决决策门，当前完成声明只结束了部分路线。"}  # 拒绝把局部停止解释为全局完成。
    if state["external_blockers"]:  # 检查是否存在必须由用户补充的外部事实。
        return {"status": "task_blocked", "task_state": state, "reason": "可计算部分已经完成，但最终工程适用性仍受外部数据阻塞。"}  # 用阻塞状态结束而不伪装成完全解决。
    return {"status": "task_resolved", "task_state": state, "reason": "原始工程问题的全部决策门均已由证据和结构化答复关闭。"}  # 接受完整任务解决。


def public_task_packet(state: dict[str, Any]) -> dict[str, Any]:  # 构造每轮模型可见的通用任务控制信息。
    descriptions = {gate["id"]: gate["description"] for gate in GENERIC_DECISION_GATES}  # 建立决策门说明映射。
    return {  # 返回不泄露隐藏工具能力的顶层任务合同。
        "global_objective": state["objective"],  # 重申必须解决的原始用户问题。
        "resolved_decision_gates": [{"id": gate_id, "description": descriptions[gate_id]} for gate_id in state["resolved_decision_gates"]],  # 展示已经关闭的通用决策门。
        "unresolved_decision_gates": [{"id": gate_id, "description": descriptions[gate_id]} for gate_id in state["unresolved_decision_gates"]],  # 展示必须继续解决的通用决策门。
        "external_blockers": state["external_blockers"],  # 展示已经确认的外部事实阻塞。
        "completion_policy": "结束当前方法只代表路线切换；只要仍有未决决策门，就必须继续提出新的工程实验或信息请求。只有resolve_task通过任务门检查，或真实外部数据阻塞被明确记录时，总体任务才可结束。",  # 明确路线停止和任务结束的作用域差异。
    }  # 完成模型可见任务包。

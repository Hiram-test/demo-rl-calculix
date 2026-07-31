#!/usr/bin/env python3  # 使用两个独立API运行隐藏Skill规划与真实执行闭环。
from __future__ import annotations  # 启用现代类型注解并保持Python 3.11兼容。

import argparse  # 解析决策模型、规划模型、轮数和输出目录。
import json  # 保存完整双API审计链和任务状态。
import os  # 读取两个API的受保护凭据和端点配置。
import sys  # 把仓库根目录加入模块搜索路径。
from pathlib import Path  # 管理隔离输出目录和仓库根目录。
from typing import Any  # 表示动态提案、计划、反馈和任务状态。

from openai import OpenAI  # 通过兼容接口调用决策API和Skill规划API。

ROOT = Path(__file__).resolve().parents[1]  # 定位仓库根目录。
sys.path.insert(0, str(ROOT))  # 确保脚本直接执行时可以导入仓库模块。

import scripts.run_deepseek_crack_hidden_executor as decision_runner  # 复用真实第一API调用、初始证据和提案冻结组件。
from experiments.hidden_executor import task_controller  # 复用总体任务决策门和完成裁决。
from experiments.skill_planner.contracts import freeze_skill_plan  # 在任何Skill执行前冻结第二API计划。
from experiments.skill_planner.executor import execute_frozen_plan  # 确定性校验并执行隐藏Skill调用图。
from experiments.skill_planner.planner_api import PLANNER_SYSTEM_PROMPT  # 保存实际第二API系统提示用于审计。
from experiments.skill_planner.planner_api import request_skill_plan  # 请求第二API把冻结提案编译为Skill计划。
from experiments.skill_planner.skills import build_registry  # 构造真实有限元和后处理Skill目录。

DECISION_SYSTEM_PROMPT = (  # 定义第一API只负责工程决策的任务级提示词。
    "你是有限元与结构工程证据分析代理。你的总体目标始终是解决initial_evidence.user_question中的原始工程问题，"  # 固定全局目标。
    "不能因为某一种评价量、网格策略或分析路线达到停止条件就结束整个任务。请区分当前路线和总体任务："  # 区分路线停止和任务停止。
    "当前方法应当结束但原始问题仍有未决决策门时，proposal_type必须为switch_route，并在route_transition中说明route_conclusion、next_route和reason；"  # 要求显式路线转换。
    "只有task_contract中的未决决策门全部关闭，并且你能分别回答是否继续细化、当前模型能否用于判断以及下一步具体行动时，proposal_type才能为resolve_task，"  # 定义总体完成条件。
    "此时final_answer必须包含continue_refinement、model_usability、next_action和remaining_uncertainties。"  # 定义结构化最终工程答复。
    "每轮至少提出两个竞争假设，区分直接观测、可能解释和未知，并选择一个最能推进未决决策门的唯一下一步。"  # 要求目标驱动的可证伪决策。
    "你不知道执行环境拥有哪些软件、Skill或能力，不得猜测能力目录，不得使用函数名、Skill名、工具名或软件命令表达方案；"  # 对第一API完全隐藏Skill目录。
    "请只用工程和物理语言描述改变什么、保持什么不变、测量什么、怎样判断以及何时结束当前实验。"  # 约束自然语言实验设计。
    "证据不足时可以使用request_information请求缺失事实；请求外部事实不会自动关闭仍可计算的决策门。"  # 防止外部阻塞提前终止可计算部分。
    "只输出一个合法JSON对象，公共字段为competing_hypotheses、evidence_refs、uncertainties、proposal_type；"  # 开始定义第一API输出合同。
    "proposal_type只能是experiment、request_information、switch_route或resolve_task，不要使用stop。"  # 移除无作用域停止。
    "当proposal_type=experiment时，experiment必须包含purpose、change、hold_fixed、measure、decision_rule、stop_condition，其中hold_fixed和measure必须是字符串数组；"  # 定义实验字段。
    "当proposal_type=request_information时，information_request必须包含question。"  # 定义信息请求字段。
)  # 完成第一API系统提示词。

DECISION_USER_INSTRUCTION = (  # 定义第一API每轮固定用户指令。
    "请只根据下面的实时证据和task_contract提出唯一下一步。当前路线结束时使用switch_route，不能把它写成总体任务完成；"  # 强调路线结束后继续。
    "只有unresolved_decision_gates为空时才允许resolve_task。提案会先被冻结，再交给独立执行层处理。"  # 说明冻结顺序但不暴露Skill目录。
    "不要猜测执行层内部能力，不要补造未提供的模型事实。\n\n"  # 保持隐藏能力和事实约束。
)  # 完成第一API用户指令。


def _write_public_feedback(round_dir: Path, feedback: dict[str, Any]) -> None:  # 保存第一API下一轮唯一可见的物理或控制反馈。
    (round_dir / "public_feedback.json").write_text(json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 写入独立公开反馈文件。


def _route_feedback(proposal: dict[str, Any], proposal_hash: str) -> dict[str, Any]:  # 把当前路线结束转换为总体任务继续反馈。
    transition = proposal.get("route_transition", {})  # 读取结构化路线结论和下一路线。
    return {"status": "route_transition", "executed_change": "未调用Skill；当前工程方法结束，但原始任务继续", "actual_parameters": {}, "observations": {"route_conclusion": str(transition.get("route_conclusion", "当前路线停止")), "next_route": str(transition.get("next_route", "继续处理未决决策门")), "reason": str(transition.get("reason", "当前路线不足以关闭总体任务")), "task_continues": True}, "limitations": [], "proposal_sha256": proposal_hash}  # 返回不泄露执行目录的路线控制反馈。


def _resolution_feedback(adjudication: dict[str, Any], proposal_hash: str) -> dict[str, Any]:  # 把独立任务门裁决转换为第一API可见反馈。
    state = adjudication["task_state"]  # 读取完成声明对应的任务状态。
    return {"status": adjudication["status"], "executed_change": "未调用Skill；独立任务控制器检查了总体完成条件", "actual_parameters": {}, "observations": {"controller_reason": adjudication["reason"], "resolved_decision_gates": state["resolved_decision_gates"], "unresolved_decision_gates": state["unresolved_decision_gates"], "external_blockers": state["external_blockers"]}, "limitations": [], "proposal_sha256": proposal_hash}  # 返回任务级完成、阻塞或拒绝状态。


def _synthetic_operation(skill_ids: list[str], feedback: dict[str, Any]) -> str:  # 把多Skill结果映射为既有任务控制器可识别的证据类别。
    mesh_skills = {"mesh.refine", "fracture.refine_and_energy"}  # 定义产生新增网格证据的Skill。
    decision_skills = {"fracture.energy_sequence", "fracture.refine_and_energy", "fracture.crack_face_displacement", "postprocess.richardson"}  # 定义产生工程评价量证据的Skill。
    has_mesh = any(skill_id in mesh_skills for skill_id in skill_ids)  # 检查本轮是否产生网格策略证据。
    has_decision = any(skill_id in decision_skills for skill_id in skill_ids)  # 检查本轮是否产生工程评价量证据。
    if has_mesh and has_decision:  # 检查同一执行图是否同时关闭两类证据门。
        return "refine_and_fracture_parameter"  # 复用既有联合操作类别。
    if has_mesh:  # 检查是否仅产生网格证据。
        return "refine"  # 复用既有网格操作类别。
    if has_decision:  # 检查是否仅产生工程评价量证据。
        return "fracture_parameter_sequence"  # 复用既有断裂量类别。
    if feedback.get("status") == "information_required":  # 检查是否只产生外部事实阻塞。
        return "request_material"  # 复用既有外部材料请求类别。
    return "skill_plan_completed"  # 对其他完整执行保留中性内部类别。


def run_experiment(decision_model: str, planner_model: str, max_rounds: int, output_dir: Path) -> dict[str, Any]:  # 运行第一API决策、第二API规划和隐藏Skill执行闭环。
    decision_key = os.environ.get("DEEPSEEK_API_KEY", "")  # 读取第一API受保护凭据。
    planner_key = os.environ.get("SKILL_PLANNER_API_KEY", "") or decision_key  # 读取独立第二API凭据并允许显式回退到同一密钥。
    if not decision_key:  # 检查第一API凭据是否存在。
        raise RuntimeError("DEEPSEEK_API_KEY is required")  # 禁止生成伪造决策链。
    if not planner_key:  # 检查第二API凭据是否存在。
        raise RuntimeError("SKILL_PLANNER_API_KEY or DEEPSEEK_API_KEY is required")  # 禁止用本地关键词替代第二API。
    planner_base_url = os.environ.get("SKILL_PLANNER_BASE_URL", "https://api.deepseek.com")  # 读取可独立配置的第二API端点。
    decision_client = OpenAI(api_key=decision_key, base_url="https://api.deepseek.com")  # 创建第一API工程决策客户端。
    planner_client = OpenAI(api_key=planner_key, base_url=planner_base_url)  # 创建第二API Skill规划客户端。
    output_dir.mkdir(parents=True, exist_ok=True)  # 创建独立双API结果目录。
    registry = build_registry()  # 构造第二API可见、第一API不可见的Skill注册表。
    catalog = registry.catalog()  # 生成稳定排序的Skill目录描述。
    catalog_hash = registry.catalog_hash()  # 计算Skill目录版本摘要。
    (output_dir / "skill_catalog.json").write_text(json.dumps({"catalog_sha256": catalog_hash, "skills": catalog}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存只供内部和第二API审计的目录快照。
    decision_runner.SYSTEM_PROMPT = DECISION_SYSTEM_PROMPT  # 让复用的第一API请求函数使用任务级无Skill提示。
    decision_runner.USER_INSTRUCTION = DECISION_USER_INSTRUCTION  # 设置第一API每轮无Skill指令。
    initial = decision_runner._initial_evidence()  # 获取真实三档网格初始证据。
    public_history: list[dict[str, Any]] = []  # 初始化第一API可见的提案和物理反馈历史。
    audit_rounds: list[dict[str, Any]] = []  # 初始化完整双API内部审计摘要。
    previous_chain_hash = "0" * 64  # 使用固定创世哈希开始第一API提案链。
    task_state = task_controller.assess_task_state(initial, public_history, audit_rounds)  # 初始化总体任务决策门。
    task_status = "in_progress"  # 初始化总体任务状态。
    route_transitions = 0  # 记录当前方法结束并切换路线的次数。
    completed_by_model = False  # 记录第一API是否提交通过任务门的完成声明。
    for round_index in range(1, max_rounds + 1):  # 在固定预算内逐轮运行双API闭环。
        evidence_packet = {"initial_evidence": initial, "task_contract": task_controller.public_task_packet(task_state), "previous_rounds": public_history, "round": round_index, "budget_policy": "预算只限制调用成本，不能把未完成任务标记为已解决。"}  # 构造完全不含Skill目录的第一API证据包。
        proposal, decision_metadata = decision_runner._request_proposal(decision_client, decision_model, evidence_packet)  # 请求第一API唯一工程下一步。
        frozen = decision_runner.freeze_proposal(proposal, output_dir, round_index, previous_chain_hash)  # 在第二API看到提案前先写盘并封存。
        proposal_hash = str(frozen["seal"]["proposal_sha256"])  # 读取冻结提案摘要。
        previous_chain_hash = str(frozen["seal"]["chain_sha256"])  # 更新下一轮提案链前驱。
        round_dir = Path(frozen["round_dir"])  # 定位当前轮独立目录。
        proposal_type = str(proposal.get("proposal_type", ""))  # 读取第一API提案作用域。
        planner_metadata: dict[str, Any] = {}  # 初始化第二API调用元数据。
        skill_plan_summary: dict[str, Any] | None = None  # 初始化Skill计划审计摘要。
        skill_audit: dict[str, Any] | None = None  # 初始化逐Skill执行审计。
        skill_ids: list[str] = []  # 初始化本轮实际规划的Skill标识。
        if proposal_type == "switch_route":  # 检查第一API是否只结束当前工程路线。
            feedback = _route_feedback(proposal, proposal_hash)  # 生成总体任务继续反馈。
            route_transitions += 1  # 记录路线切换次数。
            _write_public_feedback(round_dir, feedback)  # 保存下一轮第一API可见反馈。
            synthetic_operation = "route_transition"  # 设置任务控制器可识别的路线类别。
        elif proposal_type == "resolve_task":  # 检查第一API是否提交总体任务完成声明。
            candidate_round = {"round": round_index, "proposal": proposal, "execution_feedback": {"status": "resolution_candidate"}}  # 临时把完成候选加入公开历史供裁决读取。
            public_history.append(candidate_round)  # 追加完成候选到历史。
            candidate_audit = {"round": round_index, "proposal_sha256": proposal_hash, "chain_sha256": previous_chain_hash, "metadata": decision_metadata, "internal_mapping": {"operation": "task_resolution_candidate"}, "public_feedback": {"status": "resolution_candidate"}}  # 构造裁决所需临时审计。
            audit_rounds.append(candidate_audit)  # 追加完成候选审计。
            adjudication = task_controller.adjudicate_resolution(initial, public_history, audit_rounds, proposal)  # 使用独立决策门检查完成声明。
            feedback = _resolution_feedback(adjudication, proposal_hash)  # 生成最终任务裁决反馈。
            candidate_round["execution_feedback"] = feedback  # 更新公开历史中的候选反馈。
            candidate_audit["public_feedback"] = feedback  # 更新内部审计中的候选反馈。
            _write_public_feedback(round_dir, feedback)  # 保存任务裁决结果。
            task_state = adjudication["task_state"]  # 保存裁决后的任务状态。
            if feedback["status"] in {"task_resolved", "task_blocked"}:  # 检查总体任务是否完成或被真实外部事实阻塞。
                task_status = str(feedback["status"])  # 保存最终任务状态。
                completed_by_model = True  # 记录完成声明通过独立裁决。
                break  # 结束双API闭环。
            continue  # 过早完成被拒绝时进入下一轮。
        else:  # 对实验和信息请求调用独立第二API规划Skill。
            plan, planner_metadata = request_skill_plan(planner_client, planner_model, proposal, evidence_packet, registry)  # 把冻结提案和隐藏目录发送给第二API。
            plan_seal = freeze_skill_plan(round_dir, plan, proposal_hash, catalog_hash)  # 在任何Skill执行前冻结第二API计划。
            feedback, skill_audit = execute_frozen_plan(round_dir, proposal_hash, str(plan_seal["skill_plan_sha256"]), proposal, evidence_packet, registry)  # 确定性校验并执行Skill DAG。
            skill_ids = [str(call.get("skill_id", "")) for call in plan.get("calls", [])]  # 提取内部Skill标识供任务门证据分类。
            skill_plan_summary = {"skill_plan_sha256": plan_seal["skill_plan_sha256"], "skill_catalog_sha256": catalog_hash, "plan_type": plan.get("plan_type"), "experiment_spec": plan.get("experiment_spec"), "call_count": len(plan.get("calls", []))}  # 保存不复制完整计划的轮次摘要。
            synthetic_operation = _synthetic_operation(skill_ids, feedback)  # 把多Skill执行结果映射为任务门证据类别。
        public_round = {"round": round_index, "proposal": proposal, "execution_feedback": feedback}  # 组织下一轮第一API可见历史。
        public_history.append(public_round)  # 追加公开提案和物理反馈。
        audit_round = {"round": round_index, "proposal_sha256": proposal_hash, "chain_sha256": previous_chain_hash, "decision_metadata": decision_metadata, "planner_metadata": planner_metadata, "internal_mapping": {"operation": synthetic_operation, "skill_ids": skill_ids}, "skill_plan_summary": skill_plan_summary, "skill_execution_audit_file": "skill_execution_audit.json" if skill_audit is not None else None, "public_feedback": feedback}  # 保存完整双API轮次审计摘要。
        audit_rounds.append(audit_round)  # 追加内部审计供任务控制器和最终结果使用。
        task_state = task_controller.assess_task_state(initial, public_history, audit_rounds)  # 根据新物理证据更新总体任务门。
    if task_status == "in_progress":  # 检查是否因预算离开循环。
        task_status = "inconclusive_budget_exhausted"  # 明确标记任务仍未完成。
    result = {"experiment": "deepseek_crack_hidden_skill_planner", "decision_model": decision_model, "planner_model": planner_model, "planner_base_url": planner_base_url, "decision_system_prompt": DECISION_SYSTEM_PROMPT, "planner_system_prompt": PLANNER_SYSTEM_PROMPT, "skill_catalog_visible_to_decision_model": False, "skill_catalog_visible_to_planner_model": True, "skill_plan_visible_to_decision_model": False, "skill_catalog_sha256": catalog_hash, "rounds": audit_rounds, "public_history": public_history, "task_status": task_status, "task_state": task_state, "route_transitions": route_transitions, "completed_by_model": completed_by_model, "final_chain_sha256": previous_chain_hash}  # 组织完整双API实验记录。
    (output_dir / "experiment_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存完整决策链、计划摘要和任务状态。
    print(json.dumps({"status": task_status, "rounds": len(audit_rounds), "route_transitions": route_transitions, "completed_by_model": completed_by_model, "output": str(output_dir / "experiment_result.json")}, ensure_ascii=False))  # 向Actions输出紧凑摘要。
    return result  # 返回完整实验结果供测试或其他隔离入口使用。


def main() -> int:  # 定义命令行入口。
    parser = argparse.ArgumentParser()  # 创建参数解析器。
    parser.add_argument("--decision-model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"))  # 允许覆盖第一API模型。
    parser.add_argument("--planner-model", default=os.environ.get("SKILL_PLANNER_MODEL", "deepseek-v4-pro"))  # 允许独立配置第二API模型。
    parser.add_argument("--max-rounds", type=int, default=8)  # 设置双API闭环最大轮数。
    parser.add_argument("--output", type=Path, default=Path("artifacts/deepseek_crack_hidden_skill_planner"))  # 设置独立结果目录。
    args = parser.parse_args()  # 解析命令行参数。
    if args.max_rounds < 1 or args.max_rounds > 10:  # 限制真实API和有限元成本范围。
        raise ValueError("max-rounds must lie between 1 and 10")  # 拒绝异常预算。
    run_experiment(args.decision_model, args.planner_model, args.max_rounds, args.output)  # 执行完整双API实验。
    return 0  # 返回成功退出码。


if __name__ == "__main__":  # 仅在脚本直接执行时启动实验。
    raise SystemExit(main())  # 把主函数退出码传给操作系统。

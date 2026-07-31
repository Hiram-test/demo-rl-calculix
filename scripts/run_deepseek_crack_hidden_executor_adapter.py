#!/usr/bin/env python3  # 使用任务级控制器启动隔离隐藏执行器实验。
from __future__ import annotations  # 启用现代类型注解。

import json  # 保存适配后的完整实验结果和任务裁决。
import os  # 读取受保护DeepSeek凭据。
import sys  # 把仓库根目录加入模块搜索路径。
from pathlib import Path  # 管理隔离输出目录和仓库根目录。
from typing import Any  # 表示动态任务状态和公开反馈结构。

ROOT = Path(__file__).resolve().parents[1]  # 定位仓库根目录。
sys.path.insert(0, str(ROOT))  # 确保脚本直接执行时可以导入仓库内模块。

import scripts.run_deepseek_crack_hidden_executor as runner  # 导入真实模型调用和初始有限元证据组件。
from experiments.hidden_executor import executor_adapter_v5 as executor_adapter  # 导入路线级停止和任务级完成分离的隐藏执行器。
from experiments.hidden_executor import task_controller  # 导入不泄露工具能力的总体任务决策门控制器。

runner.SYSTEM_PROMPT = (  # 定义隐藏工具实验的任务级系统提示词。
    "你是有限元与结构工程证据分析代理。你的总体目标始终是解决initial_evidence.user_question中的原始工程问题，"  # 固定全局目标而不指定具体技术路线。
    "不能因为某一种评价量、网格策略或分析路线达到停止条件就结束整个任务。请区分当前路线和总体任务："  # 明确路线停止和任务停止的作用域。
    "当当前方法应当结束但原始问题仍有未决决策门时，proposal_type必须为switch_route，并在route_transition中说明"  # 要求显式切换路线。
    "route_conclusion、next_route和reason；随后继续提出新的工程实验或信息请求。只有task_contract中的未决决策门已经全部关闭，"  # 约束任务完成前必须继续闭环。
    "并且你能够分别回答是否继续细化、当前模型能否用于判断以及下一步具体行动时，proposal_type才能为resolve_task，"  # 定义任务级结束条件。
    "此时final_answer必须包含continue_refinement、model_usability、next_action和remaining_uncertainties。"  # 定义结构化最终答复。
    "每轮至少提出两个竞争假设，区分直接观测、可能解释和未知，并选择一个最能推进未决决策门的唯一下一步。"  # 要求形成可证伪且目标驱动的下一步。
    "你不知道执行环境拥有哪些软件或能力，不得假设存在现成工具，不得使用函数名、工具名或软件命令表达方案；"  # 保持模型对隐藏工具目录完全不可见。
    "请只用工程和物理语言描述改变什么、保持什么不变、测量什么、怎样判断以及何时结束当前实验。"  # 约束自然语言实验设计。
    "证据不足时可以使用request_information请求缺失事实；请求外部事实不会自动关闭其他仍可计算的决策门。"  # 防止一个阻塞项提前结束全部任务。
    "只输出一个合法JSON对象，公共字段为competing_hypotheses、evidence_refs、uncertainties、proposal_type；"  # 开始定义输出合同。
    "proposal_type只能是experiment、request_information、switch_route或resolve_task，不要使用stop。"  # 移除无作用域stop。
    "当proposal_type=experiment时，experiment必须包含purpose、change、hold_fixed、measure、decision_rule、stop_condition，"  # 定义控制实验字段。
    "其中hold_fixed和measure必须是字符串数组；当proposal_type=request_information时，information_request必须包含question。"  # 定义信息请求字段。
)  # 完成任务级系统提示词。

runner.USER_INSTRUCTION = (  # 定义每轮固定用户指令。
    "请只根据下面的实时证据和task_contract提出唯一下一步。当前路线结束时使用switch_route，不能把它写成总体任务完成；"  # 强调路线结束后继续。
    "只有unresolved_decision_gates为空时才允许resolve_task。提案会先被冻结，再由独立执行层判断能否忠实执行。"  # 说明冻结顺序和独立完成门。
    "不要猜测执行器内部能力，不要补造未提供的模型事实。\n\n"  # 保持隐藏工具隔离和事实约束。
)  # 完成每轮用户指令。


def _rewrite_public_feedback(round_dir: Path, feedback: dict[str, Any]) -> None:  # 把任务控制器最终裁决同步到公开反馈和内部审计。
    (round_dir / "public_feedback.json").write_text(json.dumps(feedback, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 更新下一轮模型可见反馈。
    audit_path = round_dir / "execution_audit.json"  # 定位该轮内部审计文件。
    audit = json.loads(audit_path.read_text(encoding="utf-8"))  # 读取执行器原始审计记录。
    audit["public_feedback"] = feedback  # 让审计记录引用最终任务裁决反馈。
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存更新后的内部审计。


def _resolution_feedback(candidate: dict[str, Any], adjudication: dict[str, Any], proposal_hash: str) -> dict[str, Any]:  # 把独立任务门裁决转换为模型可见状态。
    status = str(adjudication["status"])  # 读取任务级裁决状态。
    task_state = adjudication["task_state"]  # 读取裁决后的任务门状态。
    feedback = dict(candidate)  # 复制执行器候选反馈以避免改写原对象。
    feedback["status"] = status  # 写入task_resolved、task_blocked或resolution_rejected。
    feedback["executed_change"] = "未运行新模型；独立任务控制器已经检查总体完成条件。"  # 说明本轮只进行任务级裁决。
    feedback["observations"] = {"controller_reason": adjudication["reason"], "resolved_decision_gates": task_state["resolved_decision_gates"], "unresolved_decision_gates": task_state["unresolved_decision_gates"], "external_blockers": task_state["external_blockers"]}  # 返回不含隐藏工具名称的完成门结果。
    feedback["proposal_sha256"] = proposal_hash  # 保留冻结提案摘要以便审计。
    return feedback  # 返回最终任务裁决反馈。


def run_experiment(model: str, max_rounds: int, output_dir: Path) -> dict[str, Any]:  # 运行任务级闭环直到解决、阻塞或预算耗尽。
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")  # 从隔离工作流环境读取真实模型凭据。
    if not api_key:  # 检查凭据是否存在。
        raise RuntimeError("DEEPSEEK_API_KEY is required for live hidden-executor discovery")  # 禁止生成伪造轨迹。
    client = runner.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")  # 创建DeepSeek API客户端。
    output_dir.mkdir(parents=True, exist_ok=True)  # 创建独立结果目录。
    initial = runner._initial_evidence()  # 获取真实三档网格证据且不提供工具目录。
    public_history: list[dict[str, Any]] = []  # 初始化只包含模型提案和公开物理反馈的历史。
    audit_rounds: list[dict[str, Any]] = []  # 初始化完整内部审计摘要。
    previous_chain_hash = "0" * 64  # 使用固定创世哈希开始提案链。
    task_state = task_controller.assess_task_state(initial, public_history, audit_rounds)  # 初始化原始问题的通用决策门。
    task_status = "in_progress"  # 初始化总体任务状态。
    route_transitions = 0  # 记录模型主动结束当前方法并切换路线的次数。
    completed_by_model = False  # 记录模型是否提交并通过任务级完成声明。
    for round_index in range(1, max_rounds + 1):  # 在固定预算内逐轮运行隐藏工具闭环。
        evidence_packet = {"initial_evidence": initial, "task_contract": task_controller.public_task_packet(task_state), "previous_rounds": public_history, "round": round_index, "budget_policy": "轮次预算只限制实验成本，不能把未完成任务标记为已解决；预算耗尽时系统会记录inconclusive_budget_exhausted。"}  # 构造不含工具目录和剩余轮次诱导的任务证据包。
        proposal, metadata = runner._request_proposal(client, model, evidence_packet)  # 请求唯一自然语言工程实验或任务控制提案。
        frozen = runner.freeze_proposal(proposal, output_dir, round_index, previous_chain_hash)  # 在任何映射前冻结提案并写入哈希链。
        proposal_hash = str(frozen["seal"]["proposal_sha256"])  # 读取当前提案摘要。
        previous_chain_hash = str(frozen["seal"]["chain_sha256"])  # 更新下一轮链前驱。
        round_dir = Path(frozen["round_dir"])  # 定位当前冻结轮次目录。
        mapping = executor_adapter.map_frozen_proposal(round_dir, proposal_hash)  # 在模型不可见层忠实映射冻结提案。
        feedback = executor_adapter.execute_mapping(round_dir, proposal_hash, mapping)  # 执行真实求解、路线切换或完成候选。
        public_round = {"round": round_index, "proposal": proposal, "execution_feedback": feedback}  # 组织下一轮模型可见历史。
        public_history.append(public_round)  # 追加公开证据历史。
        audit_round = {"round": round_index, "proposal_sha256": proposal_hash, "chain_sha256": previous_chain_hash, "metadata": metadata, "internal_mapping": mapping, "public_feedback": feedback}  # 组织当前内部审计摘要。
        audit_rounds.append(audit_round)  # 保存当前审计摘要供任务门评估。
        if feedback.get("status") == "route_transition":  # 检查模型是否只结束当前方法。
            route_transitions += 1  # 记录路线切换但继续总体任务。
        if feedback.get("status") == "resolution_candidate":  # 检查模型是否声称原始工程问题已经完成。
            adjudication = task_controller.adjudicate_resolution(initial, public_history, audit_rounds, proposal)  # 使用独立决策门检查完成声明。
            feedback = _resolution_feedback(feedback, adjudication, proposal_hash)  # 生成最终任务级公开反馈。
            public_round["execution_feedback"] = feedback  # 更新模型可见历史中的本轮反馈。
            audit_round["public_feedback"] = feedback  # 更新内存审计摘要中的本轮反馈。
            _rewrite_public_feedback(round_dir, feedback)  # 把最终裁决同步写入冻结结果目录。
            task_state = adjudication["task_state"]  # 保存裁决后的任务门状态。
            if feedback["status"] in {"task_resolved", "task_blocked"}:  # 检查总体任务是否真正结束或被外部数据阻塞。
                task_status = str(feedback["status"])  # 保存最终任务状态。
                completed_by_model = True  # 记录模型完成声明已经通过独立控制器。
                break  # 结束总体闭环。
            continue  # 完成声明被拒绝时进入下一轮而不执行全局停止。
        task_state = task_controller.assess_task_state(initial, public_history, audit_rounds)  # 根据新物理证据更新总体决策门。
    if task_status == "in_progress":  # 检查是否因轮次预算离开循环。
        task_status = "inconclusive_budget_exhausted"  # 明确标记预算耗尽且任务未完成。
    result = {"experiment": "deepseek_crack_hidden_executor", "model": model, "system_prompt": runner.SYSTEM_PROMPT, "tool_catalog_visible_to_model": False, "internal_mapping_visible_to_model": False, "rounds": audit_rounds, "public_history": public_history, "task_status": task_status, "task_state": task_state, "route_transitions": route_transitions, "completed_by_model": completed_by_model, "stopped_voluntarily": completed_by_model, "final_chain_sha256": previous_chain_hash}  # 组织包含任务级状态的最终实验记录。
    (output_dir / "experiment_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存完整结果和审计链。
    print(json.dumps({"status": task_status, "rounds": len(audit_rounds), "route_transitions": route_transitions, "completed_by_model": completed_by_model, "output": str(output_dir / "experiment_result.json")}, ensure_ascii=False))  # 输出紧凑Actions摘要。
    return result  # 返回完整实验结果。


runner.run_experiment = run_experiment  # 让主命令行入口使用任务级多轮实现。


if __name__ == "__main__":  # 仅在脚本直接执行时启动实验。
    raise SystemExit(runner.main())  # 复用主运行器的参数解析和退出码逻辑。

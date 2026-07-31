#!/usr/bin/env python3  # 使用仓库 Python 环境运行隔离的圆轴网格发现实验。
from __future__ import annotations  # 启用现代类型注解并保持脚本兼容。

import argparse  # 解析模型名称、轮数和隔离输出目录。
import json  # 构造模型证据包并保存最终审计结果。
import os  # 从 GitHub Environment 安全读取 DeepSeek 凭据。
import sys  # 把仓库根目录加入模块搜索路径。
from pathlib import Path  # 管理实验输出目录和文件。
from typing import Any  # 标注模型返回的动态 JSON 数据。

from openai import OpenAI  # 通过兼容接口调用真实 DeepSeek 模型。

ROOT = Path(__file__).resolve().parents[1]  # 定位仓库根目录。
sys.path.insert(0, str(ROOT))  # 允许导入隔离实验包。

from experiments.hidden_executor.contracts import freeze_proposal  # 复用 PR 十八的提案冻结和哈希链。
from experiments.hidden_executor.contracts import validate_proposal  # 复用无工具提案结构校验。
from experiments.shaft_grid.experiment import build_initial_evidence  # 导入真实初始网格证据生成器。
from experiments.shaft_grid.experiment import execute_mapping  # 导入隐藏执行与公开反馈生成器。
from experiments.shaft_grid.experiment import map_frozen_proposal  # 导入冻结提案确定性映射器。
from experiments.shaft_grid.experiment import score_experiment  # 导入隐藏解析与盲测评分器。

SYSTEM_PROMPT = (  # 定义不泄露竞赛出处、答案或执行工具的工程代理提示。
    "你是一名负责给出可落地结果的结构与有限元工程师。用户只关心最后该把表面短线画成多少度，"  # 固定最终工程决策。
    "以及这个角度是否会被网格误导。请根据当前证据自行建立合适的物理量和设计变量，"  # 要求模型自己完成建模而不提示变量名称。
    "每轮只提出一个最值得做的下一步检查或计算。你不知道后台有哪些软件和能力，"  # 保持 PR 十八的无工具可见边界。
    "不得写函数名、工具名或软件命令，只能用工程语言说明想改变什么、保持什么不变、看什么结果、"  # 禁止猜测隐藏执行器实现。
    "怎样判断。不要假设某种网格方向一定更好，也不要把单次最大值直接当成可靠结论。"  # 防止把用户直觉写成预设答案。
    "证据够用时应直接停止并给出明确角度和网格建议。只输出一个合法 JSON 对象，"  # 要求最终可用结果并限制输出格式。
    "字段必须为 competing_hypotheses、evidence_refs、uncertainties、proposal_type、experiment、"  # 开始定义通用冻结提案合同。
    "information_request、provisional_answer。proposal_type 只能是 experiment、request_information 或 stop。"  # 限制提案类型。
    "至少给出两个竞争解释。当 proposal_type=experiment 时，experiment 必须包含 purpose、change、"  # 要求可证伪但不要求用户本人科学化表达。
    "hold_fixed、measure、decision_rule、stop_condition，其中 hold_fixed 和 measure 是字符串数组。"  # 定义唯一下一步实验结构。
    "当 proposal_type=request_information 时，information_request 必须包含 question。"  # 定义信息请求结构。
)  # 完成系统提示。

USER_INSTRUCTION = (  # 定义每轮固定证据请求说明。
    "下面是工程师的原始问题、已知条件、初步计算和前面实际做过的检查。"  # 把上下文保持为工程任务。
    "请只选择一个下一步动作；提案会先原样冻结，再由独立执行层判断能否忠实执行。\n\n"  # 说明冻结顺序但不泄露能力目录。
)  # 完成每轮用户指令。


def _usage_dict(usage: Any) -> dict[str, Any]:  # 把 SDK 使用量对象转换为可序列化结构。
    if usage is None:  # 检查响应是否缺少使用量信息。
        return {}  # 在缺失时返回空对象。
    if hasattr(usage, "model_dump"):  # 优先使用新版 SDK 序列化方法。
        return usage.model_dump()  # 返回完整使用量字段。
    return {name: getattr(usage, name) for name in ("prompt_tokens", "completion_tokens", "total_tokens") if getattr(usage, name, None) is not None}  # 兼容旧版 SDK 字段。


def _request_proposal(client: OpenAI, model: str, evidence_packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:  # 请求一份无工具可见的唯一工程提案。
    packet_text = json.dumps(evidence_packet, ensure_ascii=False, indent=2)  # 把实时证据序列化为模型可读文本。
    last_error: Exception | None = None  # 保存最近一次格式错误用于最终诊断。
    for attempt in range(1, 3):  # 最多允许一次只修复 JSON 结构的重试。
        repair = "" if attempt == 1 else "上一次 JSON 结构无效，请保持工程目的不变并重新输出完整合法 JSON。\n\n"  # 仅提示结构修复而不暗示物理路线。
        response = client.chat.completions.create(  # 发起真实 DeepSeek 请求。
            model=model,  # 使用命令行或环境变量指定的真实模型。
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": repair + USER_INSTRUCTION + packet_text}],  # 每轮只发送当前证据和公开历史。
            response_format={"type": "json_object"},  # 要求 API 返回单个合法 JSON 对象。
            max_tokens=10000,  # 为完整工程判断保留足够公开输出预算。
            reasoning_effort="high",  # 请求高强度推理模式。
            extra_body={"thinking": {"type": "enabled"}},  # 显式启用模型思考但不保存隐藏推理。
            stream=False,  # 使用单次响应便于完整冻结提案。
        )  # 完成模型调用。
        content = response.choices[0].message.content  # 读取模型公开 JSON 文本。
        if not content or not content.strip():  # 检查模型是否返回空内容。
            last_error = RuntimeError("DeepSeek returned an empty proposal")  # 保存空输出错误。
            continue  # 进入结构修复重试。
        try:  # 尝试解析模型 JSON 输出。
            proposal = json.loads(content)  # 把公开文本转换为提案对象。
        except json.JSONDecodeError as exc:  # 捕获 JSON 语法错误。
            last_error = exc  # 保存解析异常。
            continue  # 进入下一次请求。
        errors = validate_proposal(proposal)  # 使用现有无工具合同校验提案。
        if errors:  # 检查合同是否存在字段错误。
            last_error = RuntimeError("invalid proposal: " + "; ".join(errors))  # 保存全部结构错误。
            continue  # 允许模型只修复结构一次。
        metadata = {"requested_model": model, "response_model": response.model, "finish_reason": response.choices[0].finish_reason, "usage": _usage_dict(response.usage)}  # 记录真实模型和令牌使用量。
        return proposal, metadata  # 返回通过校验的冻结候选提案。
    raise RuntimeError(f"DeepSeek did not return a valid proposal after two attempts: {last_error}")  # 在两次失败后诚实终止。


def run_experiment(model: str, max_rounds: int, output_dir: Path) -> dict[str, Any]:  # 执行完整多轮建模、冻结、隐藏执行和盲评流程。
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")  # 从受保护环境读取真实 DeepSeek 凭据。
    if not api_key:  # 检查凭据是否可用。
        raise RuntimeError("DEEPSEEK_API_KEY is required for the live shaft-grid experiment")  # 禁止伪造模型输出。
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")  # 创建 DeepSeek 兼容 API 客户端。
    output_dir.mkdir(parents=True, exist_ok=True)  # 创建本实验独立输出目录。
    initial = build_initial_evidence(output_dir)  # 执行三组真实 CalculiX 初始网格计算。
    public_history: list[dict[str, Any]] = []  # 初始化模型可见的提案和物理反馈历史。
    audit_rounds: list[dict[str, Any]] = []  # 初始化完整隐藏审计轨迹。
    previous_chain_hash = "0" * 64  # 使用固定创世哈希开始冻结提案链。
    stopped_voluntarily = False  # 记录模型是否主动给出最终工程建议。
    for round_index in range(1, max_rounds + 1):  # 在隐藏固定轮数上限内逐轮运行。
        evidence_packet = {"initial_evidence": initial, "previous_rounds": public_history, "round": round_index, "remaining_rounds": max_rounds - round_index + 1}  # 构造不含工具目录和解析答案的模型证据包。
        proposal, metadata = _request_proposal(client, model, evidence_packet)  # 让模型自行提出唯一下一步工程动作。
        frozen = freeze_proposal(proposal, output_dir, round_index, previous_chain_hash)  # 在任何映射前把提案原样写盘并封存。
        proposal_hash = str(frozen["seal"]["proposal_sha256"])  # 读取当前冻结提案摘要。
        previous_chain_hash = str(frozen["seal"]["chain_sha256"])  # 更新下一轮哈希链前驱。
        round_dir = Path(frozen["round_dir"])  # 定位当前轮独立审计目录。
        mapping = map_frozen_proposal(round_dir, proposal_hash)  # 在模型不可见层忠实映射工程提案。
        feedback = execute_mapping(output_dir, round_dir, proposal_hash, mapping)  # 执行真实求解、后处理或诚实拒绝。
        public_round = {"round": round_index, "proposal": proposal, "execution_feedback": feedback}  # 组织下一轮模型允许看到的公开证据。
        public_history.append(public_round)  # 把当前公开结果加入后续上下文。
        audit_rounds.append({"round": round_index, "proposal_sha256": proposal_hash, "chain_sha256": previous_chain_hash, "metadata": metadata, "internal_mapping": mapping, "public_feedback": feedback})  # 保存完整审计摘要。
        if feedback.get("status") == "finished":  # 检查模型是否主动停止并形成建议。
            stopped_voluntarily = True  # 标记自主停止行为。
            break  # 结束公开决策循环。
    score = score_experiment(output_dir, public_history)  # 在循环冻结后执行隐藏解析和跨工况盲评。
    result = {"experiment": "deepseek_shaft_grid_discovery", "model": model, "system_prompt": SYSTEM_PROMPT, "user_question": initial["user_question"], "tool_catalog_visible_to_model": False, "competition_source_visible_to_model": False, "analytical_truth_visible_initially": False, "rounds": audit_rounds, "public_history": public_history, "stopped_voluntarily": stopped_voluntarily, "final_chain_sha256": previous_chain_hash, "hidden_score": score}  # 组织完整实验结果。
    (output_dir / "experiment_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存完整结果和隐藏评分。
    print(json.dumps({"status": "completed", "rounds": len(audit_rounds), "stopped_voluntarily": stopped_voluntarily, "overall_pass": score["overall_pass"], "output": str(output_dir / "experiment_result.json")}, ensure_ascii=False))  # 向 Actions 日志输出紧凑状态。
    return result  # 返回结果供测试或其他隔离入口使用。


def main() -> int:  # 定义命令行入口。
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器。
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"))  # 允许覆盖真实 DeepSeek 模型名称。
    parser.add_argument("--max-rounds", type=int, default=6)  # 设置最多六轮公开工程决策。
    parser.add_argument("--output", type=Path, default=Path("artifacts/deepseek_shaft_grid_discovery"))  # 设置完全隔离的输出目录。
    args = parser.parse_args()  # 解析命令行参数。
    if args.max_rounds < 1 or args.max_rounds > 8:  # 限制真实 API 和求解循环规模。
        raise ValueError("max-rounds must lie between one and eight")  # 拒绝异常轮数。
    run_experiment(args.model, args.max_rounds, args.output)  # 执行完整实验。
    return 0  # 返回成功退出码。


if __name__ == "__main__":  # 仅在直接运行脚本时启动实验。
    raise SystemExit(main())  # 把主函数退出码传给操作系统。

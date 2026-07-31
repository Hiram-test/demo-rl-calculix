#!/usr/bin/env python3  # 使用仓库 Python 环境运行隔离的无工具可见实验。
from __future__ import annotations  # 启用现代类型注解并保持兼容。

import argparse  # 解析模型名称、轮数和输出目录。
import json  # 构造模型证据包并保存完整轨迹。
import os  # 从 GitHub Environment 安全读取 DeepSeek 凭据。
import sys  # 把仓库根目录加入模块搜索路径。
from pathlib import Path  # 管理隔离实验文件和目录。
from typing import Any  # 表示模型生成的动态 JSON 数据。

from openai import OpenAI  # 通过兼容接口调用 DeepSeek 模型。

ROOT = Path(__file__).resolve().parents[1]  # 定位仓库根目录。
sys.path.insert(0, str(ROOT))  # 允许导入新建的隔离执行器包。

from experiments.hidden_executor.contracts import freeze_proposal  # 在执行前冻结模型实验提案。
from experiments.hidden_executor.contracts import validate_proposal  # 校验无工具提案合同。
from experiments.hidden_executor.executor import _load_backend  # 只读加载既有裂纹有限元证据后端。
from experiments.hidden_executor.executor import execute_mapping  # 执行隐藏映射并生成脱敏反馈。
from experiments.hidden_executor.executor import map_frozen_proposal  # 在模型不可见层映射冻结提案。

SYSTEM_PROMPT = (  # 定义只包含工程判断纪律的系统提示词。
    "你是有限元与结构工程证据分析代理。请先固定用户需要作出的工程决策、评价位置和可接受的不确定性，"  # 要求明确工程决策而不指定任何裂纹路线。
    "再区分直接观测、可能解释和未知。每轮至少提出两个竞争假设，并设计一个控制变量清楚、"  # 要求形成可证伪假设和最小实验。
    "最能区分这些假设的最小下一步实验。你不知道执行环境拥有哪些软件或能力，"  # 明确模型在提案阶段看不到工具能力。
    "不得假设存在现成工具，不得使用函数名、工具名或软件命令表达方案；请只用工程和物理语言描述想改变什么、"  # 禁止通过内部名称猜测候选路线。
    "保持什么不变、测量什么、怎样判断以及何时停止。证据不足时可以请求缺失信息，也可以明确停止并保留不确定性。"  # 允许请求事实和自主停止。
    "只输出一个合法 JSON 对象，字段必须为 competing_hypotheses、evidence_refs、uncertainties、proposal_type、"  # 开始定义通用无工具输出合同。
    "experiment、information_request、provisional_answer。proposal_type 只能是 experiment、request_information 或 stop。"  # 限制通用提案类型。
    "当 proposal_type=experiment 时，experiment 必须包含 purpose、change、hold_fixed、measure、decision_rule、stop_condition；"  # 定义控制实验所需字段。
    "其中 hold_fixed 和 measure 必须是字符串数组。当 proposal_type=request_information 时，information_request 必须包含 question。"  # 定义请求信息合同。
)  # 完成无工具可见系统提示词。

USER_INSTRUCTION = (  # 定义每轮固定用户指令。
    "请只根据下面的实时证据提出唯一下一步实验提案。提案提交后会先被冻结，之后由独立执行层判断能否忠实执行。"  # 说明冻结顺序但不暴露执行能力。
    "不要列举多个下一步动作，不要猜测执行器内部能力，不要补造未提供的模型事实。\n\n"  # 限制单轮只产生一个可审计方案。
)  # 完成每轮用户指令。


def _usage_dict(usage: Any) -> dict[str, Any]:  # 把 SDK 使用量对象转换为可序列化字典。
    if usage is None:  # 检查响应是否缺少使用量。
        return {}  # 在缺失时返回空对象。
    if hasattr(usage, "model_dump"):  # 优先使用新版 SDK 序列化方法。
        return usage.model_dump()  # 返回完整模型使用量字段。
    return {name: getattr(usage, name) for name in ("prompt_tokens", "completion_tokens", "total_tokens") if getattr(usage, name, None) is not None}  # 兼容旧版 SDK 字段。


def _request_proposal(client: OpenAI, model: str, evidence_packet: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:  # 请求一份不含工具动作的实验提案。
    packet_text = json.dumps(evidence_packet, ensure_ascii=False, indent=2)  # 把实时证据序列化为模型可读文本。
    last_error: Exception | None = None  # 保存最近一次格式错误用于最终报告。
    for attempt in range(1, 3):  # 最多允许一次结构修复重试。
        repair = "" if attempt == 1 else "上一次输出结构无效，请保持实验目的不变并重新输出完整合法 JSON。\n\n"  # 仅修复 JSON 合同而不提示物理路线。
        response = client.chat.completions.create(  # 发起真实 DeepSeek 请求。
            model=model,  # 使用命令行或环境变量指定的模型。
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": repair + USER_INSTRUCTION + packet_text}],  # 每次只发送当前提示和当前证据包。
            response_format={"type": "json_object"},  # 要求 API 返回合法 JSON 对象。
            max_tokens=10000,  # 为竞争假设和实验设计保留足够输出预算。
            reasoning_effort="high",  # 请求高强度推理模式。
            extra_body={"thinking": {"type": "enabled"}},  # 显式启用 DeepSeek 思考模式但不保存隐藏推理。
            stream=False,  # 使用单次响应便于冻结完整提案。
        )  # 完成模型调用。
        content = response.choices[0].message.content  # 读取公开 JSON 输出文本。
        if not content or not content.strip():  # 检查模型是否返回空内容。
            last_error = RuntimeError("DeepSeek returned empty proposal")  # 保存空输出错误。
            continue  # 进入结构修复重试。
        try:  # 尝试解析 JSON 输出。
            proposal = json.loads(content)  # 把模型文本解析为提案对象。
        except json.JSONDecodeError as exc:  # 捕获 JSON 语法错误。
            last_error = exc  # 保存解析错误供重试。
            continue  # 进入下一次请求。
        errors = validate_proposal(proposal)  # 按无工具提案合同校验字段。
        if errors:  # 检查合同是否存在错误。
            last_error = RuntimeError("invalid proposal: " + "; ".join(errors))  # 保存全部合同错误。
            continue  # 允许模型只修复格式一次。
        metadata = {"requested_model": model, "response_model": response.model, "finish_reason": response.choices[0].finish_reason, "usage": _usage_dict(response.usage)}  # 记录真实模型和令牌使用量。
        return proposal, metadata  # 返回通过校验的提案和调用元数据。
    raise RuntimeError(f"DeepSeek did not return a valid proposal after two attempts: {last_error}")  # 在两次失败后停止实验。


def _initial_evidence() -> dict[str, Any]:  # 获取与旧实验一致但不含工具目录的初始真实证据。
    backend = _load_backend()  # 只读加载既有有限元模块。
    return backend._initial_evidence()  # 运行三档真实网格并返回公开数值字段。


def run_experiment(model: str, max_rounds: int, output_dir: Path) -> dict[str, Any]:  # 执行多轮提案冻结、隐藏执行和证据回传。
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")  # 从隔离工作流环境读取密钥。
    if not api_key:  # 检查凭据是否可用。
        raise RuntimeError("DEEPSEEK_API_KEY is required for live hidden-executor discovery")  # 禁止使用伪造模型输出。
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")  # 创建 DeepSeek API 客户端。
    output_dir.mkdir(parents=True, exist_ok=True)  # 创建独立结果目录。
    initial = _initial_evidence()  # 获取用户问题、模型事实和真实网格历史。
    public_history: list[dict[str, Any]] = []  # 初始化仅包含提案和脱敏反馈的模型可见历史。
    audit_rounds: list[dict[str, Any]] = []  # 初始化完整审计轨迹。
    previous_chain_hash = "0" * 64  # 使用固定创世哈希开始提案审计链。
    stopped_voluntarily = False  # 记录模型是否主动结束。
    for round_index in range(1, max_rounds + 1):  # 在固定预算内逐轮运行开放发现。
        evidence_packet = {"initial_evidence": initial, "previous_rounds": public_history, "round": round_index, "remaining_rounds": max_rounds - round_index + 1}  # 构造不含工具目录和内部映射的证据包。
        proposal, metadata = _request_proposal(client, model, evidence_packet)  # 让模型自由提出下一步工程实验。
        frozen = freeze_proposal(proposal, output_dir, round_index, previous_chain_hash)  # 在任何映射前写盘并生成哈希链。
        proposal_hash = str(frozen["seal"]["proposal_sha256"])  # 读取当前冻结提案摘要。
        previous_chain_hash = str(frozen["seal"]["chain_sha256"])  # 更新下一轮审计链前驱哈希。
        mapping = map_frozen_proposal(Path(frozen["round_dir"]), proposal_hash)  # 在模型不可见层执行确定性忠实映射。
        feedback = execute_mapping(Path(frozen["round_dir"]), proposal_hash, mapping)  # 执行真实求解或诚实返回不支持状态。
        public_round = {"round": round_index, "proposal": proposal, "execution_feedback": feedback}  # 组织下一轮模型允许看到的历史。
        public_history.append(public_round)  # 把当前公开证据加入下一轮上下文。
        audit_rounds.append({"round": round_index, "proposal_sha256": proposal_hash, "chain_sha256": previous_chain_hash, "metadata": metadata, "internal_mapping": mapping, "public_feedback": feedback})  # 保存完整审计摘要。
        if feedback.get("status") == "finished":  # 检查模型是否主动停止。
            stopped_voluntarily = True  # 标记自主结束。
            break  # 结束多轮实验。
        if feedback.get("status") == "information_required":  # 检查实验是否需要用户补充事实。
            break  # 在缺少真实信息时诚实停止自动循环。
    result = {"experiment": "deepseek_crack_hidden_executor", "model": model, "system_prompt": SYSTEM_PROMPT, "tool_catalog_visible_to_model": False, "internal_mapping_visible_to_model": False, "rounds": audit_rounds, "public_history": public_history, "stopped_voluntarily": stopped_voluntarily, "final_chain_sha256": previous_chain_hash}  # 组织完整实验结果。
    (output_dir / "experiment_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存最终审计结果。
    print(json.dumps({"status": "completed", "rounds": len(audit_rounds), "stopped_voluntarily": stopped_voluntarily, "output": str(output_dir / "experiment_result.json")}, ensure_ascii=False))  # 向 Actions 日志输出紧凑摘要。
    return result  # 返回结果供测试或其他隔离调用使用。


def main() -> int:  # 定义命令行入口。
    parser = argparse.ArgumentParser()  # 创建命令行解析器。
    parser.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro"))  # 允许覆盖 DeepSeek 模型名称。
    parser.add_argument("--max-rounds", type=int, default=5)  # 设置最大开放发现轮数。
    parser.add_argument("--output", type=Path, default=Path("artifacts/deepseek_crack_hidden_executor"))  # 设置隔离结果目录。
    args = parser.parse_args()  # 解析命令行参数。
    if args.max_rounds < 1 or args.max_rounds > 8:  # 限制真实 API 和有限元调用预算。
        raise ValueError("max-rounds must lie between 1 and 8")  # 拒绝异常运行预算。
    run_experiment(args.model, args.max_rounds, args.output)  # 执行完整隔离实验。
    return 0  # 返回成功退出码。


if __name__ == "__main__":  # 仅在脚本直接执行时启动实验。
    raise SystemExit(main())  # 把主函数退出码传给操作系统。

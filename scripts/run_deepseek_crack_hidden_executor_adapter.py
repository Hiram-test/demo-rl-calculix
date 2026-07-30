#!/usr/bin/env python3  # 使用兼容适配层启动隔离隐藏执行器实验。
from __future__ import annotations  # 启用现代类型注解。

import json  # 保存适配后的完整实验结果。
import os  # 读取受保护 DeepSeek 凭据。
import sys  # 把仓库根目录加入模块搜索路径。
from pathlib import Path  # 管理隔离输出目录和仓库根目录。

ROOT = Path(__file__).resolve().parents[1]  # 定位仓库根目录。
sys.path.insert(0, str(ROOT))  # 确保脚本直接执行时可以导入仓库内模块。

import scripts.run_deepseek_crack_hidden_executor as runner  # 导入已经通过静态合同的主运行器组件。
from experiments.hidden_executor import executor_adapter  # 导入语序无关映射和中性公开状态适配层。


def run_experiment(model: str, max_rounds: int, output_dir: Path) -> dict:  # 运行能够立即识别中性结束状态的多轮闭环。
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")  # 从隔离工作流环境读取真实模型凭据。
    if not api_key:  # 检查凭据是否存在。
        raise RuntimeError("DEEPSEEK_API_KEY is required for live hidden-executor discovery")  # 禁止生成伪造轨迹。
    client = runner.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")  # 创建 DeepSeek API 客户端。
    output_dir.mkdir(parents=True, exist_ok=True)  # 创建独立结果目录。
    initial = runner._initial_evidence()  # 获取真实三档网格证据且不提供工具目录。
    public_history: list[dict] = []  # 初始化只包含模型提案和公开物理反馈的历史。
    audit_rounds: list[dict] = []  # 初始化完整内部审计摘要。
    previous_chain_hash = "0" * 64  # 使用固定创世哈希开始提案链。
    stopped_voluntarily = False  # 初始化模型自主停止标记。
    for round_index in range(1, max_rounds + 1):  # 在固定预算内逐轮执行。
        evidence_packet = {"initial_evidence": initial, "previous_rounds": public_history, "round": round_index, "remaining_rounds": max_rounds - round_index + 1}  # 构造不含内部执行能力的当前证据包。
        proposal, metadata = runner._request_proposal(client, model, evidence_packet)  # 请求唯一自然语言工程实验提案。
        frozen = runner.freeze_proposal(proposal, output_dir, round_index, previous_chain_hash)  # 在任何映射前冻结提案并写入哈希链。
        proposal_hash = str(frozen["seal"]["proposal_sha256"])  # 读取当前提案摘要。
        previous_chain_hash = str(frozen["seal"]["chain_sha256"])  # 更新下一轮链前驱。
        round_dir = Path(frozen["round_dir"])  # 定位当前冻结轮次目录。
        mapping = executor_adapter.map_frozen_proposal(round_dir, proposal_hash)  # 在模型不可见层忠实映射冻结提案。
        feedback = executor_adapter.execute_mapping(round_dir, proposal_hash, mapping)  # 执行真实求解并生成中性公开反馈。
        public_round = {"round": round_index, "proposal": proposal, "execution_feedback": feedback}  # 组织下一轮模型可见历史。
        public_history.append(public_round)  # 追加公开证据历史。
        audit_rounds.append({"round": round_index, "proposal_sha256": proposal_hash, "chain_sha256": previous_chain_hash, "metadata": metadata, "internal_mapping": mapping, "public_feedback": feedback})  # 保存完整审计摘要。
        if feedback.get("status") == "analysis_complete":  # 检查模型是否自主决定结束。
            stopped_voluntarily = True  # 记录自主停止。
            break  # 立即终止自动循环并尊重模型决定。
        if feedback.get("status") == "information_required":  # 检查是否需要用户提供外部事实。
            break  # 在事实缺失时停止自动计算。
    result = {"experiment": "deepseek_crack_hidden_executor", "model": model, "system_prompt": runner.SYSTEM_PROMPT, "tool_catalog_visible_to_model": False, "internal_mapping_visible_to_model": False, "rounds": audit_rounds, "public_history": public_history, "stopped_voluntarily": stopped_voluntarily, "final_chain_sha256": previous_chain_hash}  # 组织最终实验记录。
    (output_dir / "experiment_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")  # 保存完整结果和审计链。
    print(json.dumps({"status": "completed", "rounds": len(audit_rounds), "stopped_voluntarily": stopped_voluntarily, "output": str(output_dir / "experiment_result.json")}, ensure_ascii=False))  # 输出紧凑 Actions 摘要。
    return result  # 返回完整实验结果。


runner.run_experiment = run_experiment  # 让主命令行入口使用适配后的多轮实现。


if __name__ == "__main__":  # 仅在脚本直接执行时启动实验。
    raise SystemExit(runner.main())  # 复用主运行器的参数解析和退出码逻辑。

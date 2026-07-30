#!/usr/bin/env python3  # 使用兼容适配层启动隔离隐藏执行器实验。
from __future__ import annotations  # 启用现代类型注解。

import scripts.run_deepseek_crack_hidden_executor as runner  # 导入已经通过语法检查的主运行器。
from experiments.hidden_executor import executor_adapter  # 导入语序无关映射和中性公开状态适配层。

runner.map_frozen_proposal = executor_adapter.map_frozen_proposal  # 替换主运行器中的隐藏映射入口。
runner.execute_mapping = executor_adapter.execute_mapping  # 替换主运行器中的公开反馈执行入口。

_original_run_experiment = runner.run_experiment  # 保存原始多轮实验函数供包装调用。


def run_experiment(model: str, max_rounds: int, output_dir):  # 包装原实验并识别中性结束状态。
    result = _original_run_experiment(model, max_rounds, output_dir)  # 执行完整冻结和隐藏执行闭环。
    if result.get("public_history"):  # 检查实验是否至少完成一轮。
        last_feedback = result["public_history"][-1].get("execution_feedback", {})  # 读取最后一轮公开反馈。
        if last_feedback.get("status") == "analysis_complete":  # 检查模型是否通过中性状态自主结束。
            result["stopped_voluntarily"] = True  # 修正最终审计摘要中的自主停止标记。
    return result  # 返回修正后的实验结果。


runner.run_experiment = run_experiment  # 让命令行入口调用带中性状态识别的包装函数。


if __name__ == "__main__":  # 仅在脚本直接执行时启动实验。
    raise SystemExit(runner.main())  # 复用主运行器的参数解析和退出码逻辑。

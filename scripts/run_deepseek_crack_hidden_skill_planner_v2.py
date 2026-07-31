#!/usr/bin/env python3  # 使用扩展Skill目录和修正后的任务证据语义运行双API闭环。
from __future__ import annotations  # 启用现代类型注解。

import scripts.run_deepseek_crack_hidden_skill_planner as runner  # 复用已经通过真实双API验证的主循环。
from experiments.skill_planner.executor_v2 import execute_frozen_plan  # 导入标准化外部阻塞的执行入口。
from experiments.skill_planner.skills_v2 import build_registry  # 导入包含Irwin塑性区计算的扩展Skill目录。

runner.build_registry = build_registry  # 让第二API看到扩展后的不可变Skill注册表。
runner.execute_frozen_plan = execute_frozen_plan  # 让任务控制器接收顶层标准外部阻塞字段。
runner.task_controller._MESH_EVIDENCE_OPERATIONS.add("fracture_parameter_sequence")  # 把跨网格断裂参量序列计入网格策略证据，因为它直接比较离散变化。
runner.DECISION_SYSTEM_PROMPT += ("已经记录在task_contract.external_blockers中的同一外部事实不得反复请求；当全部可计算决策门已关闭且剩余限制来自外部数据时，应提交包含条件性结论的resolve_task，由独立控制器裁决为task_blocked。")  # 防止有限轮次被重复材料请求消耗。
runner.DECISION_USER_INSTRUCTION += ("若外部阻塞已经记录，请继续关闭可计算决策门或提交条件性最终答复，不要重复请求同一事实。\n\n")  # 每轮提醒第一API区分外部阻塞和可计算任务。


if __name__ == "__main__":  # 仅在脚本直接执行时启动扩展双API实验。
    raise SystemExit(runner.main())  # 复用既有参数解析并使用已经替换的模块级入口。

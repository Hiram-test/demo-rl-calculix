# DeepSeek 裂纹开放发现：隐藏执行器与任务级控制器实验

## 隔离边界

本实验位于独立分支 `exp/deepseek-crack-hidden-executor`，堆叠在 `exp/deepseek-crack-open-discovery` 之上。所有实现均限定在 `experiments/hidden_executor/`、隔离运行脚本、专属测试、专属工作流、本说明文件和独立结果目录中。原论文脚本、原 Skill 合同、原 MD、原工作流和既有实验结果目录不被修改。

## 实验目标

实验要验证的完整闭环是：DeepSeek 在看不到工具目录的条件下自主识别当前路线是否有效；当当前路线应停止时，自主提出下一条工程路线；自然语言提案先被冻结；隐藏执行器再判断能否忠实实现；真实物理反馈返回后，DeepSeek继续决策，直到原始用户问题被解决、被真实外部数据阻塞，或预算耗尽并明确标记为未完成。

当前路线结束不等于总体任务结束。旧版单一 `stop` 会被兼容解释为 `switch_route`，只能关闭当前方法。总体任务只有在独立任务控制器确认全部决策门关闭后，才接受 `resolve_task`。

## 模型可见任务合同

第一阶段只向 DeepSeek 提供用户问题、模型事实、已有数值证据、前轮公开提案、公开物理反馈和通用任务决策门。模型看不到工具目录、函数名称、内部参数合同、映射规则和执行器能力。

任务控制器使用四个不指定具体工程路线的完成门：

1. 已有证据足以判断是否继续改变网格或离散策略；
2. 至少一个适合原始工程决策的评价量已经得到可比较的数值证据，不能只依赖局部峰值或数值平衡；
3. 当前物理模型的适用范围、缺失机制和所需外部数据已经明确评估；
4. 最终答复分别说明是否继续细化、当前模型能否用于判断以及下一步具体行动。

模型每轮只能输出一个提案：

- `experiment`：提出一个受控工程实验；
- `request_information`：请求缺失事实；
- `switch_route`：结束当前方法并明确下一路线，总体任务继续；
- `resolve_task`：提交结构化最终答复，等待独立任务门检查。

`resolve_task` 的最终答复必须包含 `continue_refinement`、`model_usability`、`next_action` 和 `remaining_uncertainties`。当前路线满足停止条件时不得使用总体完成语义。

## 冻结与隐藏执行

每轮模型原始提案先写入 `proposal.json`，随后生成 `proposal_seal.json`。封条包含提案 SHA256、前一轮链哈希、当前链哈希和 UTC 时间。隐藏映射和执行开始前会重新计算提案哈希，任何改写都会终止运行。

确定性隐藏执行器只允许忠实映射当前后端能够完整实现的实验。无法完整实现时返回 `unsupported`，不得只执行联合实验的一部分，也不得替换成已知路线。内部映射和原始结果保存在 `mapping_receipt.json` 与 `execution_audit.json`；下一轮 DeepSeek 只能看到 `public_feedback.json` 中的物理改变、实际参数、数值观测和限制。

路线切换和任务完成候选不会调用有限元后端。`switch_route` 生成 `route_transition` 反馈并进入下一轮；`resolve_task` 由独立任务控制器检查。若仍有未决决策门，状态为 `resolution_rejected` 并继续；若可计算部分完成但必须等待外部数据，状态为 `task_blocked`；全部门关闭时状态为 `task_resolved`。

## 预算语义

轮次预算只限制模型调用和有限元计算成本，不能作为任务完成证据。工作流最多运行八轮。八轮后仍有未决决策门时，最终状态必须是 `inconclusive_budget_exhausted`，不得写成已经解决或自主完成。

## 允许复用的既有代码

隐藏执行器以只读方式动态加载 `scripts/run_deepseek_crack_open_discovery.py` 中的真实有限元求解函数和缓存，不更新该文件，也不向模型暴露其中的工具目录。裂纹微增量若不与结构网格节点对齐，执行器会吸附到完整网格步长，并记录请求值和实际值。

## 合同测试

静态检查覆盖：模型提示无内部工具名；提案先冻结后映射；映射前后哈希一致；显式目标尺寸忠实执行；无法完整实现的联合实验整项拒绝；公开反馈删除内部字段；遗留 `stop` 只能成为路线切换；只有网格证据时任务完成声明必须被拒绝；隐藏评价量证据和完整工程答复齐备后才允许任务完成；预算耗尽必须标记未完成。

## GitHub Actions

PR事件只运行编译和静态合同测试，不读取 DeepSeek Secret。真实模型闭环只在隔离分支 push 或手动触发中运行，使用既有 `ds` Environment。工作流上传独立 Artifact `deepseek-crack-hidden-executor`，并把脱敏决策链冻结到 `experiments/results/latest_deepseek_crack_hidden_executor/`。工作流不会触碰原论文分支或旧开放发现结果。

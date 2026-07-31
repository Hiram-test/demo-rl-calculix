# DeepSeek裂纹问题：双API隐藏Skill规划实验

## 实验目标

本实验验证以下闭环：第一API在看不到Skill目录的条件下提出工程实验；提案先冻结；第二API单独读取冻结提案、真实证据和隐藏Skill目录，把工程意图编译为一个或多个Skill调用；确定性控制器检查Skill是否存在、参数是否有证据来源、全部实验要求是否完整覆盖、调用依赖是否有效；只有通过全部合同后才执行真实Skill；第一API下一轮只看到物理变化、实际参数、数值观测和限制。

该实验不再使用不断扩展的关键词适配器作为主路由。旧隐藏执行器分支和结果保留，用于和双API Skill规划架构做对照。

## 隔离边界

实现位于独立分支 `exp/deepseek-crack-hidden-skill-planner-v3`，堆叠在 `exp/deepseek-crack-hidden-executor` 之上。新增内容限定在：

- `experiments/skill_planner/`
- `scripts/run_deepseek_crack_hidden_skill_planner.py`
- `tests/test_deepseek_crack_hidden_skill_planner.py`
- `.github/workflows/deepseek-crack-hidden-skill-planner.yml`
- 本协议文件
- 独立结果目录 `experiments/results/latest_deepseek_crack_hidden_skill_planner/`

旧开放发现、旧隐藏执行器、原论文流程和既有结果目录均不修改。

## 两个API的职责

### 第一API：工程决策代理

第一API只接收：用户原始问题、模型事实、数值证据、前轮公开提案、前轮公开物理反馈和总体任务决策门。它不知道有哪些Skill，也不能输出Skill名、函数名或工具命令。它只用工程语言表达：

- 竞争假设；
- 准备改变什么；
- 保持什么不变；
- 需要观测或派生什么；
- 如何判断；
- 当前步骤、当前路线或总体任务何时结束。

第一API提案写入 `proposal.json` 并由 `proposal_seal.json` 绑定SHA256链后，第二API才允许读取。

### 第二API：隐藏Skill执行编译器

第二API可以看到隐藏Skill目录，但不负责重新回答工程问题。它必须输出：

- `experiment_spec`：统一实验中间表示；
- `calls`：有序Skill调用图；
- 每项参数的真实来源；
- 每项实验要求由哪个调用覆盖；
- 无法完整执行时的诚实拒绝理由。

第二API计划写入 `skill_plan.json`，由 `skill_plan_seal.json` 同时绑定第一API提案摘要、Skill目录摘要和计划摘要。第一API永远看不到该计划和Skill目录。

## ExperimentSpec

统一实验规范包含：

- `objective`
- `scope`
- `interventions`
- `invariants`
- `observables`
- `derivations`
- `external_dependencies`
- `acceptance_rule`
- `completion_scope`

`observables`、`derivations` 和 `external_dependencies` 中的每一项都有稳定ID。可执行计划必须逐项覆盖全部ID；任何未覆盖要求都会使整项计划被拒绝，不允许只执行容易的一部分。

## 确定性安全门

第二API输出后，普通程序执行以下检查：

1. Skill标识必须存在于冻结目录；
2. 参数名、类型和必需性必须满足Skill合同；
3. 参数来源只能是冻结提案、初始证据、前轮公开证据或前序Skill输出；
4. 非前序Skill输出的数值参数必须真实出现在冻结提案或证据包中；
5. 依赖只能指向前方已经完成的调用，禁止环和未来依赖；
6. 所有实验要求必须完整覆盖；
7. 计划和提案的SHA256在执行前再次验证。

任何一项失败都拒绝执行，不把第二API的自然语言判断直接当作可信工具调用。

## 初始Skill目录

初始目录包含可组合能力：

- 多档网格裂纹微增能量差G/K序列；
- 目标网格细化并复算G/K；
- 普通网格求解；
- 真实裂纹面节点位移提取及位移法K；
- 基于已有网格结果的广义Richardson外推；
- 外部材料数据请求。

每项Skill均声明输入合同、输出物理字段、模型影响和固有限制。Skill处理函数由确定性注册表绑定，第二API不能动态生成代码或新增Skill。

## 公开反馈与内部审计

每轮内部文件包括：

- `proposal.json`
- `proposal_seal.json`
- `skill_plan.json`
- `skill_plan_seal.json`
- `skill_plan_validation.json`
- `skill_execution_audit.json`
- `public_feedback.json`

第一API只收到 `public_feedback.json`。该文件不包含Skill标识、规划模型选择或内部调用参数来源；完整信息只保留在内部审计文件中。

## 双API配置

第一API使用现有：

- `DEEPSEEK_API_KEY`
- `DEEPSEEK_MODEL`

第二API独立支持：

- `SKILL_PLANNER_API_KEY`
- `SKILL_PLANNER_BASE_URL`
- `SKILL_PLANNER_MODEL`

未单独配置第二API密钥时，实验允许使用相同密钥发起独立第二次API调用，但决策上下文、系统提示和可见目录仍完全隔离。论文实验应记录两个模型名、端点和使用量。

## 终止语义

当前实验停止、当前路线停止和总体任务完成继续分离。`switch_route` 不调用Skill并进入下一轮；`resolve_task` 由既有独立任务控制器检查。预算耗尽时只能记录 `inconclusive_budget_exhausted`，不能伪装成任务解决。

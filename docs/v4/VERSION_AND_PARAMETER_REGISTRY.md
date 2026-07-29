# V4 版本与参数登记表

> 状态：强制参考文档。后续修改不得从聊天上下文、旧论文、旧 artifact 或模型记忆中复制参数；只能从本文件登记的版本来源、当前用户输入和当前任务解析结果中取值。

## 1. 文档目的

本项目曾把不同实验谱系、不同参数规模和不同 Agent 定义混入同一条 V4 工作流。为避免后续模型依赖旧上下文继续拼接，本文件把“已经观察到的版本”“当前运行版本”和“目标重构版本”明确隔离。

任何新提交必须先回答：

1. 修改属于哪个版本谱系？
2. 参数来自用户输入、模型文件解析、代码默认值，还是旧实验？
3. 是否引入了案例专用常量？
4. 是否改变了论文、工作流门槛、求解规模或 Skill 语义？

## 2. 版本谱系总表

| 项目 | 旧版/回放论文谱系 | 当前 V4 Live Attempt 2 | 目标重构版 |
|---|---|---|---|
| 主要用途 | 固定案例、确定性回放与论文复现 | DeepSeek 实时调度四个预制案例 | 面向任意用户模型的代码生成工程 Agent |
| 问题来源 | 预构造 benchmark | 预构造 benchmark | 用户文件、自然语言与补充问答 |
| 几何来源 | 代码内固定 | 代码内固定 | 从 `.inp`/CAD/网格/JSON 解析；缺失则询问用户 |
| 材料来源 | 代码内固定 | 代码内固定 | 从模型解析；缺失则询问用户或要求确认假设 |
| 荷载/约束来源 | 代码内固定 | 代码内固定 | 从模型解析；无法判定时不得擅自补值 |
| DeepSeek 角色 | 回放预先冻结的决策 | 从预写 Skill 菜单中选择 action 和参数 | 选择方法 Skill，并针对当前对象生成/修改代码 |
| Skill 含义 | 固定实验步骤 | 案例专用预写函数 | 方法论、规则库、接口和验证原则 |
| 代码生成 | 无/有限 | 关键建模代码预先写好 | DeepSeek 为当前任务生成 builder、solver、postprocess、optimizer 代码 |
| 区域划分 | 旧文档描述的回放方案 | DeepSeek 生成 region JSON，预写代码执行 | DeepSeek 基于当前模型生成区域与代码；校验规则公开 |
| PSO 规模 | 旧文档曾出现 4 粒子、1 代、约 28 次区域求解；未经当前源码复核不得引用 | 工作流门槛要求每例 `actual_solver_evaluations >= 18`、`executed_iterations >= 3`，总 `solver_calls >= 85` | 由当前问题规模与预算确定，并写入任务 manifest |
| Provider | 冻结/回放 | `deepseek` / `deepseek-v4-pro` | 可配置；provider 与模型必须记录在 manifest |
| 结束条件 | 旧论文/回放条件 | 固定 required artifact 清单 + 总体验收 | 证据充分、用户目标满足、缺失信息已处理；不得靠案例专属 artifact 强迫路线 |
| 日志 | 旧版未知 | 单一黑箱步骤，stdout 缓冲 | 案例/决策/求解/PSO 逐级实时日志与检查点 |
| 失败恢复 | 旧版未知 | 四案例全跑后汇总失败 | 案例级与步骤级续跑，不重复已完成计算 |

## 3. 当前 V4 Live 的工作流硬门槛

以下是当前 `.github/workflows/deepseek-ai-region-mesh-paper-v4.yml` 中的验收值。它们只属于当前 Live 谱系，不得反向写入旧论文，也不得自动沿用到重构版。

| 门槛 | 当前值 | 性质 |
|---|---:|---|
| 场景数 | 4 | 固定 benchmark 数量 |
| optimizer 调用 | 4 | 每案例一次 |
| 总 solver 调用 | `>= 85` | 验收门槛，不是循环控制参数 |
| 每例 PSO 求解 | `>= 18` | 验收门槛 |
| 每例 PSO 代数 | `>= 3` | 验收门槛 |
| 同预算正向案例 | `>= 2` | 总体结果门槛 |
| 区域数量 | `>= 2` | 几何合同 |
| 可粗化区域 | 至少 1 个 | 几何/策略合同 |
| 同预算单元数偏差 | `<= 8%` | 比较合同 |
| 无效单元 | 0 | 网格质量合同 |
| job 总超时 | 180 分钟 | GitHub Actions 上限 |

## 4. Attempt 2 的实际运行记录

| 项目 | 实际值 |
|---|---:|
| GitHub Run | `30450374761`，Attempt 2 |
| DeepSeek 决策 | 73 |
| solver 调用 | 119 |
| optimizer 调用 | 4 |
| 场景完成状态 | 四个场景均 `finished` |
| 最终 suite | `valid: false` |
| 最终错误 | `bearing_load_introduction: partition lacks geometric diversity` |
| 论文构建 | 跳过 |
| artifact | 已上传 |

### 各案例调用数量

| case_id | 模型决策 | 结果状态 | 已知问题 |
|---|---:|---|---|
| `bearing_load_introduction` | 20 | finished | 区域几何多样性校验失败 |
| `web_circular_opening` | 18 | finished | 未触发最终一票否决 |
| `diaphragm_multi_opening_budget` | 19 | finished | 未触发最终一票否决 |
| `bridge_web_crack` | 16 | finished | 未触发最终一票否决 |

## 5. 当前四个预制案例的硬编码对象参数

这些值是当前 payload 在 DeepSeek 第一次调用前就写入 `model_facts` 的内容。它们不是 DeepSeek 从用户模型解析得到的，也不是 DeepSeek 生成的。

### 5.1 `bearing_load_introduction`

| 字段 | 硬编码值 | 分类 | 重构要求 |
|---|---:|---|---|
| `kind` | `bearing_cantilever_plate` | 预制对象类型 | 从模型拓扑/用户说明推断 |
| `length_mm` | 1000 | 几何 | 从模型解析 |
| `height_mm` | 100 | 几何 | 从模型解析 |
| `thickness_mm` | 12 | 几何 | 从 section/实体解析 |
| `young_mpa` | 210000 | 材料 | 从材料卡解析 |
| `poisson` | 0.3 | 材料 | 从材料卡解析 |
| `total_vertical_force_n` | -600000 | 荷载 | 从载荷/反力数据解析 |
| `candidate_bearing_width_mm` | 40 | **候选答案/决策变量** | 禁止预置；由 DeepSeek 论证、搜索或询问用户 |

### 5.2 `web_circular_opening`

| 字段 | 硬编码值 | 分类 | 重构要求 |
|---|---:|---|---|
| `kind` | `circular_opening_plate` | 预制对象类型 | 从几何解析 |
| `width_mm` | 240 | 几何 | 从模型解析 |
| `height_mm` | 240 | 几何 | 从模型解析 |
| `hole_radius_mm` | 20 | 几何 | 从孔边界解析 |
| `thickness_mm` | 10 | 几何 | 从 section 解析 |
| `young_mpa` | 210000 | 材料 | 从材料解析 |
| `poisson` | 0.3 | 材料 | 从材料解析 |
| `remote_tension_mpa` | 100 | 荷载 | 从边界/载荷解析 |

### 5.3 `diaphragm_multi_opening_budget`

| 字段 | 硬编码值 | 分类 | 重构要求 |
|---|---:|---|---|
| `kind` | `multi_hole_diaphragm` | 预制对象类型 | 从模型解析 |
| `width_mm` | 600 | 几何 | 从模型解析 |
| `height_mm` | 260 | 几何 | 从模型解析 |
| `thickness_mm` | 12 | 几何 | 从 section 解析 |
| `young_mpa` | 210000 | 材料 | 从材料解析 |
| `poisson` | 0.3 | 材料 | 从材料解析 |
| `remote_tension_mpa` | 90 | 荷载 | 从载荷解析 |
| `holes` | `(-170,0,24)`, `(0,0,42)`, `(150,0,30)` | 几何对象 | 从孔边界解析 |
| `angular_budget` | 144 | 算法/网格预算 | 由用户预算或 Agent 规划，不得冒充模型事实 |

### 5.4 `bridge_web_crack`

| 字段 | 硬编码值 | 分类 | 重构要求 |
|---|---:|---|---|
| `kind` | `central_crack_panel` | 预制对象类型 | 从裂纹/几何解析 |
| `width_mm` | 200 | 几何 | 从模型解析 |
| `height_mm` | 200 | 几何 | 从模型解析 |
| `half_crack_mm` | 20 | 缺陷几何 | 从裂纹定义解析 |
| `thickness_mm` | 10 | 几何 | 从 section 解析 |
| `young_mpa` | 210000 | 材料 | 从材料解析 |
| `poisson` | 0.3 | 材料 | 从材料解析 |
| `remote_tension_mpa` | 220 | 荷载 | 从载荷解析 |
| `yield_curve` | `(355,0)`, `(400,0.02)`, `(450,0.1)` | 材料本构 | 从真实材料曲线解析；缺失则询问用户 |

## 6. 参数来源分类规则

每个运行参数必须带来源标签：

| 标签 | 含义 | 是否可直接执行 |
|---|---|---|
| `parsed_from_user_model` | 从用户文件确定性解析 | 可以 |
| `provided_by_user` | 用户明确给出 | 可以 |
| `derived_by_agent` | DeepSeek 根据已知事实推导，并提供依据 | 经校验后可以 |
| `assumption_pending_confirmation` | Agent 提出但用户未确认 | 不得用于正式结论 |
| `algorithm_configuration` | 粒子数、代数、容差、预算等算法配置 | 必须写入任务 manifest |
| `legacy_fixture` | 旧 benchmark 固定值 | 只能用于明确标记的回归测试 |
| `forbidden_hidden_default` | 缺参时静默补入的默认值 | 禁止 |

## 7. 禁止混用规则

1. 不得把旧论文的粒子数、代数和求解次数复制到 Live 工作流，反之亦然。
2. 不得把 Attempt 2 的四个固定 `model_facts` 当成新任务默认值。
3. 不得从聊天历史或模型记忆恢复几何、材料、荷载和优化参数。
4. 不得把 `candidate_*` 字段放进客观模型事实；候选值必须有来源与依据。
5. 不得让论文描述与执行 manifest 使用不同的参数。
6. 不得仅修改工作流 gate 而不更新代码、测试、文档和论文。
7. 不得在同一版本名 `V4` 下同时表示 replay 与 live 两套实验；新实现必须使用新的 version id。

## 8. 每次运行必须生成的 `run_manifest.json`

```json
{
  "schema_version": "1.0",
  "system_version": "v5-codegen-agent",
  "mode": "live",
  "provider": "deepseek",
  "model": "...",
  "input_files": [],
  "parsed_facts": {},
  "missing_required_facts": [],
  "user_confirmed_assumptions": [],
  "agent_derived_parameters": [],
  "algorithm_configuration": {
    "pso_particles": null,
    "pso_iterations": null,
    "solver_budget": null
  },
  "legacy_values_used": [],
  "source_commit": "..."
}
```

当 `missing_required_facts` 非空时，正式求解不得启动。

# CalculiX 过盈接触诊断：原帖证据、DeepSeek loop 与论文产物

## 这次实验回答什么

实验对象是 CalculiX 论坛问题 [Interference Contact and mesh refinement](https://calculix.discourse.group/t/interference-contact-and-mesh-refinement/2747)。用户报告二次实体单元的过盈接触在粗网格下可以运行，细化后边中间节点进入干涉，求解器反复 cutback；后续模型还出现接触反力存在、位移不动、残差近零和位移修正约 `1e-30` 的现象。

这不是 DQN 训练、网格优化或关键词分类。实验只检验一件事：DeepSeek 能否在有限调用中提出竞争机制，选择有判别力的真实 CalculiX 实验，根据求解反馈修正判断，并最终形成带边界的工程答复。

## 精确原始证据

原帖公开 Proton 附件已经取得并解密。归档和两个 `.inp` 的文件名、大小、SHA-256、节点数、C3D20R 单元数、接触面规模、步骤差异及作者观测冻结在：

- [`experiments/calculix_interference_2747/original_deck_audit.json`](../experiments/calculix_interference_2747/original_deck_audit.json)

两个附件不是单因素对照。除网格规模和 PIN2 接触面数量不同外，细模型还在 Step 1 移除该接触、Step 2 重新加入；因此不能把“细网格失败”直接解释成单一网格效应。

原模型约有一千一百六十万自由度，并指定 Pardiso。标准 GitHub Actions 的 CalculiX 包使用不同后端，也没有证明具备原模型所需内存。本实验不会冒充已经精确执行原始大型模型。

## 代表性真实求解

真实求解基于 CalculiX 官方 `test/contact4.inp` 的两个二次实体接触块，并把单元统一为原帖同族的 `C3D20R`。为形成健康对照，代表模型使用 node-to-surface；这是隔离机制的实验设置，不是原帖的预设修复。

首轮模型调用前固定运行两例：

1. 对照：上块接触面四个边中间节点与角点共面，`z=1.00`；
2. 目标：只把四个边中间节点改成 `z=0.95`，角点仍为 `z=1.00`。

两例的材料、边界、载荷、接触、增量和求解命令相同。目标例使二次接触面发生局部初始过闭合和弯曲。它是与原帖“中间节点进入干涉后失败”相似的机制实验，不是原圆柱孔模型的几何复刻。

脚本不以退出码或 `.frd` 是否存在判断成功，而是同时记录：

- `Job finished` 与 `*ERROR: too many cutbacks`；
- 增量、尝试和 `no convergence` 次数；
- 接触弹簧活动数量；
- 最大残差和最大位移修正；
- `.sta`、`.cvg`、`.dat`、`.frd` 与 deck 哈希；
- 成功结果中的位移和反力摘要。

## 应用级 Skill 在哪里

本实验使用的通用工程经验位于：

- [`skills/engineering/nonlinear-contact-diagnosis.json`](../skills/engineering/nonlinear-contact-diagnosis.json)

它不是 Codex 的 `SKILL.md`，而是应用运行时发送给 DeepSeek 的工程经验 catalog。它只规定以下通用纪律：

- 分离用户观察、输入事实、求解观察和未验证解释；
- 保留多个可证伪竞争假设；
- 在实验前声明正、负结果各自意味着什么；
- 单次只改变一个登记字段；
- 把失败求解作为证据；
- 在留出几何上复验；
- 区分已确认修复、可能 workaround 和缩小但未解决。

它不包含本帖的坐标、接触参数、正确工具顺序或最终答案。

## Schema 是什么

应用级 Skill 的结构合同位于：

- [`schemas/engineering_skill.schema.json`](../schemas/engineering_skill.schema.json)

主要字段含义：

| 字段 | 含义 |
|---|---|
| `skill_id` / `title` / `purpose` | 可复用工程经验的身份和用途 |
| `engineering_questions` | 应主动区分的工程问题，不是当前案例答案 |
| `evidence_requirements` | 所需证据、用途、可接受来源和是否可选 |
| `procedure` | 通用诊断步骤 |
| `tool_guidance` | 工具选择与单因素控制原则 |
| `output_records` | 应保存的审计记录 |
| `non_goals` | 明确禁止硬编码、伪复现和越界结论 |
| `tags` / `version` | 检索标签和版本 |

当前问题事实使用：

- [`schemas/problem_manifest.schema.json`](../schemas/problem_manifest.schema.json)

`ProblemManifest` 把事实、缺失事实和算法配置分开。每个事实都带 `provenance`，例如 `parsed_from_user_model`、`provided_by_user`、`derived_by_agent` 或 `algorithm_configuration`。代表模型的 `z=0.95`、最大三次调用等属于实验或算法配置，不冒充原帖事实。

JSON 语法不支持注释，因此本节逐项说明两个 JSON schema 和 Skill 配置；不会为了添加注释而破坏 JSON 有效性。

## DeepSeek 决策 loop

整个运行只创建一个 API client，并使用一条不断追加的 `messages` 历史。不会为每轮开新对话，也不会因为缓存未命中、非法 JSON 或判断不理想而重跑。

通常流程：

1. Python 冻结原帖、附件审计、应用 Skill、问题清单和两例初始 CalculiX 结果；
2. 第一次 DeepSeek 调用提出竞争假设，并只选择一个判别动作；
3. Python 执行所选检查或单因素 CalculiX 对照，把原始结果追加到同一历史；
4. 第二次 DeepSeek 调用更新假设，可直接结束，也可再选择一个互补动作；
5. 若仍选择动作，Python 执行后进行第三次且最后一次 DeepSeek 调用；
6. 无论结论是确认、workaround 或未解决，都生成 trace、Markdown 论文和 PDF 论文。

工具目录包括原 deck 语义检查、代表模型几何检查、官方接触语义查阅、减小初始增量对照、显式线性 penalty 对照、接触形式对照和主动结束。目录不告诉模型哪个工具会成功。

硬预算：

- 最多三次 DeepSeek HTTP 请求；
- `max_retries=0`；
- 每轮只执行一个动作；
- 同一求解变体命中本地结果缓存时不重复执行；
- 非法 JSON 冻结原始回答，不发“修 JSON”请求；
- 记录服务返回的 cache hit/miss token；
- 论文生成额外 DeepSeek 调用为零。

## 没有工程门禁

运行时只检查能否解析并执行模型请求，不用预设答案、指定 Skill 顺序或 required-artifact 清单退回 DeepSeek。求解失败、工具参数错误、模型提前结束和证据不足都进入 trace，并照常形成论文。

唯一的硬停止条件是三次调用预算、传输错误或无法执行求解器。这些是费用和执行边界，不是要求模型走到某个答案才能“通关”。

## 论文来源契约

论文生成方式复用远端论文分支 `feat/ai-dynamic-region-mesh-paper` 中 `decision_paper.py` 的来源原则，并适配为单一接触案例：

- 最终工程答复使用 trace 中的 DeepSeek 原文；
- 逐轮假设、动作、预测和工具结果来自同一 `agent_trace.json`；
- 求解表格来自实际 CalculiX 运行；
- 原帖事实来自冻结的附件审计；
- 适用和不适用范围明确列出；
- `paper_provenance.json` 保存 trace、Skill、source audit、Markdown 和 PDF 的 SHA-256；
- 论文渲染不产生任何额外模型请求。

最终 artifact 预期包含：

- `agent_trace.json`
- `problem_manifest.json`
- `solver_runs/` 下的 deck、日志及原始结果
- `PAPER.md`
- `output/pdf/deepseek_calculix_interference_diagnosis.pdf`
- PDF 页面 PNG 预览
- `paper_provenance.json`
- `live_exit_code.txt`

## 结论边界

若代表模型上的措施通过留出验证，也只能称为机制证据或候选 workaround。只有在原始圆柱孔几何、原始 contact pair、Pardiso/批准后端和完整边界上再次验证位移、反力、接触传力与穿透后，才可声称解决原帖大型模型。

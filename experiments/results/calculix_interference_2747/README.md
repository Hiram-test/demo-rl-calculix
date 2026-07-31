# CalculiX 论坛 2747：DeepSeek 有限调用诊断结果

## 结论

原帖附件已经取得并按 SHA-256 冻结，不是根据描述虚构的替代题。原输入约有 1160 万自由度并请求 Pardiso；本次标准 GitHub runner 没有冒充完成原始大型模型，而是用真实 CalculiX C3D20R 代表模型复现“边中间节点进入过盈后反复 cutback”的数值现象。

最终 DeepSeek 状态为 `narrowed_unresolved`：它找到了一条在代表模型上能切换收敛状态的单因素路径，但没有把该路径误报为原帖的确认修复。

## 两次云端运行

1. [运行 30587437626](https://github.com/Hiram-test/demo-rl-calculix/actions/runs/30587437626)：第一轮形成了三个竞争假设并准备选择较小初始增量，但推理与 JSON 在 2400-token 上限处被截断。实际请求 1 次，没有执行补充求解。
2. [运行 30587593587](https://github.com/Hiram-test/demo-rl-calculix/actions/runs/30587593587)：保持首轮题目、证据和 Skill 不变，仅把单次完成上限调整为 4000。实际请求 3 次、API client 1 个、SDK 自动重试 0；缓存命中 15104 tokens、未命中 4424 tokens，命中率 77.35%。

第二次运行的首轮 4369-token 输入中有 4352 tokens 命中缓存，说明“保持稳定前缀、在同一增长历史中继续”的费用控制确实生效。

## DeepSeek 实际决策链

### 第 1 轮

DeepSeek看到以下单因素初始证据：

- 对照：接触面边中间节点 `z=1.00`，node-to-surface，求解完成；
- 目标：只把四个边中间节点改为 `z=0.95`，其余设置不变，求解失败；
- 失败目标累计 68 条 `no convergence`，6 次增量尝试，最大位移修正约 `1.837256e11`。

它选择 `run_surface_to_surface_pair`，在执行前声明：

- 若 surface-to-surface 收敛，则接触形式假设得到支持；
- 若仍以相似极端修正失败，则该假设被削弱。

真实 CalculiX 反馈为：只改 `contact_pair.type` 后，surface-to-surface 代表模型完成，`SOLVE-ID=SOLVE-F06394DCE1DB`。

### 第 2 轮

DeepSeek没有停在“surface-to-surface 就是答案”。它注意到原始附件本来已经使用 surface-to-surface，于是选择 `inspect_original_deck_semantics`，确认：

- 原模型为 C3D20R、surface-to-surface、线性 pressure-overclosure `69000`、请求 Pardiso；
- 两个附件同时改变网格规模、PIN2 接触面数量和接触激活历史；
- 因而附件不是单因素网格细化对照。

### 第 3 轮

DeepSeek以 `narrowed_unresolved` 停止。它把非均匀初始过闭合、圆柱曲面 face matching、penalty 和 Pardiso/SPOOLES 差异保留为候选，并明确说明未在原始大型模型上确认修复。

## 工程审计

这次结果可用于证明 DeepSeek会根据求解反馈改判，也会在发现“候选措施原模型已经使用”后退回继续检查；不能用于证明论坛问题已经解决。

还需要特别注意：

- surface-to-surface 代表模型虽然最终完成，过程中仍记录 68 条 `no convergence`，最大位移修正约 `0.1837113`；因此模型原文中的 “robustly handles” 应理解为“本代表算例最终完成”，不能解释为干净、普遍稳定；
- DeepSeek 最终建议的 penalty `1e5/1e6`、初始增量 `0.01` 和 `ADJUST=NO` 没有在这条 DeepSeek 决策 trace 中执行；随后不调用 DeepSeek 的本地反事实已经执行初始增量与 penalty，并把 `ADJUST=NO` 保留为尚未验证项；
- 代表模型不是原圆柱孔几何，也没有使用原 deck 请求的 Pardiso；
- 只有在原几何、原 contact pair、批准后端和完整边界上重新检查位移、反力、接触压力与穿透后，候选措施才能升级为确认修复。

## 交付文件

- [原始大型输入 Release](https://github.com/Hiram-test/demo-rl-calculix/releases/tag/calculix-interference-2747-evidence-v1)
- [原始输入来源清单、哈希、权利边界与发布决策](../../calculix_interference_2747/source/README.md)
- [论文 Markdown](PAPER.md)
- [论文 PDF](output/pdf/deepseek_calculix_interference_diagnosis.pdf)
- [最终完整决策 trace](agent_trace.json)
- [首次截断 trace](agent_trace_first_attempt_truncated.json)
- [ProblemManifest](problem_manifest.json)
- [交付来源收据](paper_provenance.json)
- [云端原始论文来源收据](paper_provenance_cloud.json)
- [本地三项反事实复核](local_followup/LOCAL_FOLLOWUP.md)
- [本地工程决策日志](local_followup/LOCAL_ENGINEER_DECISIONS.md)
- [本地机器决策 trace](local_followup/local_engineer_trace.json)
- [本地复核 PDF](local_followup/output/pdf/local_calculix_counterfactual_followup.pdf)
- [本地复核来源收据](local_followup/local_followup_provenance.json)

云端 PDF 使用的 Noto TTC 与 ReportLab 不兼容，中文被渲染成方块。交付 PDF 由同一个冻结 trace 在本地改用可嵌入 CJK 字体重新排版，额外 DeepSeek 调用为 0；三页均已逐页检查。

## 后续本地反事实结论

本地 CalculiX 2.22 / SPOOLES 共形成 15 个有效算例，DeepSeek 调用数为 0：

- 把初始增量从 `0.5` 降至 `0.01`，在 `mid_z=0.95` 和留出幅度 `0.75` 上仍均因过多 cutback 失败，因此它不是独立修复；
- 自动 penalty（CalculiX 报告 `1.05e7`）在两个幅度失败，显式 `69000` 与 `1e5` 在两个幅度完成，而 `1e6` 在留出幅度重新失败，因此存在非单调的有限稳定窗口；
- 单独 REMOVE→ADD 且约束不变正常完成；在 ADD 同时释放上块 z 约束会把 12 次增量尝试放大为 33 次，并产生 4 次失败尝试和 81 条 no-convergence，但仍未复现持续约 `1e-30` 的不推进循环。

合并状态仍为 `narrowed_unresolved`。当前最高价值下一步是建立本地可执行的圆柱曲面 C3D20R、surface-to-surface 缩减模型，同时保留 midside 曲率、`69000`、REMOVE→ADD 历史和原边界 `OP` 路径。

## Skill 与 schema

这里的 Skill 是应用运行时 JSON，不是 Codex 的 `SKILL.md`：

- [`nonlinear-contact-diagnosis.json`](../../../skills/engineering/nonlinear-contact-diagnosis.json)：只规定证据纪律、竞争假设、单因素实验、正反预测和结果边界，不包含本案例答案或固定工具顺序；
- [`engineering_skill.schema.json`](../../../schemas/engineering_skill.schema.json)：规定应用级工程 Skill 的字段与约束；
- [`problem_manifest.schema.json`](../../../schemas/problem_manifest.schema.json)：规定问题来源、症状、已知事实、混杂因素、运行环境和证据引用的数据契约；
- [`CALCULIX_INTERFERENCE_2747_LOOP.md`](../../../docs/CALCULIX_INTERFERENCE_2747_LOOP.md)：逐项解释 Skill、schema、运行协议和适用边界。

本实验不包含 DQN 训练、强化学习优化或网格优化。

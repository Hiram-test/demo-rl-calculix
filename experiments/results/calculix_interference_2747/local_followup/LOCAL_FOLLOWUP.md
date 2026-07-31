# CalculiX 过盈接触三项本地反事实复核

## 摘要

本轮不调用 DeepSeek、不训练 DQN、不做优化，只用本地 CalculiX 2.22 / SPOOLES 执行三项受控反事实：减小初始增量、改变接触 penalty、以及拆分接触 REMOVE→ADD 与约束路径变化。结果排除了两个过于简单的解释：较小初始增量不能独立修复；penalty 也不是越大越稳定。接触重新激活本身没有复现停滞，但在 ADD 的同时释放约束会把 12 次正常增量尝试放大为 33 次尝试、4 次失败尝试和 81 条 no-convergence。

原问题因此被收窄，但尚未解决。最有价值的下一项实验是圆柱曲面 C3D20R、surface-to-surface 的本地缩减模型，并同时保留原 midside 曲率、`69000`、REMOVE→ADD 历史和边界 `OP` 路径。

## 1. 证据与适用边界

原论坛问题是 [Interference contact and mesh refinement](https://calculix.discourse.group/t/interference-contact-and-mesh-refinement/2747)。附件归档 SHA-256 为 `9AF9CE2F677384BF3AB2E44C188E92E71B47841D9C6C3C67C066DCF5AD8BDB9C`。原输入使用 C3D20R、surface-to-surface、线性 penalty `69000` 并请求 Pardiso；两个大型模型约为 1150 万到 1160 万自由度。

本轮代表模型是两个 C3D20R 方块，接触形式为 node-to-surface，本地后端为 SPOOLES。因此它能反证简单解释、识别数值机制和候选 workaround，但不能冒充原圆柱孔模型的精确复现或确认修复。

本地求解器可执行文件 SHA-256 为 `612144A1441D2A32C33325AA9562C2FF79F60F466E306F7F2680AB76EEEA3961`。本轮 DeepSeek 调用数为 `0`。

## 2. 决策记录方法

每项实验均记录：

1. 求解前已有证据与竞争解释；
2. 唯一改变的字段和保持不变的字段；
3. 正结果、负结果和混合结果各自允许得出的结论；
4. 真实 solve ID、完成标志、cutback、no-convergence、增量尝试、残差与修正量；
5. 结果如何更新原假设。

这里的“决策路径”是可审计的工程记录，不是对模型内部隐藏推理的声称。机器可读版本见 `local_engineer_trace.json`，逐步文字版见 `LOCAL_ENGINEER_DECISIONS.md`。

## 3. D1：较小初始增量

单因素对照只把初始增量从 `0.5` 改为 `0.01`；每对 deck 仅第 73 行不同。两个幅度均保留 C3D20R、node-to-surface、自动 penalty、拓扑、材料、载荷和边界。

| mid_z | 初始增量 | solve ID | 完成 | no-convergence | 最大残差 | 最大修正 |
|---:|---:|---|---|---:|---:|---:|
| 0.95 | 0.5 | SOLVE-DDCAEE99F171 | 否 | 68 | 5.004e25 | 1.837e11 |
| 0.95 | 0.01 | SOLVE-DF3E3150B9A4 | 否 | 71 | 2.658e15 | 2.426e9 |
| 0.75 | 0.5 | SOLVE-847F780B753C | 否 | 15 | 1.479e28 | 5.825e11 |
| 0.75 | 0.01 | SOLVE-2960D1A3BB04 | 否 | 12 | 1.477e28 | 3.774e7 |

两个 `0.01` 变体都因过多 cutback 终止，且没有最终位移/反力块。较小增量虽然改变某些数值量级，但“较小初始增量是独立修复”已被反证。

## 4. D2：penalty 稳定窗口

单因素对照只改变线性 penalty。CalculiX 自动值报告为 `1.05e7`，显式比较 `69000`、`1e5` 与 `1e6`。同一 `mid_z` 内其余字段保持不变。

| mid_z | penalty | solve ID | 完成 | no-convergence | 最大修正 | 底面 RFz |
|---:|---:|---|---|---:|---:|---:|
| 0.95 | 自动 1.05e7 | SOLVE-DDCAEE99F171 | 否 | 68 | 1.837e11 | - |
| 0.95 | 69000 | SOLVE-2B6021325778 | 是 | 3 | 0.02373 | 25.000002 |
| 0.95 | 1e5 | SOLVE-C534AD4EF3AF | 是 | 3 | 0.02397 | 24.9999992 |
| 0.95 | 1e6 | SOLVE-BD438D78DF3B | 是 | 5 | 0.02643 | 24.999998 |
| 0.75 | 自动 1.05e7 | SOLVE-847F780B753C | 否 | 15 | 5.825e11 | - |
| 0.75 | 69000 | SOLVE-D761175D118C | 是 | 6 | 0.12466 | 24.9999996 |
| 0.75 | 1e5 | SOLVE-1B5D6CAA602B | 是 | 7 | 0.12603 | 24.999998 |
| 0.75 | 1e6 | SOLVE-8563A910EE08 | 否 | 14 | 1.065e11 | - |

`69000` 与 `1e5` 在搜索幅度和留出幅度都完成，所有完成算例的全模型 z 向不平衡量绝对值不超过 `2.0e-6`。`1e6` 却在留出幅度重新失败，证明稳定性不是刚度的单调函数。

这支持“代表模型存在有限 penalty 数值稳定窗口”，但不能解释原帖：原 deck 本来就使用 `69000`，而且是 surface-to-surface。一次早期调用曾使用无效的带连字符 contact type；它在修正前已隔离，未纳入八个有效样本。

## 5. D3：重新激活与约束路径

三个算例使用同一 40 节点、两个 C3D20R 单元、两步 NLGEOM 静力模型：

- A：接触两步持续激活，上块 U3 固定；
- B：第一步 REMOVE、第二步 ADD，上块 U3 保持固定；
- C：第一步 REMOVE、第二步 ADD，并在第二步同时释放上块 U3。

| 案例 | 完成 | 增量尝试 | 失败尝试 | no-convergence | 最大残差 | 最大修正 | 1e-30 停滞 |
|---|---|---:|---:|---:|---:|---:|---|
| A 持续激活 | 是 | 12 | 0 | 14 | 5.432 | 0.004353 | 否 |
| B REMOVE→ADD | 是 | 12 | 0 | 13 | 3.438 | 0.003216 | 否 |
| C ADD + 释放 U3 | 是 | 33 | 4 | 81 | 7.828e8 | 9.046e4 | 否 |

B 第一步的确出现 12 次精确零残差和零修正，但每个增量都在第二次迭代收敛并前进，因此是无载荷平凡平衡，不是不推进循环。C 则发生明显切步和收敛恶化，但最终仍完成；三案均无正修正量小于等于 `1e-25`。

因此 REMOVE→ADD 本身不是该异常的充分条件；重新激活与约束路径同步变化是强放大器，但原帖近 `1e-30` 停滞仍未复现。

## 6. 合并判断

当前状态为 `narrowed_unresolved`：

- 已反证：较小初始增量可以单独修复；
- 已反证：penalty 越大越稳定；
- 已反证：在当前代表模型中 REMOVE→ADD 本身足以产生停滞；
- 已支持：代表 node-to-surface 模型有有限 penalty 稳定窗口；
- 已支持：接触状态切换与约束路径变化的耦合会显著放大收敛困难；
- 尚未确认：原圆柱 surface-to-surface 模型的根因与修复；
- 尚未复现：持续约 `1e-30` 修正且不推进的循环。

下一项实验应使用可本地运行的圆柱曲面 C3D20R surface-to-surface 缩减模型。控制对照应分别改变曲面 midside 离散、REMOVE→ADD 历史和边界 `OP`，而不是继续无边界地扫描参数。

## 7. 可复核工件

- `LOCAL_ENGINEER_DECISIONS.md`：逐步决策日志；
- `local_engineer_trace.json`：机器可读决策路径；
- `local_followup_provenance.json`：论文、trace、脚本和原始证据的哈希及逐页检查收据；
- `evidence/`：三项汇总、单因素比较和求解收据；
- `raw/increment/solver_runs/`：四个增量对照的输入及原始结果；
- `raw/penalty/solver_runs/`：八个有效 penalty 对照的输入及原始结果；
- `raw/activation/`：A/B/C 三案输入及原始结果。

四个主汇总文件的 SHA-256 已写入 `local_engineer_trace.json`。原始输入和 console、STA、CVG、DAT、FRD 的单文件哈希保存在各自 summary 中。

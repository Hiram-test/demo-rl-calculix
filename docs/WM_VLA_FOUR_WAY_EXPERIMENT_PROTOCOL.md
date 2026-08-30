# WM-VLA 四方竞争实验与策略改进冻结方案

**协议编号：** WMVLA-4WAY-P1  
**起始提交：** `692c4229f85fae07340c87d52323141e8928520e`  
**关联 PR：** #40  
**实验对象：** 三维钢箱梁横隔板构件上的 WM-VLA、局部预测、监督学习、区域图 RL 与 Dörfler。

本协议要求先冻结当前 WM-VLA V0，完成四方盲测，再根据诊断结果决定 V1 的唯一策略改进。执行者不得边看测试结果边调策略并继续把同一测试称为盲测。

## 1. 唯一主问题

WM-VLA 是否能在同一组三维钢箱梁横隔板盲测工况上同时打赢：

1. `local_prediction`；
2. Dörfler 专家网格监督尺寸场 `supervised`；
3. 三个独立训练种子的冻结区域图 Double DQN，以逐工况中位数记为 `RL-median`。

Dörfler 只承担安全底线和渐近参照。通过 Dörfler 门不等于实验成功。

总成功条件：

\[
G_{safe}\land G_{LP}\land G_{SUP}\land G_{RL}\land G_{WM\ mechanism}.
\]

任一项失败，`OVERALL_WIN=false`。

## 2. 方法独立性

WM-VLA 不得读取、导入或复制：

- `predicted_sizes()` 或局部预测尺寸场；
- 监督网络输出；
- RL 动作或 Q 值；
- PSO、Nelder-Mead 或连续 LLM 参数搜索；
- 测试参考解、真实误差或未来 Dörfler 轨迹。

比较器只能在各方法全部执行完成后读取结果。

主实验采用冻结的确定性语义分区，以隔离世界模型策略。每个工况只生成一次 `partition_spec.json`。WM-VLA 与 RL 必须读取同一份区域名称、顺序、多边形或剖切、元素归属、邻接图和背景区域。不得让 WM 使用 drawn partition、RL 使用另一套 geodesic partition。

真实 VLM 只作为补充端到端实验，每个工况最多调用一次并缓存，不能在网格循环中反复调参数。

## 3. 公共有限元合同

所有方法共享：

- 同一 `Problem` 和参数；
- 同一均匀 `h0` 探针；
- 同一 Gmsh 和 CalculiX 版本；
- 同一 ZZ 指标；
- 同一材料、荷载、约束和 QoI；
- 同一参考解；
- 同一真实 CalculiX 求解计数；
- 同一有效方程数预算；
- 同一停止上限。

参考求解不计入在线求解次数，但单独报告成本。

## 4. 参数化钢箱梁横隔板家族

保持现有 `make_box_girder_diaphragm()` 拓扑：顶板、底板、两道腹板、横隔板、圆形检修孔及开口框、偏心轮载和两个支承区。

使用六维 maximin Latin hypercube：

| 参数 | 范围 |
|---|---:|
| `wheel_offset_x` | `[-140, 140]` mm |
| `wheel_offset_y` | `[-75, 75]` mm |
| `opening_radius` | `[48, 76]` mm |
| `diaphragm_thickness` | `[24, 40]` mm |
| `pressure` | `[2.8, 6.0]` MPa |
| `support_width` | `[55, 90]` mm |

生成 48 个工况：训练 24、验证 8、首轮盲测 16。manifest 种子固定为 `20260830`。生成后立即保存 `case_manifest.json`、SHA-256、每个 case_id、split、参数和几何哈希。不得删除困难或不利测试工况。

## 5. 参考解

每个工况建立两级独立参考 `reference_A` 和更细的 `reference_B`。要求：

\[
|U_B-U_A|/|U_B|\le0.5\%,\qquad |Q_B-Q_A|/|Q_B|\le0.5\%.
\]

未通过时只允许继续细化参考，不得修改被比较方法。最终误差统一相对 `reference_B`。

## 6. 冻结部署合同

### 6.1 WM-VLA V0

首轮使用 PR #40 当前策略，不先修改决策逻辑：

- 公共均匀探针；
- 每轮精确 Dörfler 标记；
- 世界模型只在 Dörfler 基础上提出额外区域深度；
- receding-horizon，每轮只执行第一步；
- MCP 形态工具层完成动作验证、尺寸物化、Gmsh 预网格、有效方程数、哈希和回退；
- 测试工况之间不共享在线新转移，避免测试顺序泄漏。

`frozen_config.json` 至少保存 horizon、beam_width、最大附加区域数、最大深度、最小预测增益、不确定性门、失败概率门、预算安全系数、回归门和 ensemble 参数。

### 6.2 局部预测

使用当前强版本：逐单元误差等分配、每轮新 ZZ 指标、按公共方程预算换算目标单元数，并运行到相同真实求解上限。不得用 `p=1` 的故障诊断版本作为主基线。

### 6.3 监督学习

- 标签只来自 24 个训练工况的 Dörfler 专家网格；
- 训练 3 个独立网络种子；
- 只在验证集选择冻结网络；
- 测试部署固定为探针求解加一次预测重网格，共 2 次真实求解；
- `k>2` 使用 hold-last；
- 不准看测试误差调预算缩放。

报告专家求解数、训练墙钟、样本数、模型参数量和模型哈希。

### 6.4 RL

- 与 WM-VLA 共用 `partition_spec.json`；
- 训练集与 WM/监督相同；
- 训练 3 个独立种子，每种子 300 个完整 episode；
- 每 25 个 episode 在验证集评估；
- 每个种子按预注册验证指标选择 checkpoint；
- 测试时冻结 greedy policy；
- 每个 `(case,K,B)` 对三个种子逐点取中位数，不得选择最佳测试种子；
- 每个区域动作计一次真实 Gmsh 重网格和 CalculiX 求解。

### 6.5 Dörfler

固定 `theta=0.50`，精确逐单元 bulk marking，相同 Gmsh、求解上限和方程预算。它是安全底线，不属于主要胜负。

## 7. 实现和冻结顺序

执行者先补齐统一 harness，不改 WM-VLA V0 策略。新增或等价实现：

```text
scripts/make_bridge_case_manifest.py
scripts/train_bridge_supervised.py
scripts/train_bridge_rl.py
scripts/run_four_way_bridge_benchmark.py
scripts/analyze_four_way_bridge.py
```

所有脚本必须从 manifest 读取工况。

训练顺序：

1. 24 个训练工况建立 WM 转移库；
2. 同一 24 个工况生成监督专家数据并训练 3 个网络；
3. 同一 24 个工况训练 3 个 RL 种子；
4. 只用 8 个验证工况选择 checkpoint 或检查稳定性；
5. 测试前提交冻结 commit。

WM 转移库首轮固定：每训练工况 6 次真实求解、`budget=120000`，不使用任何竞争方法轨迹作为标签。

冻结 commit 必须包含 manifest 和哈希、全部模型和 SHA-256、`frozen_config.json`、训练/验证结果、环境锁定和 `TEST_NOT_RUN=true`。冻结后任何影响动作或评分的修复都使原测试作废。

## 8. 在线预算与交付误差

真实求解次数：

\[
K\in\{2,3,4,6\}.
\]

有效方程数预算：

\[
B\in\{30000,60000,120000\}.
\]

每个方法在每个预算下独立运行到 6 次，再用真实前缀构造不同 K。不得为每个 K 重新调参。

定义：

\[
E_m(c;K,B)=\min_{j\le K,\,N_j\le B}e_{E,m}^{(j)}.
\]

没有预算内解即失败。主要运行点固定为：

\[
\mathcal G=\{(2,30000),(2,60000),(3,60000),(4,60000),(4,120000),(6,120000)\}.
\]

完整 12 点网格也必须保存，但主门只使用上述六点。

## 9. 三个主要胜负门

对竞争方法 `b in {LP,SUP,RL}`：

\[
r_b(c,g)=E_{WM}(c;g)/E_b(c;g),
\]

\[
R_b=\exp\left[\frac{1}{16|\mathcal G|}\sum_{c,g}\log r_b(c,g)\right].
\]

WM-VLA 只有同时满足以下条件，才算打赢方法 b：

1. `R_b <= 0.95`；
2. 以工况为重采样单位的配对 bootstrap 95% 置信区间上界 `< 1.00`；
3. 全部 `case × operating point` 的胜率不少于 60%；
4. 比值 95 分位数不超过 1.15；
5. 聚合 QoI 误差比不超过 1.05；
6. WM 预算违规为 0；
7. 至少 75% 测试工况执行过一次被认证的 proactive world action。

输出：

```text
BEAT_LOCAL_PREDICTION = true/false
BEAT_SUPERVISED       = true/false
BEAT_RL               = true/false
OVERALL_WIN           = 三项逻辑与，并同时要求安全门和机制门通过
```

## 10. Dörfler 安全门

每个实际 WM 目标必须对前一网格所有节点满足：

\[
h_{WM}(x_i)\le h_D(x_i).
\]

实测还要求：聚合能量误差比不超过 1.02；任何主要运行点不超过 1.15；预算违规为 0；失信、超预算和低置信动作确实回退 Dörfler。

## 11. 时间和训练成本

分别记录 VLM/分区、世界模型、参数工具、Gmsh、CalculiX、总在线时间、离线训练时间和训练求解数。

WM 工程效率门：

- 每个工况 VLM 调用不超过 1 次；
- 非求解器开销中位数不超过总在线时间 15%；
- 相对局部预测的总在线墙钟聚合比不超过 1.25。

监督和 RL 的离线成本另算，并报告 `T_train + n*T_online` 的摊销交叉点。

## 12. 必须执行的机制消融和上界

在 `B=60000, K=6` 上运行全部 16 个测试工况：

- `WM-full`：完整策略；
- `WM-h1`：horizon=1；
- `WM-prior-only`：禁用 residual ensemble；
- `WM-no-history`：去掉 hit_count 和历史转移；
- `random-safe-extra`：同一安全动作集内随机附加，5 个随机种子；
- `oracle-future-hit`：事后读取完整 Dörfler 未来命中深度形成不可部署上界。

Oracle 回答当前场景和动作空间是否客观存在可压缩未来反馈。若 oracle 仍打不过局部预测，应优先修改区域粒度、动作空间或场景机制，而不是只调模型。

## 13. 世界模型预测诊断

每步保存：区域误差与资源预测、总误差、方程数、不确定性、失败概率、候选排序、执行动作、真实转移和回退原因。

汇总：

- 总误差 log-MAE；
- 方程数 MAPE；
- 区间覆盖率；
- 候选预测排序与真实收益排序的 Spearman 相关；
- proactive 接受率；
- 接受后真实改善率；
- 各类回退次数。

## 14. 输出和交回材料

建议目录：

```text
results/wm_vla_four_way_p1/
├── protocol/
├── training/
├── references/<case_id>/
├── test/<case_id>/<budget>/<method>/
├── ablations/<case_id>/
├── aggregate/
├── figures/
└── EXECUTION_REPORT.md
```

至少交付：最终分支/PR/head SHA、manifest 和哈希、frozen config、全部模型和训练日志、全部 SolveRecord、网格和动作哈希、CalculiX 日志、`primary_results.csv`、`pairwise_ratios.csv`、`bootstrap.json`、`prediction_calibration.csv`、`failure_matrix.csv`、`final_gate.json`、六个消融和 oracle 上界。

`EXECUTION_REPORT.md` 首页必须先写：

```text
DORFLER_SAFE            = true/false
BEAT_LOCAL_PREDICTION   = true/false
BEAT_SUPERVISED         = true/false
BEAT_RL                 = true/false
WORLD_MODEL_MECHANISM   = true/false
ONLINE_TIME_ACCEPTABLE  = true/false
OVERALL_WIN             = true/false
```

不得以“总体良好”替代明确失败门。

## 15. 结果返回后的策略诊断

### A. Oracle 也输给局部预测

优先判断区域过粗、只允许当前 Dörfler 支持内加深、Gmsh gradation 浪费资源或场景热点基本静止。V1 可选唯一方向：core/halo/transition、动态子区、区内分层深度、受约束邻域前瞻。不得复制局部预测尺寸输出。

### B. Oracle 能赢，WM-full 不能赢

说明动作空间有余量，主要问题在模型。V1 可选唯一方向：训练集安全 off-policy 动作覆盖、区域边特征、误差/资源分头建模、不确定性校准、训练轨迹上的未来命中概率辅助任务。

### C. `WM-prior-only` 与 `WM-full` 相同

残差学习没有贡献。优先改训练覆盖、特征或模型结构，不能只增大 horizon。

### D. `WM-h1` 与 `WM-full` 相同

多步规划没有贡献。检查 rollout 是否真实演化、terminal value、beam 多样性、预算惩罚和不确定性累积。V1 可选 terminal value、剩余预算/步数状态、多样性约束或 adaptive horizon。

### E. 赢监督和 RL，但输局部预测

最可能是区域分辨率不足。V1 优先层级区域、core/halo 编译、动态子区和更准确的 Gmsh 资源预测。

### F. 赢局部预测和监督，但输 RL

比较 RL 状态、访问分布、停止动作以及 WM 评分与 Q 值分歧。V1 可增加独立价值头或改善停止价值，不能读取测试 RL 动作训练 WM。

### G. 大部分动作退回 Dörfler

按不确定性、预算、低增益和真实失信分别处理。不得简单放宽全部门槛制造 proactive action。

### H. 准确率赢但时间输

优先向量化 rollout、复用动作无关状态、adaptive beam/horizon、只对最终少数候选生成真实 Gmsh 预网格并缓存 VLM。

## 16. V1 改进规则

首轮测试查看后，原测试集转为已披露诊断集。V1 必须只修改一类主机制，使用原训练/验证集开发，并用 manifest 种子 `20260930` 生成新的 16 个盲测工况。V1 复用同一基线和统计门，同时报告 V0、V1 和全部基线。

若同时修改区域粒度、世界模型、规划器、预算门和场景，则无法归因，判为无效改进。

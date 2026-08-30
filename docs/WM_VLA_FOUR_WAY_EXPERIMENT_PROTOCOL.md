# WM-VLA（VRA）四方竞争实验与策略改进冻结方案

**协议编号：** WMVLA-4WAY-P1  
**适用仓库：** `Hiram-test/demo-rl-calculix`  
**起始分支：** `feat/world-model-vla-bridge-diaphragm-ready-r1-20260830`  
**起始提交：** `692c4229f85fae07340c87d52323141e8928520e`  
**关联 PR：** #40  
**实验对象：** 三维钢箱梁横隔板构件上的 WM-VLA、局部预测、监督学习、区域图 RL 与 Dörfler  
**协议性质：** 先冻结、后执行、再审查；执行者不得在看到盲测结果后现场修改 WM-VLA 策略并重新宣称同一轮实验成功。

> 下文统一使用仓库中的名称 **WM-VLA**。本协议的首轮任务是测清当前策略能否赢，而不是边跑边改。首轮结果交回审查后，才进入策略改进轮。

---

## 1. 实验要回答的唯一主问题

在同一组三维钢箱梁横隔板盲测工况上，WM-VLA 是否能够同时打赢：

1. 逐单元局部预测 `local_prediction`；
2. 基于 Dörfler 专家网格训练的监督尺寸场 `supervised`；
3. 三个独立训练种子的冻结区域图 Double DQN，以逐工况中位数作为 `RL-median`。

Dörfler 只承担安全底线和渐近参照，不属于“主要被击败对象”。

总成功条件固定为：

\[
\boxed{
G_{\mathrm{safe}}
\land
G_{\mathrm{LP}}
\land
G_{\mathrm{SUP}}
\land
G_{\mathrm{RL}}
\land
G_{\mathrm{WM\ mechanism}}
}
\]

其中：

- \(G_{\mathrm{safe}}\)：WM-VLA 不破坏 Dörfler 尺寸场底线，且实测不出现不可接受的 Dörfler 回归；
- \(G_{\mathrm{LP}}\)：WM-VLA 打赢局部预测；
- \(G_{\mathrm{SUP}}\)：WM-VLA 打赢监督学习；
- \(G_{\mathrm{RL}}\)：WM-VLA 打赢 RL 中位策略；
- \(G_{\mathrm{WM\ mechanism}}\)：收益确实来自世界模型的未来推演，而不是纯 Dörfler 回退、额外求解次数或首网优势。

任何一项失败，`OVERALL_WIN=false`。

---

## 2. 首轮实验的科学边界

### 2.1 方法必须独立

WM-VLA 运行路径不得读取、导入或复制：

- `predicted_sizes()` 或局部预测尺寸场；
- 监督网络输出；
- RL 动作或 Q 值；
- PSO、Nelder–Mead 或连续 LLM 参数搜索；
- 盲测参考解、真实误差或未来 Dörfler 轨迹。

局部预测、监督学习和 RL 只能在全部方法执行结束后，由汇总器读取结果进行比较。

### 2.2 视觉部分只调用一次

主实验采用冻结的确定性语义分区，目的是隔离“世界模型策略”本身。每个几何实例只生成一次分区并缓存为 `partition_spec.json`，后续所有真实求解均复用该分区。

WM-VLA 与 RL 必须读取**完全相同**的 `partition_spec.json`，包括：

- 区域名称；
- 区域顺序；
- 多边形或剖切定义；
- 元素归属规则；
- 邻接图；
- 背景区域定义。

不得让 WM-VLA 使用 drawn partition，而 RL 使用另一套 geodesic partition。

真实 VLM 只作为第二阶段的端到端补充实验，不进入首轮主结论。真实 VLM 补充实验也只能每个工况调用一次并缓存，不得在网格循环中重复询问模型。

### 2.3 相同有限元合同

所有方法必须共享：

- 同一 `Problem` 对象和参数；
- 同一初始均匀网格尺度 `h0`；
- 同一 Gmsh 版本、算法和单线程设置；
- 同一 CalculiX 版本和求解设置；
- 同一 ZZ 指标实现；
- 同一材料、荷载、约束和 QoI；
- 同一参考解；
- 同一真实 CalculiX 求解计数规则；
- 同一有效方程数预算；
- 同一停止上限。

参考解不计入在线求解次数，但参考构造成本单独记录。

---

## 3. 三维桥梁构件与参数化工况

采用现有 `make_box_girder_diaphragm()`，保持拓扑不变：

- 顶板、底板；
- 两道纵向腹板；
- 横隔板；
- 横隔板圆形检修孔及开口框；
- 顶板偏心轮载；
- 一端固定型支承区和一端滚动型支承区。

主要竞争机制为：

\[
\text{轮载边缘}
+\text{检修孔边缘}
+\text{横隔板—腹板交线}
+\text{支承传力区}
\]

### 3.1 参数空间

在不改变模型拓扑的前提下，使用六维 maximin Latin hypercube 生成工况：

| 参数 | 范围 |
|---|---:|
| `wheel_offset_x` | \([-140,140]\) mm |
| `wheel_offset_y` | \([-75,75]\) mm |
| `opening_radius` | \([48,76]\) mm |
| `diaphragm_thickness` | \([24,40]\) mm |
| `pressure` | \([2.8,6.0]\) MPa |
| `support_width` | \([55,90]\) mm |

轮载面必须完整位于顶板内，检修孔和开口框不得穿出横隔板。生成脚本必须在写入 manifest 前执行几何可行性检查。

### 3.2 数据划分

共 48 个工况：

- 训练集：24 个；
- 验证集：8 个；
- 首轮盲测集：16 个。

Latin hypercube 随机种子固定为 `20260830`。生成后立即写出：

- `case_manifest.json`；
- `case_manifest.sha256`；
- 每个工况的参数、split、case_id 和几何哈希。

测试集不得因求解困难、结果不利或某一方法失败而被删除。数值失败必须作为结果保留，并按本协议的失败规则计分。

---

## 4. 参考解验证

每个工况先构造两级独立参考：

- `reference_A`：当前强分级参考；
- `reference_B`：在 `reference_A` 基础上进一步减小背景尺度和局部最小尺度，且不得由任何被比较方法的最终网格生成。

参考有效条件：

\[
\frac{|U_B-U_A|}{|U_B|}\le 0.5\%,
\qquad
\frac{|Q_B-Q_A|}{|Q_B|}\le 0.5\%.
\]

若任一工况不满足：

1. 只允许继续细化参考；
2. 不得改被比较方法；
3. 参考收敛后重新计算该工况全部方法的误差；
4. 记录参考升级前后的哈希和差异。

最终误差统一相对 `reference_B` 计算。

---

## 5. 五种方法的冻结部署合同

### 5.1 WM-VLA V0

首轮必须使用 PR #40 当前策略，不得先改成“更可能赢”的版本。

固定行为：

- 第一次求解使用与其他方法相同的均匀探针网格；
- 每轮计算精确 Dörfler 标记；
- 世界模型只在 Dörfler 基础上提出额外区域深度；
- 有限时域滚动规划，每轮只执行第一步；
- MCP 形态工具层负责动作验证、尺寸物化、Gmsh 预网格、有效方程数核算、哈希和回退；
- 世界模型只从训练集和当前工况已经真实执行的历史转移学习；
- 测试工况之间默认不在线共享新转移，避免测试顺序泄漏；如需研究持续部署学习，必须另列实验，不能混入主结果。

首轮默认参数以 `frozen_config.json` 为准，至少写入：

- `horizon`；
- `beam_width`；
- `max_extra_regions`；
- `max_extra_depth`；
- `min_relative_gain`；
- `uncertainty_limit`；
- `failure_limit`；
- `budget_safety`；
- `regression_tolerance`；
- 世界模型 ensemble 数量和 ridge 系数。

### 5.2 局部预测

使用仓库现有逐单元误差等分配实现：

- 与 WM-VLA 相同的均匀探针；
- 每轮重新计算 ZZ 指标；
- 每轮以当前公共方程预算换算目标单元数；
- 允许执行到相同真实求解上限；
- 不使用 WM-VLA 的分区、状态或动作。

不得只跑一轮后称其为完整局部预测，也不得故意使用已知较弱的 `p=1` 诊断版本作为主基线。

### 5.3 监督学习

使用现有 Dörfler 专家网格监督尺寸场路线：

- 训练标签只由训练集上的 Dörfler 专家轨迹生成；
- 训练 3 个独立网络种子；
- 根据验证集表现选择一个冻结网络，选择规则在测试前写入 `frozen_config.json`；
- 测试部署严格为“均匀探针求解 + 一次预测重网格求解”，共 2 次真实求解；
- 在 \(k>2\) 的比较中采用 hold-last，不伪造额外求解收益；
- 每个方程预算单独进行确定性预算缩放，但不得看测试参考误差调缩放系数。

训练成本必须报告：专家 CalculiX 求解数、网络训练墙钟、训练样本数和模型参数量。

### 5.4 RL

使用现有区域图 Double DQN，并做以下公平修正：

- 与 WM-VLA 读取同一 `partition_spec.json`；
- 训练集与 WM/监督完全相同；
- 训练 3 个独立种子；
- 每个种子 300 个完整 episode，除非出现明确的数值故障；
- 每 25 个 episode 在验证集评估一次；
- 每个种子按验证集预注册指标选择 checkpoint；
- 测试时冻结，使用 greedy policy；
- 测试结果按每个工况、每个 \((k,B)\) 对三个种子取中位数，不能选最好种子；
- 每个 region-refine 动作均计一次真实 Gmsh 重网格和 CalculiX 求解。

训练成本必须报告：episode 数、真实训练求解数、梯度更新数、训练墙钟和三个种子的离散程度。

### 5.5 Dörfler

Dörfler 使用：

- `theta=0.50`；
- 每轮精确逐单元 bulk marking；
- 与其他方法相同的 Gmsh 重网格；
- 相同求解上限和方程预算。

Dörfler 是安全底线，不计入三项主要胜负。

---

## 6. 训练和冻结顺序

### 阶段 A：实现验收

执行者先补齐统一四方 harness，但不能修改 WM-VLA 决策逻辑。必须新增或等价实现：

```text
scripts/make_bridge_case_manifest.py
scripts/train_bridge_supervised.py
scripts/train_bridge_rl.py
scripts/run_four_way_bridge_benchmark.py
scripts/analyze_four_way_bridge.py
```

所有脚本必须从 manifest 读取工况，不得在脚本内部另写一套随机采样。

### 阶段 B：训练

1. 用 24 个训练工况建立 WM 转移库；
2. 用同一 24 个训练工况生成监督专家数据并训练 3 个网络；
3. 用同一 24 个训练工况训练 3 个 RL 种子；
4. 所有模型只在 8 个验证工况上选择 checkpoint 或检查数值稳定性；
5. 禁止访问 16 个测试工况的任何参考误差或方法结果。

WM 转移库首轮建议固定执行：

- 每个训练工况 6 次真实求解；
- `budget=120000`；
- acquisition 阶段允许当前代码已有的安全探索设置；
- 不得使用局部预测、监督或 RL 轨迹作为转移标签。

### 阶段 C：冻结

在测试前必须提交一个单独 Git commit，包含：

- `case_manifest.json` 及哈希；
- 所有模型文件及 SHA-256；
- `frozen_config.json`；
- 训练和验证结果；
- 当前代码 commit SHA；
- 环境锁定文件；
- 明确文字：`TEST_NOT_RUN=true`。

该 commit 之后，除修复导致程序无法执行的机械错误外，不得修改策略、超参数、训练数据或统计门。任何影响动作或评分的修复都使原测试作废，必须重新冻结后再运行。

### 阶段 D：盲测

冻结后一次性执行 16 个测试工况。测试顺序按 `case_id` 排序；不得根据中间结果中断、重排或调整配置。

---

## 7. 公共在线预算

真实 CalculiX 求解次数取：

\[
K\in\{2,3,4,6\}.
\]

有效方程数预算取：

\[
B\in\{30000,60000,120000\}.
\]

每个方法在每个预算下独立运行到 6 次求解，随后从真实前缀构造 \(K=2,3,4,6\) 的交付结果。不得为每个 \(K\) 重新调参数。

对方法 \(m\)、工况 \(c\)、求解上限 \(K\) 和预算 \(B\)，定义：

\[
E_m(c;K,B)
=
\min_{j\le K,\;N_j\le B} e_{E,m}^{(j)}.
\]

QoI 同理：

\[
Q_m(c;K,B)
=
\min_{j\le K,\;N_j\le B} e_{Q,m}^{(j)}.
\]

若一个方法在给定 \((K,B)\) 下没有任何预算内解，则记为失败，不得用超预算解替代。

主要运行点固定为：

\[
\mathcal G=
\{(2,30000),(2,60000),(3,60000),(4,60000),(4,120000),(6,120000)\}.
\]

完整 12 点网格仍需保存，但主要统计门只使用上述六个预注册运行点，防止事后挑选有利预算。

---

## 8. 主要统计指标与胜负门

对每个竞争方法 \(b\in\{LP,SUP,RL\}\)，定义配对能量误差比：

\[
r_b(c,g)=\frac{E_{WM}(c;g)}{E_b(c;g)},
\qquad g\in\mathcal G.
\]

聚合比值为：

\[
R_b=
\exp\left[
\frac{1}{16|\mathcal G|}
\sum_{c,g}\log r_b(c,g)
\right].
\]

WM-VLA 只有同时满足以下条件，才算打赢方法 \(b\)：

1. \(R_b\le0.95\)，即聚合能量误差至少降低 5%；
2. 以工况为重采样单位的配对 bootstrap 95% 置信区间上界小于 1.00；
3. 在全部 `case × operating point` 中，\(r_b<1\) 的比例不少于 60%；
4. \(r_b\) 的 95 分位数不超过 1.15，避免用少数大胜掩盖严重退化；
5. 聚合 QoI 误差比不超过 1.05；
6. WM-VLA 预算违规数为 0；
7. 至少 75% 测试工况真正执行过一次被认证的 world-model proactive action。

RL 的 \(E_b\) 和 \(Q_b\) 是三个冻结策略在每个工况和运行点上的中位数，而不是最好种子。

总胜负：

```text
BEAT_LOCAL_PREDICTION = true/false
BEAT_SUPERVISED       = true/false
BEAT_RL               = true/false
OVERALL_WIN           = 三项逻辑与
```

---

## 9. Dörfler 安全门

结构硬门：每一个实际执行的 WM-VLA 目标尺寸场必须满足：

\[
h_{WM}(x_i)\le h_D(x_i)
\]

对所有前一网格节点成立，并输出逐步验证结果和哈希。

实测安全门：相对独立 Dörfler，要求：

- 聚合能量误差比不超过 1.02；
- 任一工况主要运行点的误差比不超过 1.15；
- 预算违规为 0；
- 世界模型拒绝、失信或超预算时确实执行 Dörfler 回退。

通过 Dörfler 门只说明安全，不构成主要成功。

---

## 10. 时间与训练成本

每种方法都要分开记录：

- VLM/语义分区时间；
- 世界模型推演时间；
- 参数工具时间；
- Gmsh 重网格时间；
- CalculiX 时间；
- 总在线时间；
- 离线训练时间和训练求解次数。

WM-VLA 的工程效率门：

- 每个测试工况 VLM 调用不超过 1 次；
- 非求解器开销中位数不超过总在线时间的 15%；
- 与局部预测相比，总在线墙钟聚合比不超过 1.25。

监督和 RL 的离线成本不塞进单工况在线时间，但必须计算随部署工况数 \(n\) 的摊销成本：

\[
T_m(n)=T_{m,train}+nT_{m,online}.
\]

报告 WM-VLA 相对监督和 RL 的成本交叉点，不得只报推理速度。

---

## 11. 必须执行的机制消融与上界

这些诊断在 `B=60000, K=6` 上运行全部 16 个测试工况，不参与三项主要胜负，但用于后续策略审查。

### 11.1 `WM-full`

完整当前世界模型、历史状态和多步规划。

### 11.2 `WM-h1`

规划 horizon 固定为 1，其他完全相同。用于判断多步未来推演是否真正有价值。

### 11.3 `WM-prior-only`

禁用在线 residual ensemble，只保留解析网格演化先验。用于判断学习到的转移残差是否有价值。

### 11.4 `WM-no-history`

去掉 `hit_count`、持续热点历史和上一步转移残差，其他不变。用于判断“热点持续性”是否是关键状态。

### 11.5 `random-safe-extra`

在满足同一 Dörfler 包含和预算认证的候选中随机选取附加区域，重复 5 个随机种子。用于排除“只要比 Dörfler 多加一点就会赢”的解释。

### 11.6 `oracle-future-hit`

先独立完成完整 Dörfler 轨迹，事后读取未来区域命中深度，构造允许动作空间中的最优未来深度动作。该方法仅是不可部署上界，不进入主要比较。

Oracle 用于回答：

\[
\boxed{
\text{当前场景和动作空间中，是否客观存在可被世界模型压缩的未来反馈？}
}
\]

若 oracle 仍打不过局部预测，则不能把失败归因于“模型还没学好”；应优先检查场景、区域粒度和动作空间。

---

## 12. 世界模型预测质量诊断

每一个真实执行转移都要保存预测和真实结果：

- 区域 \(\Delta\log\eta_i^2\)；
- 区域 \(\Delta\log N_i\)；
- 总误差预测；
- 有效方程数预测；
- 不确定性；
- 失败概率；
- 候选动作排序；
- 最终执行与回退原因。

至少汇总：

- 总误差 log-MAE；
- 方程数 MAPE；
- 预测区间覆盖率；
- 候选动作预测排序与真实收益排序的 Spearman 相关；
- proactive action 接受率；
- 接受后真实改善率；
- 因不确定性、预算、低增益和失信分别触发的回退次数。

这些指标不直接决定主要胜负，但决定下一轮应改世界模型、规划器还是动作编译器。

---

## 13. 输出目录与必交文件

建议固定为：

```text
results/wm_vla_four_way_p1/
├── protocol/
│   ├── case_manifest.json
│   ├── case_manifest.sha256
│   ├── frozen_config.json
│   ├── environment.json
│   └── git_state.json
├── training/
│   ├── world_model/
│   ├── supervised_seed0/
│   ├── supervised_seed1/
│   ├── supervised_seed2/
│   ├── rl_seed0/
│   ├── rl_seed1/
│   ├── rl_seed2/
│   └── training_costs.json
├── references/<case_id>/
├── test/<case_id>/<budget>/<method>/
│   ├── records.json
│   ├── mesh_receipts.json
│   ├── action_log.json
│   └── solver_logs/
├── ablations/<case_id>/
├── aggregate/
│   ├── primary_results.csv
│   ├── pairwise_ratios.csv
│   ├── bootstrap.json
│   ├── prediction_calibration.csv
│   ├── failure_matrix.csv
│   └── final_gate.json
├── figures/
└── EXECUTION_REPORT.md
```

必须同时提交：

- 训练模型文件和 SHA-256；
- 全部原始 `SolveRecord`；
- 每步网格 SHA；
- 每步实际有效方程数；
- CalculiX 返回码和日志；
- 数值失败工况；
- 完整 aggregate JSON/CSV；
- 生成表图的源码；
- GitHub Actions run ID 和 artifact ID。

不得只提交截图、PDF 汇总或人工填写表格。

---

## 14. 执行报告的固定结论格式

`EXECUTION_REPORT.md` 首页只能先给以下机器结果：

```text
DORFLER_SAFE            = true/false
BEAT_LOCAL_PREDICTION   = true/false
BEAT_SUPERVISED         = true/false
BEAT_RL                 = true/false
WORLD_MODEL_MECHANISM   = true/false
ONLINE_TIME_ACCEPTABLE  = true/false
OVERALL_WIN             = true/false
```

其后列出每项失败的具体门，不得使用“总体表现良好”“基本达到目标”等替代明确结论。

---

## 15. 首轮结果交回后的审查逻辑

首轮执行者不得自行大改 WM-VLA。结果交回后，按下面的诊断树决定 V1 只改哪一类机制。

### 情形 A：Oracle 也打不过局部预测

结论：问题主要不在世界模型拟合，而在动作空间或场景没有可压缩的未来反馈。

优先检查：

1. 区域过粗，局部预测的逐单元分辨率具有不可弥补优势；
2. 只允许在当前 Dörfler 支持内追加深度，无法提前处理尚未显影的邻域热点；
3. 区域加密被 Gmsh gradation 扩散，资源大量浪费；
4. 当前构件的热点基本静止，未来推演没有信息价值。

允许的 V1 改进方向：

- 将语义区确定性展开为 `core/halo/transition`；
- 加入受约束的动态子区分裂；
- 将动作从“整区深度”改为“区内核心深度 + 过渡带深度”；
- 在严格风险门下允许相邻候选区的前瞻加密；
- 保持参数由工具编译，不让 LLM 输出连续尺寸。

### 情形 B：Oracle 能赢，但 WM-full 不能赢

结论：场景和动作空间有理论余量，主要问题是世界模型没有恢复未来命中或后果。

优先检查：

- 训练转移覆盖是否过窄；
- 模型是否只学到自身策略产生的同质轨迹；
- action-conditioned 残差是否可辨识；
- 预测不确定性是否失准；
- 是否缺少区域间误差迁移特征；
- 训练与测试状态归一化是否漂移。

允许的 V1 改进方向：

- 在训练集上增加独立、安全、离散的 off-policy 动作采样；
- 增加边特征：界面面积、距离、尺寸梯度、传力方向；
- 分别建模误差迁移和资源迁移；
- 用 conformal 或分位数校准修正 ensemble 风险；
- 将未来 Dörfler 命中概率和累计深度作为辅助任务，但标签只能来自训练轨迹。

### 情形 C：WM-prior-only 与 WM-full 几乎相同

结论：残差学习没有提供信息，当前“世界模型”实际退化为人工解析先验。

重点改进训练覆盖、特征和模型结构；不能通过调大规划 horizon 冒充模型改进。

### 情形 D：WM-h1 与 WM-full 几乎相同

结论：多步规划没有实际贡献。

检查：

- rollout 状态是否真正随动作演化；
- terminal value 是否为空；
- beam 是否只保留同一类动作；
- 预算惩罚是否使所有多步候选退化成一步贪心；
- 模型不确定性是否随 horizon 正确累积。

允许的 V1 改进方向：

- 加入从训练轨迹学习的 terminal value；
- 使用预算剩余和未来求解次数作为显式状态；
- 对候选序列增加多样性约束；
- 使用 adaptive horizon，而不是无条件增大 horizon。

### 情形 E：能赢监督和 RL，但输给局部预测

结论：WM-VLA 已优于学习基线，但区域分辨率仍不足，这是最可能的首轮结果。

优先改进：

- 区域层级化；
- 区内 core/halo 编译；
- 动态局部子区；
- 更准确的 Gmsh 资源预测；
- 保持局部预测完全独立，不能直接复制其尺寸公式或输出。

### 情形 F：能赢局部预测和监督，但输给 RL

结论：RL 在相同区域动作空间中学到了更好的长期价值或停止策略。

优先比较：

- RL 的状态变量与 WM 状态变量；
- RL 的实际区域访问分布；
- RL 停止动作；
- WM 规划评分与 RL Q 值的分歧。

允许的 V1 改进方向：

- 为世界模型增加独立价值头；
- 用模型预测进行离线 planning-policy 蒸馏，但不得读取测试 RL 动作；
- 改善停止价值和剩余预算价值估计。

### 情形 G：大部分动作回退到 Dörfler

结论：方法虽然安全，但没有证明世界模型价值。

根据回退原因区分：

- 不确定性过高：增加训练覆盖或校准；
- 预算预检失败：改工具层资源模型；
- 预测增益太低：检查评分函数和尺度；
- 执行后失信：提高风险惩罚并改善转移模型。

不得简单放宽全部门槛以制造 proactive action。

### 情形 H：准确率能赢，但在线时间输得明显

优先改进：

- 批量向量化候选 rollout；
- 复用动作无关状态；
- adaptive beam；
- adaptive horizon；
- 只对最终少数候选执行真实 Gmsh 预网格；
- 缓存一次性 VLM 结果。

不得用不计入总时间的方式隐藏 MCP 或 Gmsh 认证开销。

---

## 16. V1 策略改进规则

首轮测试一旦查看，原 16 个测试工况即转为“已披露诊断集”，不能继续作为 V1 的最终盲测集。

V1 必须：

1. 只选择上节诊断树指出的一类主机制修改；
2. 在 Git diff 中明确列出策略变化；
3. 使用原训练集和验证集进行开发；
4. 重新训练受影响的模型；
5. 使用新的 16 个盲测工况，manifest 种子固定为 `20260930`；
6. 复用完全相同的四方基线和统计门；
7. 同时报告 V0、V1 和所有基线，不能只报 V1。

若 V1 同时修改区域粒度、世界模型结构、规划器、预算门和场景，实验无法归因，判为无效改进。

---

## 17. 交回审查时的最小材料

执行者完成后，至少交付：

1. GitHub 分支、PR 和最终 head SHA；
2. `frozen_config.json`；
3. `case_manifest.json` 及哈希；
4. `final_gate.json`；
5. `primary_results.csv`；
6. `pairwise_ratios.csv`；
7. `failure_matrix.csv`；
8. `prediction_calibration.csv`；
9. 全部模型和训练日志；
10. 全部测试原始记录与 solver logs；
11. 六个机制消融和 oracle 上界；
12. 失败工况的最终网格、误差分布、动作序列及预测—真实差异。

审查重点不是看一张总表，而是回答：

\[
\boxed{
\text{WM-VLA 为什么赢或为什么输，失败发生在区域、模型、规划、工具还是时间成本哪一层？}
}
\]

只有在这一层归因完成后，才决定 V1 的具体策略修改。

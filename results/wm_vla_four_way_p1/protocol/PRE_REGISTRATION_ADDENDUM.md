# WMVLA-4WAY-P1 执行前补充冻结

本补充文件只消除原协议中会影响实现或评分、但未给出唯一数值定义的空白。它在任何首轮测试工况的方法结果或参考误差被读取之前写入。原协议已按附件全文原样同步，正文优先；本文件不得在盲测后修改。

## 唯一 V0 与工况实现

- V0 控制器唯一绑定 `visionamr.vla.world.{model,planner,tool_gateway,pipeline}`。这是 PR #40 提交历史中较新的控制器，具备 Dörfler/world 双候选预网格、目标哈希、候选方程预算核算和确定性回退。
- 工况唯一绑定 `visionamr.bridge_cases.make_box_girder_diaphragm`，因为六维参数、开口框和 `support_width` 均对应该工厂。`visionamr.vla.world.bridge_case` 不进入本实验。
- 冻结前只允许修复运行时 API 适配和增加行为无关日志；`model.py`、`planner.py` 的状态、动作、候选、评分和常数不得改变。
- 现有 legacy `visionamr.vla.world_pipeline` 只保留为 CI 回归路径，不进入四方主结论。

## 固定随机种子与训练选择

- Manifest/LHS：`20260830`；候选 maximin LHS 数：`4096`。
- 统计 bootstrap：`20260830`，按工况整簇重采样，`10000` 次，双侧 95% R-7 分位区间。
- 监督网络种子：`20260831, 20260832, 20260833`。用 8 个验证工况和三个预算的结果按字典序选择一个网络：失败点数、有限失败惩罚后的能量 log 均值、QoI log 均值、预算违规数、seed。
- RL 种子：`20260901, 20260902, 20260903`。每个种子只有一个 budget-conditioned 策略、恰好 300 个 episode；预算按 `[30000, 60000, 120000]` 固定循环，各出现 100 次。每 25 episode 在 `8 case × 3 budget` 上验证，按失败点数、有限失败惩罚后的能量 log 均值、QoI log 均值、预算违规数、checkpoint episode 选择一个 checkpoint。
- random-safe-extra 种子：`20260911, 20260912, 20260913, 20260914, 20260915`。

## 数值失败与比值

- 成功误差只接受有限、非负值；比值计算的成功误差下限为 `1e-300`。
- 只有分子方法失败时，该配对比值固定计 `10.0`；只有分母方法失败时固定计 `0.1`；双方都失败时计 `1.0`。失败记录仍保留在 failure matrix，不以比值替代原始失败状态。
- 无预算内解、求解器失败、无效参考、非有限误差或缺少必需证据均视为方法失败；不得使用超预算解或删除工况。
- RL 的逐点中位数把失败排在所有有限成功值之后；三种子中至少两个失败时，RL-median 失败。

## `WORLD_MODEL_MECHANISM`

该门固定在 `K=6, B=60000` 的 16 个首轮测试工况上。必须同时满足：

1. 至少 75% 工况实际执行过一个经认证的 proactive action；
2. 每一个实际 proactive action 都有认证回执；
3. WM-full、WM-h1 与 random-safe-extra 使用相同均匀 probe、相同求解/方程预算，并与竞争方法隔离；
4. WM-full / WM-h1 的工况几何均值比严格小于 1，且 case-bootstrap 95% 上界严格小于 1；
5. WM-full / 五种子 random-safe-extra 逐工况中位数的几何均值比严格小于 1，且 case-bootstrap 95% 上界严格小于 1。

WM-prior-only、WM-no-history 和 oracle-future-hit 保留为诊断，不额外加入该布尔门，避免在正文未规定时事后叠加新成功条件。

## 最终公式与时间

- `OVERALL_WIN` 采用第 1 节的完整公式：`DORFLER_SAFE && BEAT_LOCAL_PREDICTION && BEAT_SUPERVISED && BEAT_RL && WORLD_MODEL_MECHANISM`。
- 第 8 节仅写三竞争者逻辑与的简写不覆盖第 1 节。`ONLINE_TIME_ACCEPTABLE` 必须报告，但不加入 `OVERALL_WIN`。
- 非求解器在线开销按协议字面定义为除 CalculiX 外的全部在线时间，即语义分区、世界模型、参数工具和 Gmsh 之和；不把 Gmsh 隐藏到 CalculiX 时间内。
- 所有在线方法的 `NodalSizeField` 统一使用 `gradation=1.0`。该值保留 PR #40 新 V0 工具层的实际默认行为；LP、SUP、RL 与独立 Dörfler 显式对齐，不改变 V0 的动作、评分或常数。
- 训练成本摊销统一使用 `B=60000, K=6` 的完整真实轨迹作为代表在线成本：WM 与监督分别对 16 个测试工况的总在线墙钟取中位数；RL 的科学输出需要三个冻结策略，所以先对每个工况累加三个策略的在线墙钟，再对 16 个工况取中位数。离线成本使用训练与验证阶段实际消耗墙钟的总和；RL 三个种子按总计算成本求和，不把并行调度造成的 elapsed 缩短冒充训练成本下降。
- 对监督和 RL 分别完整报告 `T_m(n)=T_{m,train}+nT_{m,online}` 与 WM 曲线的实数交点、最小非负整数部署范围以及 `always/never/crosses` 关系；平行斜率和负交点不得省略或强行裁成有利的正数。

## 参考升级

- Reference A 使用仓库现有解析强分级场。
- 独立候选 B 不读取任何被比较方法网格；背景尺度每级乘 `0.8`，解析局部 floor 每级乘 `0.7`。
- 若 A/B 的 U 或 QoI 相对差超过 `0.5%`，将候选 B 提升为新 A，并仅按同一固定 schedule 继续生成更细候选；最多 6 个等级。未收敛时工况保留并记 reference failure。

## 冻结与盲测边界

- Freeze commit 必须包含 manifest/sidecar、共享 partition schema、模型及 SHA、配置、训练/验证结果、环境与代码 SHA，并明确 `TEST_NOT_RUN=true`。
- 测试 runner 必须从 freeze commit 的模型快照为每个 `(case, budget, method/seed)` 独立重载；测试工况之间不共享在线残差。
- 只有 freeze commit 之后的独立命令才允许生成测试参考与方法结果；测试默认按 `case_id` 排序，一次执行全部 16 例。

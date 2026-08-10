# 当前误差证据、跨轮次资源重投与 P–H 路由：Skill 和实验 V3

## 1. 核心科学问题

局部误差估计与 Dörfler marking 继续承担可靠的当前状态判断：它回答“这一轮哪里值得投入资源”。LLM 不重新发明误差估计器，也不以几何视觉替代数值热点；LLM 的主要任务是利用理论、粗网格解、当前 Dörfler 支持、任务 QoI 与跨问题先验，预测这些当前热点在后续自适应轨迹中是否会持续被命中、最终大约需要多少累计加密深度，并把原本分散在若干轮 `solve → estimate → mark → refine` 中的资源提前委派为区域级 macro-action。本文真正检验的是：在相同当前局部误差证据下，先验驱动的资源重投能否用更少真实 FEA 反馈达到相近的最终精度—资源状态，同时在先验错误时通过审计、刷新与回退维持可靠性。

## 2. 人工智力过程的可执行化

人工专家的功能过程被写成：`粗网格真实求解 → 当前局部误差与 Dörfler 支持 → 将离散标记聚合为共同干预区域 → 判断热点持续性与未来重复命中次数 → 估计累计资源深度 → 选择一次性重投或设置中间审计点 → 执行真实 FEA → 刷新误差支持 → 修正持续性模型或停止`。LLM 只承担持续性、机制归因、累计动作结构和反馈购买策略；PSO、MILP、动态规划或 DQN 负责区域内部的数值定解；有限元求解器和后验估计器负责最终裁决。

## 3. 核心 Skill 链

`compile_task_contract → solve_coarse_and_mark → aggregate_current_support → derive_mechanism_evidence → predict_future_hits_and_depth → assess_ph → delegate_resources → execute_and_audit → refresh_support_or_stop → calibrate_ph_gap`

`solve_coarse_and_mark` 在共同初始网格上完成真实 FEA，计算局部指标并生成当前 Dörfler 支持；`aggregate_current_support` 将离散标记按连通性、尺度和共同作用机制聚合为少量资源区域，但不得删除未被解释的高指标单元；`derive_mechanism_evidence` 利用局部正则性、奇异阶、材料/界面尺度、边界层宽度、QoI 敏感性或粗网格场演化，给出为什么某一区域可能持续被命中的可证伪解释；`predict_future_hits_and_depth` 对每个当前资源区域输出未来命中次数分布、累计加密深度区间、区域扩张概率、预计收益饱和点、置信度与中间审计需求；`delegate_resources` 将预测转换成深加密、浅加密、保持、背景粗化或两阶段 macro-action；`execute_and_audit` 比较预测误差下降、实际误差下降、热点迁移和资源增长；`refresh_support_or_stop` 在真实证据显示出现新热点、原热点消失或重投过深时重新计算 Dörfler 支持；`calibrate_ph_gap` 使用执行轨迹校正下一批任务的 P、H 与失效风险评估。

## 4. 边界与限制门

LLM 不允许仅凭“孔、圆角、高应力颜色”直接宣告持续热点，不允许直接给出未经计算的精确尺寸，也不允许绕过当前局部误差证据。每个资源重投建议必须绑定当前 Dörfler 支持、理论或粗网格机制证据、未来响应假设和证伪条件。当前支持不足以覆盖未来轨迹时，系统必须保留中间真实 FEA 审计，不能把高 P 当作无条件 one-shot。限制门包括任务完整性门、当前证据门、理论适用域门、未来热点可辨识门、数值定解门、过度重投风险门、新热点发现门和物理验收门。

## 5. 方法池

固定方法池采用：`DORFLER_H2` 与 `DORFLER_H3`，代表相同 Dörfler 规则在两次或三次有限反馈下的经典基线；`INDICATOR_DELEGATE`，仅依据当前指标强度把多轮动作提前合并；`LLM_DELEGATE`，依据冻结机制合同预测当前热点的累计加密深度并一次重投；`LLM_DELEGATE_GUARDED`，在高风险任务上重投后回退到一次动态 Dörfler 校正；`LLM_TWO_STAGE`，先对当前持续热点重投，再用一次真实 FEA 刷新 Dörfler 支持并处理后来显现的热点；`D_SUPPORT_PSO`，在相同当前 Dörfler 支持内由 PSO 搜索区域深度；`LLM_PSO`，以 LLM 预测深度为搜索中心进行低维 PSO 定解；`D_SUPPORT_DQN` 与 `LLM_DQN` 将在现有 GCN-DQN 接口中分别使用纯数值支持和机制重投支持；`SL_ONE_SHOT` 代表高 P 终态预测；`PH_ROUTER` 根据校准后的成功概率、P、H、N_FE、W_phys、风险和剩余预算选择路线。

## 6. 两个必要 Oracle

`ORACLE_CURRENT_SUPPORT` 使用长程动态 Dörfler 的真实未来命中次数，但只能在第一次粗网格得到的当前支持内分配深度，用于测量“只解决重投多少”能够达到的理论上限；`ORACLE_FULL_TRAJECTORY` 同时知道后续轮次中新出现的热点和最终区域深度，用于测量完整轨迹压缩上限。两者之间的差距直接量化“初始热点深度预测”和“后来热点发现”各自贡献多少，因此能够判断 LLM 是否可以完全 one-shot，还是必须保留少量物理刷新。

## 7. 主要评价量

核心结果不以当前热点 IoU 为主，而报告达到 `E≤ε` 且 `C≤B` 所需的最小 H、相同 H 下的误差与 Pareto 距离、成功率、`success | reference-feasible`、累计真实 FEA 次数 N_FE、物理工作 W_phys、最终资源、未来命中次数 MAE/校准、预测累计深度与长程 AFEM 累计深度偏差、成功样本节省轮次、重投过深率、新热点漏检率、回退率和 CVaR。PH 图中空心点为执行前评估，实心点为实际执行，箭头表示 P/H 偏差，点大小表示 N_FE 或 W_phys。

## 8. Stage-0B 已启动结果

已在 384 个参数化一维变系数椭圆有限元任务上完成三算法种子运行，严格使用同一物理测试集；测试集 174 例，其中 139 例可由长程动态 Dörfler 在给定资源上限内达到误差容限。`DORFLER_H2` 总体成功率为 0.310、条件成功率为 0.388、平均 H 为 1.83；`DORFLER_H3` 为 0.431、0.540、H=2.52；`INDICATOR_DELEGATE` 为 0.299、0.353、H=2.00；`LLM_DELEGATE` 为 0.351、0.417、H=2.00；`LLM_TWO_STAGE` 达到 0.678、0.791、H=2.64，并在成功样本中相对长程动态 Dörfler平均节省 0.62 轮；`SL_ONE_SHOT` 为 0.477、0.583、H=2.00；`ORACLE_CURRENT_SUPPORT` 的条件成功率只有 0.583，而 `ORACLE_FULL_TRAJECTORY` 达到 1.000。该差距表明只把初始 Dörfler 热点加得更深具有明确上限，后来显现的热点使至少一次真实物理刷新具有结构性必要；当前最有价值的路线是“机制重投 + 一次支持刷新”，而不是无审计 one-shot。

当前 PH 路由器仅达到约 0.444 总体成功率并过度调用 PSO，保留为负结果；下一步必须把 N_FE、W_phys、后来热点风险和 PH 评估偏差直接纳入路线价值模型。Stage-0B 使用冻结机制合同验证算法结构，不构成在线 LLM 推理能力的最终证据，DQN 和真实二维/三维 CalculiX 矩阵仍需接入。

## 9. 正式实验推进顺序

先在现有 Stage-0B 中完成 D_SUPPORT_DQN、LLM_DQN 和保守 PH 路由校准；随后迁移到二维板孔、L 形域、界面、边界层、多热点、诱饵 QoI 与复合机制 CalculiX 参数化库；最后在桥梁三维构件上验证 transfer。每个阶段都先比较相同当前 Dörfler 支持下的“一级逐轮投入、指标重投、机制重投、PSO/DQN 数值定解”，再检验动态 P–H 路由，避免把收益错误归因于重新寻找热点。
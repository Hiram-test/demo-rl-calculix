# 机制推理、热点分区与 P–H 路由：Skill 和实验 V2

## 科学问题

论文只解决四个卡点：LLM 如何在明确边界下融合理论推导和粗网格证据形成可证伪机制模型；如何对每条候选路线评估先验压缩 P、物理反馈 H、成功概率与尾部风险；如何从机制模型得到有限预算下真正值得共同干预、会被后续多轮持续命中的热点区域；动态 P–H 路由能否提高系统效率，并在机制偏移、证据退化和模型失配下可靠回退。Agent、MCP、消息传递和日志属于工程执行层。

## 核心 Skill 链

`compile_task_contract → derive_theory_evidence + extract_coarse_evidence → infer_mechanism_graph → assess_ph → select_hotspot_partition → route_by_ph → execute_regional_method → audit_execution → calibrate_ph_gap`

`derive_theory_evidence` 把解析解、局部渐近展开、奇异阶、尺度律、能量估计或伴随敏感性转成具有假设、空间签名、动作响应、适用域和证伪条件的结构化证据；`extract_coarse_evidence` 从共同粗网格 FEA 提取 Dörfler 支持、指标密度、恢复梯度、QoI 影响、网格质量和粗网格分辨率警告；`infer_mechanism_graph` 融合两类证据，输出区域、误差成因链、竞争机制、未来持续性、允许动作、响应模型结构、置信区间和最小辨识探针，任何区域都必须绑定数值证据和证伪条件，几何显著性不能单独通过证据门。

`assess_ph` 对 `DM_D`、`DM_PSO`、`DM_DQN`、`LM_D`、`LM_PSO`、`LM_DQN`、`SL_ONE_SHOT` 分别输出四维 P 分量、预计 H、N_FE、W_phys、成功概率和区间。P 不等于模型规模，采用 `P_support=1-K/N`、`P_action=1-log(1+n_candidate_hat)/log(1+A_retained)`、`P_trajectory=max(0,1-H_hat/H_ref)`、`P_solve=max(0,1-N_FE_hat/N_FE_ref)`；绘图投影固定为 `P_plot=1-[(1-P_support)(1-P_action)(1-P_trajectory)(1-P_solve)]^(1/4)`，路由仍读取完整向量。H 正式记录 `(H_seq,N_FE,W_phys)`，二维图纵轴只使用 `H_plot=min(H_seq/H_cap,1)`，点大小显示 N_FE 或 W_phys，避免把大量 PSO 并行求解隐藏在低顺序深度内。

`select_hotspot_partition` 同时输出 Dörfler marking 数值分区、LLM 机制分区和证据门控融合分区。热点真值不能只等同当前 Dörfler 集合，使用当前指标高值区、长程动态 AFEM 的未来重复命中区、预算内离散 oracle 或局部灵敏度边际收益区并行审计。`route_by_ph` 在成功概率、H、资源、Skill 适用域和 CVaR 约束下选择路线；所有高压缩路线不满足约束时必须回退动态 Dörfler。`audit_execution` 从完整执行轨迹重新计算 P_exec 和 H_exec，发生回退时按各路线实际物理工作加权 P，累加 H 和 N_FE；输出 ΔP、ΔH、PH 欧氏差距、反馈低估、过度压缩和回退建议。`calibrate_ph_gap` 按方法、机制、QoI、预算和 OOD 类型学习偏差与置信区间，下一批路由使用校准后的保守 P/H，而非继续相信未经校准的 LLM 评估。

## 实验块

E1 机制证据实验比较 G 几何文本、T 理论证据、C 粗网格证据、TC 理论加粗网格、TC-no-gate 无限制门五种条件。评价机制 macro-F1、区域 IoU、未来持续热点召回、诱饵误报、动作响应排序相关、Brier/ECE、区间覆盖和超域 abstain；TC 至少在机制分类、未来热点或动作响应中的两个指标显著优于 G，才能支持机制推理主张。

E2 PH 评估实验在执行前对每个任务和路线冻结 PHAssessment，执行后从轨迹重算 PH，逐算例记录 ΔP、ΔH、Gap、H 低估率、P 高估率、回退率和额外物理工作；比较未校准方法卡、偏差校准、保守分位校准和在线批次更新。核心检验是校准后 routing regret、高压缩失败率和 H 低估率是否下降。

E3 热点分区实验形成 `partition source × executor` 两因素设计：DPartition、LPartition、FusedPartition 分别连接 Dörfler、PSO、DQN。报告当前支持召回、未来重复命中区召回、边际收益加权召回、支持大小、诱饵资源浪费、同预算误差、E–C Pareto 距离和达到容限所需 H；使用混合模型检验分区主效应、执行器主效应及交互。

E4 动态路由实验比较固定 DM_D、固定最高 P、固定验证集最优、简单阈值路由、无 PH 校准 LLM 路由、完整校准路由和逐实例 oracle。成功判据预先固定：总体和 OOD 成功率不得比最可靠固定路线低超过 2 个百分点；成功样本中中位 H 或 W_phys 至少一个相对 DM_D 降低 20%；routing regret 中位数接近零且 90% 分位受控；高 P 失败时完整系统的检测和回退恢复率显著提高。

E5 鲁棒性实验从主测试库分层抽取 120 例，每例加入六类单因素扰动：粗网格变化、指标噪声/漏报、LLM 漏热点或诱饵区、理论 Skill 超域、DQN/监督模型 OOD、PSO 预算截断或求解器/网格失败，共 720 个鲁棒性任务。报告错误路线率、错误接受率、失效检测率、回退率、恢复率、额外 H/N_FE/W_phys 和 CVaR90/95。

## 算例库

二维 CalculiX 主库为 720 例：10 个机制家族×6 个几何实例×2 个载荷×2 个 QoI×3 个预算。前八个单机制家族的实例 0–2 用于训练，实例 3 用于验证，实例 4–5 用于参数外推测试；各向异性家族全部作为新机制测试，mixed_shift 全部作为组合机制 OOD 测试。三维迁移库为 96 例：6 个结构模板×4 个参数实例×2 个载荷×2 个预算。PSO 五个种子，DQN 五个独立训练种子，监督模型三个训练种子；120 个高歧义任务采用至少三个 LLM、每个模型五次调用。

## 输出与主图

每个任务输出 TaskContract、TheoryEvidence、CoarseEvidence、MechanismGraph、PHAssessment、Partitions、RouteDecision、ExecutionTrace、ExecutionAudit 和求解器收据。聚合输出 `ph_gap_case.csv` 与 `ph_gap_aggregate.csv`。主图中空心点是执行前评估 PH，实心点是实际执行 PH，箭头表示偏差，误差条表示跨算例/种子置信区间，点大小表示在线真实 FEA 调用，颜色表示方法家族，虚线是由实际执行点拟合的 P–H 描述性下降趋势。图形同时输出 PNG 和 PDF；差距图是校准工具，不只是示意图。

# Live LLM V4：一次物理探针后的 future-hit 资源委派成功结果

V3 已证明此前空输出来自 reasoning-token 截断；在 8192 completion 上限下，真实 `deepseek-v4-pro` 可以正常输出最终公开委派。然而仅凭初始粗网格与第一次 Dörfler 支撑，V3 选择了错误区域 10 和 12，没有减少真实反馈深度，因此剩余科学问题转化为“哪些最小物理证据足以支持未来持续性判断”。

V4 在共同粗网格后先执行一次标准 Dörfler 一级加密和真实 CalculiX 求解，把每个候选区域加密前后的 `q_before`、`q_after`、Dörfler 是否持续命中、retention ratio、结构角色和邻接关系交给 LLM；future-hit 长程标签始终不进入在线证据。LLM 只输出少量区域及“剩余累计加密深度”，deterministic compiler 负责生成完整网格动作并执行预算门。

真实运行中，初始 Dörfler 支撑为 `[8,13,15,14,7,9]`；一次真实探针后，支撑收缩为 `[14,13,15]`。长程动态 Dörfler 的剩余 future-hit 真值为 region 13=3、14=3、15=2，其余区域为 0。真实 `deepseek-v4-pro` 委派为 region 14 depth 3 confidence 0.99、region 13 depth 2 confidence 0.95、region 15 depth 2 confidence 0.90，并额外给 region 12 depth 1 confidence 0.85。对应完整预测向量的 depth MAE 为 0.125，persistent-hotspot F1 为 1.0。

动态 Dörfler 基线达到 objective `1.256228669594032` 需要额外真实反馈深度 `H=3`。V4 路线使用一次 Dörfler 探针后执行一次 LLM 委派宏动作，在 `H=2` 时达到 objective `1.226656012648666`，优于共同终态质量门，因此成功减少 1 个顺序真实 FEA 反馈轮次，并且没有触发第三轮安全回退。该结果是当前首个“真实 LLM + 真实 CalculiX + 非 oracle future-hit 证据”下的正 trajectory-compression 结果。

本结果仍然只是一项单算例机制验证，不能外推为总体性能结论。下一步应在参数化机制库中同时比较：有限 H 的动态 Dörfler、只看当前强度的重投、V3 粗网格 one-shot 机制委派、V4 一次物理探针机制委派、PSO/DQN、监督学习以及 P–H 动态路由，并重点统计 `H`、`N_FE`、`W_phys`、终态 E–C Pareto 距离、future-hit depth MAE、持续热点 F1、回退率和 assessed-versus-executed P–H gap。

# 世界模型执导的多步 VLA

## 1. 目标与边界

本实现把 VLA 收敛为一个窄定义的自适应有限元控制器：视觉头只建立桥梁构件的语义区域，世界模型预测区域网格动作对后续误差、资源和热点迁移的影响，确定性工具把离散动作转换为 Gmsh 参数，CalculiX 提供真实状态转移。

主路径不导入 `local_prediction`，不读取其逐单元目标尺寸、误差允许值、幂指数或轨迹。局部预测仍是独立对照方法。

闭环为：

\[
\text{三维构件与边界标注}
\rightarrow
\text{视觉语义区域}
\rightarrow
\text{共同均匀 probe}
\rightarrow
\text{区域图状态}
\rightarrow
\text{世界模型多步推演}
\rightarrow
\text{MCP 参数认证}
\rightarrow
\text{Gmsh 重网格}
\rightarrow
\text{CalculiX 真值反馈}.
\]

## 2. 为什么以 Dörfler 为安全策略

每轮真实求解后仍执行标准的逐单元 ZZ 指标与 Dörfler bulk marking。世界模型不替换误差估计器，也不能撤销 Dörfler 标记。它只能在 Dörfler 节点目标场上追加少量语义区域加密：

\[
h^{\mathrm{WM}}_j\le h^{\mathrm{D}}_j
\qquad \forall j.
\]

`MCPMeshGateway.materialize()` 从 `refine_size_map()` 产生的精确 Dörfler 节点目标开始，再对被选区域取更小尺寸，并在送入 Gmsh 前逐节点验证上述不等式。以下情况直接执行纯 Dörfler：

- 世界模型相对 Dörfler 的稳健多步收益不足；
- 集成模型的误差或资源不确定性超过阈值；
- 预测方程数没有保留预算安全裕度；
- MCP 动作维数、稀疏性、深度或预算认证失败；
- 上一轮主动动作导致真实 ZZ 总量明显回升。

这形成的是安全策略改进结构。由于 Gmsh 重网格非嵌套，节点目标场更细并不构成真实有限元误差必然更小的数学证明。因此实现同时保留独立的等求解次数实算闸门，不能用节点级保证替代数值结果。

## 3. 世界状态、动作和转移

第 \(t\) 次真实求解被压缩为区域图状态：

\[
s_t=\left(G_R,\left\{x_{i,t}\right\}_{i=1}^{R},x_t^g\right).
\]

区域状态包含：

\[
x_{i,t}=\left[
\eta_i^2,
N_i,
h_i,
\sigma_{\mathrm{vm},i}^{\max},
V_i,
\rho_i^{\mathrm D},
\nu_i^{\mathrm D},
c_i
\right],
\]

其中 \(\rho_i^{\mathrm D}\) 和 \(\nu_i^{\mathrm D}\) 分别是本轮精确 Dörfler 集在区域内覆盖的误差比例和单元比例，\(c_i\) 是该区域此前进入 Dörfler 集的次数。

动作不是连续尺寸，而是相对 Dörfler 的离散区域提前量：

\[
a_i\in\{0,1,2\}.
\]

`0` 表示仅执行本轮逐单元 Dörfler；`1` 表示把该语义区域扩展为一个完整的本轮加密层；`2` 表示对已反复显现的持久热点提前一个附加层。每轮最多主动选择两个区域。

世界模型预测区域误差与单元数的对数变化：

\[
W_\theta(s_t,a_t)
\rightarrow
\left(
\Delta\log\eta_i^2,
\Delta\log N_i,
\sigma_E,
\sigma_N,
P_{\mathrm{risk}}
\right).
\]

模型由两部分组成：

1. 与 Dörfler 加密因子和空间维数一致的保守物理先验；
2. 从真实 CalculiX 转移中在线学习的 bootstrap ridge 残差集成。

它不重建完整位移场。预测对象只限于决策所需的误差、资源、热点持续性和不确定性。

## 4. 多步规划

规划器采用有限时域 beam search。纯 Dörfler 零动作始终存在于每一层搜索树中。目标函数综合考虑：

\[
J=\sum_{k=1}^{H}\gamma^{k-1}
\left[
\log\frac{\widehat E_{t+k}}{E_t}
+\lambda_N\log\frac{\widehat N_{t+k}}{N_t}
+\lambda_U U_{t+k}
+\lambda_R P_{t+k}^{\mathrm{risk}}
+\lambda_B\left(\frac{\widehat N_{t+k}}{B}-1\right)_+^2
\right].
\]

默认真实求解上限为 6，内部预测时域为 4。内部 rollout 不调用 LLM、Gmsh 或 CalculiX，因此可以评估多条未来轨迹，而真实代价仍只发生在最终选中的一次重网格与求解。

世界模型相对当前局部尺寸预测的预期优势被限定为三项：

- 识别同一孔缘、轮载边缘、横隔板—腹板交线或支承反力区是否会连续多轮进入 Dörfler 集；
- 预测一次区域提前加密对相邻区域渐变带和总资源的耦合影响；
- 在同一家族不同几何、开孔和轮位之间复用已经观测到的动作后果。

如果这些跨步、跨区和跨实例信息没有带来稳健收益，规划器退回 Dörfler，不能以“使用了世界模型”作为继续主动加密的理由。

## 5. 三维桥梁构件

`make_box_girder_diaphragm()` 建立一个中等复杂度的钢箱梁节段，包含：

- 顶板、底板和双腹板；
- 带圆形检修孔的内部横隔板；
- 检修孔局部框架；
- 偏置轮载压力斑；
- 一端固定平动、另一端竖向约束的支承条带；
- 轮载边缘、开孔双侧孔缘、横隔板—腹板交线和支承区等竞争热点。

荷载斑和支承条带均通过 OCC fragment 压印到边界，避免重网格改变荷载或约束作用范围。随机族改变轮位、开孔半径、横隔板厚度和压力，用于积累跨实例转移。

## 6. 运行

安装核心依赖和 CalculiX 后，执行一个最多 6 次真实求解的世界模型与 Dörfler 对照：

```bash
python scripts/run_world_model_vla.py --compare-dorfler --max-solves 6 --horizon 4 --budget 60000  # 运行共同 probe 下的等求解次数对照。
```

显式要求经验上不弱于 Dörfler：

```bash
python scripts/run_world_model_vla.py --compare-dorfler --enforce-dominance --max-solves 6 --horizon 4 --budget 60000  # 未通过等求解次数闸门时返回非零状态。
```

先在随机桥梁构件上建立可复用转移库：

```bash
python scripts/build_world_model_library.py --instances 6 --solves-per-instance 4 --budget 120000  # 用真实动作转移建立 JSON 世界模型库。
```

再在未参与建库的规范构件上加载该库：

```bash
python scripts/run_world_model_vla.py --transition-library results/world_model_library/transition_library.json --compare-dorfler --max-solves 6 --horizon 4 --budget 60000  # 验证跨实例世界模型收益。
```

所有 CalculiX 调用仍由 `FemRunner` 计数。结果写入 `summary.json`、方法独立的 `records.json` 和透明的 `transition_library.json`。

## 7. MCP 参数工具

MCP 只承担确定性参数层，不负责预测物理后果。可选安装和启动命令为：

```bash
python -m pip install -r requirements-mcp.txt  # 安装可选的 MCP v2 Python SDK。
python -m visionamr.vla.mcp_server  # 通过 stdio 启动参数工具服务器。
```

暴露的工具包括：构件与单位检查、离散层级到尺寸的精确转换、区域动作与预算认证、Dörfler 节点目标场支配性认证。世界模型的准确性仍由独立实算、集成不确定性和经验闸门负责。

## 8. 验收指标

必须分别报告：

- 等真实求解次数下的 \(e_E\)、\(e_{\mathrm{QoI}}\) 和 \(\sum\eta_K^2\)；
- 等方程预算下的最优可交付点；
- VLM、世界模型规划、Gmsh 和 CalculiX 的分项墙钟；
- 主动动作比例、Dörfler 回退比例和预测校准残差；
- 每次节点目标场是否逐节点不粗于 Dörfler；
- 跨实例转移库启用前后的差异。

“反超”只在未见构件上、共同 probe、相同求解器与相同预算下，由真实误差曲线判定。实现本身不预写反超结论。

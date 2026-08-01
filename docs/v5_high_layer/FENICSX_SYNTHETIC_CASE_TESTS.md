# FEniCSx 轻量数值案例测试

本切片在固定的官方 DOLFINx `v0.11.0` 容器中实际执行网格生成、UFL/FFCx 形式编译、有限元组装、PETSc 线性求解和结果提取。它不再是仅导入模块的环境检查。

## 证据边界

两个测试均属于 `synthetic_numerical_contract_evidence`：输入由 CI 确定性生成，用于验证 FEniCSx 数值链。它们不是 FEN-003 或 FEN-014 的原始研究输入，不能关闭 profile 中的缺失事实，也不能支持工程设计或论文数值结论。

因此，机器回执同时记录：

- `synthetic_numeric_results_generated: true`：本轮确实产生了网格、有限元解和数值指标。
- `uses_original_research_inputs: false`：本轮没有使用用户尚未提供的原始案例材料。
- `research_case_execution_status: not_executed_missing_current_evidence`：原研究案例仍保持阻断。
- `scientific_claim_allowed: false`：不得把合成基准绿色状态解释为科学或工程结论。

## SYN-FEN-003-MMS-POISSON-P1

该基准验证 FEN-003 所需的受控网格序列与 QoI 提取管线。

| 字段 | 冻结值 |
|---|---|
| 区域 | 单位方形 `(0,1) × (0,1)` |
| 方程 | `-Δu=f` |
| 解析解 | `u=sin(πx)sin(πy)` |
| 源项 | `f=2π²sin(πx)sin(πy)` |
| 边界 | 四边齐次 Dirichlet |
| 单元 | 连续一阶 Lagrange 三角形 |
| 对角线 | `DiagonalType.right` |
| 网格层级 | `8, 16, 32, 64` |
| 主 QoI | `∫Ω u dx`，解析值 `4/π²` |
| 独立量 | L2 误差、H1 半范误差、离散能量与载荷功平衡 |
| 求解器 | PETSc `preonly + lu`，不收敛立即失败 |

预冻结验收门包括网格/自由度精确计数、四次 KSP 成功、误差逐级下降、后两段 L2 阶位于 `[1.7, 2.3]`、后两段 H1 阶位于 `[0.8, 1.2]`、最细层误差限制以及每层能量相对不平衡不超过 `1e-10`。最细层解保存为 XDMF/HDF5。

## SYN-FEN-014-XDMF-MESHTAGS-ROUNDTRIP

该基准验证 FEN-014 所需的网格导入、命名标签、Measure 和边界自由度映射链。

1. 生成 `8 × 8` 单位方形右对角三角网格。
2. 用 cell tags `1/2` 标记左右半区，用 facet tags `11/12/13/14` 标记左、右、下、上四边。
3. 将网格和两组命名 `MeshTags` 写入 HDF5 XDMF，关闭文件。
4. 按网格和标签名称重新读取为全新对象，并重新创建函数空间、`dx/ds Measure` 和边界自由度映射。
5. 在参考网格与重读网格上分别求解 `-Δu=0`、全边界 `u=x+y` 的线性补丁问题。

固定预期为 81 个顶点、128 个单元、208 条边；左右 cell tags 各 64 个且面积各 `0.5`；四边 facet tags 各 8 个且长度各 `1`；标签派生边界自由度与独立几何定位集合相同且数量为 32；两次 KSP 均成功；`∫Ωu dx=1`、能量为 `2`，解析误差与往返差异不超过 `1e-10`。

## GitHub Actions 输出

工作流将输出独立 artifact `v5-fenicsx-synthetic-contracts-<run>-<attempt>`，其中包含：

- `campaign_receipt.json`；
- 两个案例的 `numerical_contract_receipt.json`；
- 与数值 runner 分离生成的 `artifact_validation_receipt.json`；
- FEN-003 最细层解的 XDMF/HDF5；
- FEN-014 带标签输入包和重读解的 XDMF/HDF5；
- 容器日志、真实退出码、来源快照和 `SHA256SUMS`。

任一数值门失败都会使 Actions 失败，同时仍上传已经形成的回执和日志。

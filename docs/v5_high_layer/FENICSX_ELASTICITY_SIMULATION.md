# FEniCSx 轻量结构模拟

本切片新增一个真正的二维结构场模拟，而不是用标量 Poisson 方程代替结构问题。GitHub Actions 在固定的官方 DOLFINx `0.11.0.post0` 镜像中建立平面应力弱式、生成四层三角形网格、组装刚度和均布边界荷载、调用 PETSc 求解位移，并从位移梯度计算 DG0 von Mises 应力。

## 演示假设

当前没有用户确认的 FEN-003 或 FEN-014 工程模型，因此下列数值只属于本次新建的演示模型，不能回填原案例的 missing facts。

| 项目 | 演示值 | 含义 |
|---|---:|---|
| 几何 | `4 m × 1 m` | 二维矩形悬臂板中面 |
| 厚度 | `0.01 m` | 平面应力刚度与合力换算厚度 |
| 弹性模量 | `210 GPa` | 演示用各向同性线弹性参数 |
| 泊松比 | `0.3` | 演示用各向同性线弹性参数 |
| 左边界 | `u_x=u_y=0` | 全固定边界 |
| 右边界 | `t_y=-1 MPa` | 整条右边均布向下载荷 |
| 物理 | 小应变、线弹性、静力、平面应力 | 不包含塑性、接触、疲劳和几何非线性 |
| 网格序列 | `16×4`、`32×8`、`64×16`、`128×32` | 只改变网格离散的受控序列 |
| 单元 | 连续一阶 Lagrange 三角形 | 位移为向量场，应力由位移梯度计算 |

`analysis_charter.json` 将该模型用途限定为“证明前端/提高层能够部署、执行、显示并审计真实 FEniCSx 结构场模拟”。因为没有正式工程输入、适用规范和专业批准，G0 保持 `BLOCKED`；绿色 Actions 不表示任何桥梁构件或原研究案例满足设计要求。

## FEN-003 能力映射

四层网格保持几何、材料、载荷、边界和提取协议不变。固定控制量为右端整条边的平均竖向位移、总应变能以及板中部固定物理条带的平均 von Mises 应力。固定条带范围为 `x∈[7L/16, 9L/16]`，不会随网格编号漂移。

最细层相对前一层必须满足：

- 右端平均竖向位移相对变化不超过 `2%`；
- 应变能相对变化不超过 `2%`；
- 固定条带平均 von Mises 应力相对变化不超过 `3%`。

固定端角点附近的最大 von Mises 应力只作为诊断字段，不作为收敛或工程判定量。

## FEN-014 能力映射

最细层网格使用命名 cell/facet `MeshTags` 写入 HDF5 XDMF，关闭文件后重新读取为新的 Mesh、FunctionSpace 和 Measure，再独立执行一次相同平面应力求解。导入前后比较右端平均位移、应变能、固定条带平均应力、反力和力矩，确认拓扑、尺度、标签与边界映射没有在往返中变化。

## 解验证

每次求解都保存 PETSc 收敛原因码，并从未施加边界条件的离散残差在固定自由度上提取代数反力。验证范围包括：

- 自由自由度残差；
- 外荷载与固定端反力的全局力平衡；
- 关于原点的全局力矩平衡；
- 离散内功与外功一致性；
- 荷载方向与右端位移方向一致；
- 三个固定控制量的网格变化；
- XDMF 往返前后的结果差异。

这些检查属于解验证，不替代规范验算、材料强度评定或真实结构校核。

## Actions 产物

`v5-fenicsx-elasticity-simulation-<run>-<attempt>` artifact 包含：

- `simulation_receipt.json` 和独立 `artifact_validation_receipt.json`；
- 分析边界工件：`analysis_charter.json`、`standards_manifest.json`、`response_metric_register.json`、`approval_matrix.json`、`scope_exclusion_register.json`；
- 解验证工件：`solution_verification_report.json`、`verified_result_set.json`、`global_equilibrium_report.json`、`substructure_free_body_report.json`、`mesh_and_step_convergence_report.json`、`solver_warning_disposition.json`、`solution_issues.json`；
- 最细层命名网格及标签的 XDMF/HDF5；
- 位移向量场和 DG0 von Mises 应力场的 XDMF/HDF5；
- 由真实求解数据绘制的 `elasticity_simulation_summary.png`；
- stdout、退出码、来源快照和完整 `SHA256SUMS`。

图像由 Matplotlib 读取本次 DOLFINx 解向量、网格拓扑和应力函数确定性绘制，不使用生成式图像模型。

上表描述成功包。若容器求解、绘图或文件写出中途失败，Actions 仍会上传已经形成的部分工件、stdout、退出码、哈希清单和失败的 `artifact_validation_receipt.json`；缺失的场文件不会被伪造为空壳。

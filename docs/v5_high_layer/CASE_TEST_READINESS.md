# V5 提高层案例测试：T0 就绪性

## 已确认的执行技术族

当前用户已明确确认以下映射：

- FEN-003 与 FEN-014 使用 FEniCS/FEniCSx。
- CCX-015 使用 CalculiX。

profile 使用 execution_family 记录该映射。FEniCS/FEniCSx 是有限元应用框架；FEniCSx 当前核心运行环境是 DOLFINx。具体采用 legacy FEniCS 还是 FEniCSx、版本、入口程序和底层线性或非线性求解配置必须由当前案例证据确认，不能由案例前缀自动补齐。

CCX-015 中 CalculiX 负责读取输入、求解和结果输出。候选网格由哪个网格器或细化器生成、怎样保持节点集和单元集、怎样转换为 CalculiX 输入，必须作为独立工具链事实确认。

## 本轮 T0 测试做什么

T0 是真实的前置条件测试，但不是有限元数值测试。它读取当前提交中的三个 profile 和实际引用的通用 Skill，通过现有 ProblemManifest 合同逐项保留 missing_facts，并验证：

- FEN 与 CCX 的 execution_family 没有接反。
- profile 仍为 draft_not_executed。
- 当前没有物理 facts、旧案例 fixture 或待确认假设混入。
- QoI 字段和停止条件没有被默认赋值。
- 需要的当前证据均明确标记为缺失。
- 在 source_audit 阶段停止。
- 模型、FEniCS、解析器、网格检查器、网格生成器、CalculiX 和优化器调用全部为零。

GitHub Actions 成功仅表示系统正确执行 fail-closed，不表示三个工程案例已经通过。

## FEniCSx 环境资格测试

FEN-003 和 FEN-014 另做一次独立环境资格测试。工作流拉取以下官方 FEniCS 镜像的不可变摘要：

    ghcr.io/fenics/dolfinx/dolfinx:v0.11.0@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8

资格脚本只导入 DOLFINx、Basix、UFL、FFCx、mpi4py、petsc4py 以及 dolfinx.common、dolfinx.fem、dolfinx.mesh 和 dolfinx.io，记录实际版本、DOLFINx 构建提交、Python、MPI、PETSc、镜像 ID 和 RepoDigest。

该步骤不会创建 Mesh，不会读取外部网格，不会编译或组装变分形式，不会调用线性或非线性求解，也不会生成工程结果。通过只证明当前 Actions runner 能启动固定的 FEniCSx 环境。

官方来源：

- FEniCSx 文档：https://docs.fenicsproject.org/
- DOLFINx 仓库与安装说明：https://github.com/FEniCS/dolfinx
- 官方镜像构建工作流：https://github.com/FEniCS/dolfinx/blob/v0.11.0.post0/.github/workflows/docker-end-user.yml

## Artifact 结构

三案例 T0 artifact 包含：

- campaign_receipt.json：三案例整体状态、提交、运行编号、零执行计数和来源快照哈希。
- fen-003/readiness_receipt.json：FEN-003 入口门、缺失事实、问题和允许用途。
- fen-014/readiness_receipt.json：FEN-014 入口门、缺失事实、问题和允许用途。
- ccx-015/readiness_receipt.json：CCX-015 入口门、缺失事实、问题和允许用途。
- 每个案例的 problem_manifest.json、required_input_questions.json 和 source_manifest.json。
- source/：本次实际使用的 profile、Skill、合同、runner、测试、说明和 workflow 快照。

FEniCSx 环境 artifact 包含：

- fenicsx_runtime_receipt.json：实际导入版本、构建提交、MPI/PETSc 状态和零计算边界。
- image_inspect.json：固定镜像的 Image ID、RepoDigest、架构、系统和创建时间。
- docker_pull.log：Actions runner 拉取固定镜像的真实日志。
- SHA256SUMS：上述证据文件的 SHA-256。

## 真实数值阶段仍需的当前输入

### FEN-003

进入网格收敛数值试验前必须提供：

1. FEniCS 或 FEniCSx 变体、版本、入口和运行命令。
2. 当前案例源代码或输入材料，包含网格、函数空间、变分形式、系数或材料、载荷、边界和求解配置来源。
3. 完整 QoI、位置、单位、坐标系、分析时刻、聚合和提取实现。
4. 保持问题和提取协议不变的当前网格序列与真实运行 artifact。
5. 当前停止或继续加密规则、来源和计算预算。

### FEN-014

进入外部网格差分诊断前必须提供：

1. FEniCS 或 FEniCSx 变体、版本、网格读取入口和运行命令。
2. 未修改的当前外部网格及导出来源。
3. 格式、版本、节点、单元、几何与拓扑维数和 tags 表达合同。
4. 同一物理问题、函数空间和变分形式的可运行参考案例。
5. cell 或 facet tags、MeshTags、Measure、函数空间和边界自由度映射代码。
6. 坐标、系数或材料、载荷和结果的一致单位定义。
7. 真实失败命令、Python traceback、stdout、stderr 和 Actions artifact。

### CCX-015

进入单候选受约束细化前必须提供：

1. CalculiX 版本、命令、输入 deck 入口和 Actions 环境。
2. 候选网格生成或细化工具链、版本、转换入口和集合保持规则。
3. 最后一个已接受 CalculiX 输入、网格、结果、哈希和接受理由。
4. 完整当前 QoI 与提取协议。
5. 适用单元族的质量量、硬门、来源和失败处理。
6. 局部尺寸、邻接、过渡和回退合同。
7. 网格、求解、失败和墙钟预算。
8. 当前结果驱动指标、目标区域和确定性实现来源。

## 使用边界

T0 receipt 可以证明：当前系统知道应该调用哪一类技术、没有借用旧参数，并能在证据不足时正确停止。

T0 receipt 不能证明：FEN-003 已收敛、FEN-014 网格可信、CCX-015 细化安全，或任何 FEniCS/CalculiX 工程结果成立。

固定 FEniCSx 镜像压缩体积约一 GiB，因此环境资格工作流的主要成本是首次镜像拉取；工作流超时边界为十五分钟，不使用 DeepSeek、秘密变量或求解器计算。

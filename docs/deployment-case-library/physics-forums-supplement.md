# Physics Forums 桥梁与结构问题补充索引

这三条来源记录来自现有 bridge FEA mesh corpus。它们提供“复杂工程提问”的补充信号，但不属于 [`research-user-triad.md`](research-user-triad.md) 中的三个 FEniCS 主场景，也没有被包装成已经求解的模型。

| Corpus ID | 来源 | 工程问题 | 本项目中的判读 |
|---|---|---|---|
| `C04` | [Fix-fix Arch Approximate Static Analysis](https://www.physicsforums.com/threads/fix-fix-arch-approximate-static-analysis.469779/) | 固端拱桥的近似静力分析与有限元结果如何对应，关注固定端弯矩。 | 近似模型、FE 理想化与 QoI 定义必须对齐，不能只比较一个数值。 |
| `C06` | [Structural Member Undergoing Pure Bending: Detached Mesh Regions](https://www.physicsforums.com/threads/structural-member-undergoing-pure-bending-detached-mesh-regions.769815/) | 纯弯构件中脱开的网格区域如何传力并产生合理变形。 | 网格“看起来接近”不等于拓扑连接成立；连接、约束和传力路径是部署变量。 |
| `C07` | [ANSYS Bending Moments Calculation](https://www.physicsforums.com/threads/ansys-bending-moments-calculation.505251/) | RC 桥梁/梁的实体与梁单元如何比较弯矩、刚度与重分布。 | 元素理想化决定可直接输出的量，实体应力与梁内力不能未经恢复规则直接比较。 |

原始结构化记录位于另一实验工作树的 `data/bridge_fea_mesh_cases/t3_t4_signals.csv`。本页只保存公开标题、链接和项目自己的简要判读，不复制论坛正文或附件。

## 为什么单独放置

- `C04/C06/C07` 主要检验模型保真度、网格连接和结果定义；
- `FEN-009/FEN-001/FEN-012` 分别检验相场断裂物理、经典到非经典迁移和多方法耦合；
- 把两组都叫作“另外三个问题”会丢失来源、求解器和研究能力的差异。

因此主 triad 用于后续 FEniCS 研究型实验，Physics Forums 三条记录作为跨来源补充案例。两组都已上库，但状态均明确为来源/问题定义，不宣称已经有完整原始模型或求解结果。

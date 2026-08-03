# 三类研究型有限元用户问题

本页从 30 条 deployment case library 中固定三个主场景，用来检验系统能否面对工程研究中的细微变体，而不是看到“接触”“裂纹”或“不收敛”关键词后直接套用规则库。它们与 CalculiX 过盈接触问题并列，但不混为同一案例。

## 选择原则

三个问题分别隔离一种不同的研究能力：

1. **相场断裂中的物理判读：** 程序可以运行，但裂纹路径可能物理上错误。
2. **经典案例向非经典案例迁移：** 原方法有效，但维数和模型语义变化后不能机械照搬。
3. **多个已有方法的组合：** 每个子方法单独有效，不代表耦合系统正确。

这比“高级使用者问某个单项功能怎么调用”更难，但目标仍然是有限的工程能力：识别关键差异、提出最小判别实验、调用真实求解证据并保留不确定性，而不是追求无边界的通用智能。

## 场景 1：相场断裂路径与自适应细化

- **案例：** `FEN-009`
- **来源：** [Challenges in reproducing phase-field fracture benchmark](https://fenicsproject.discourse.group/t/challenges-in-reproducing-phase-field-fracture-benchmark-unphysical-crack-trajectory-and-mesh-refinement-behavior/19720)
- **用户目标：** 在 FEniCSx 中复现 Mode-I 自适应相场断裂 benchmark，得到直裂纹和受控的细化带。
- **表面症状：** 求解可以推进，但裂纹呈锯齿或混合型，细化区也比参考结果更宽。
- **真正要判断的细微差异：** 是边界标记与加载破坏了对称性、网格方向偏置了裂纹、正则化长度与网格不相容、AMR 判据错误，还是 benchmark 本身被误读。
- **最小判别证据：** 冻结几何与材料后，分别检查边界标签图、对称反力、`h/ell` 比、无 AMR 基线、结构化/非结构化网格裂纹路径以及能量演化。
- **成功边界：** 不能把“Newton 收敛”当成成功；至少要同时验证裂纹模式、路径对称性、耗散能和网格/长度尺度敏感性。
- **当前交付状态：** 已有来源与问题定义；本仓库当前持久 FEniCSx/PETSc 环境是执行底座，但尚没有把论坛代码整理成可运行、经许可再分发的原始模型包。

## 场景 2：经典三维 Hertz 接触向二维迁移

- **案例：** `FEN-001`
- **来源：** [Contact Mechanics — 2D problem](https://fenicsproject.discourse.group/t/contact-mechanics-2d-problem/4445)
- **用户目标：** 把经典三维 Hertz 刚性压头 penalty 示例迁移成二维接触分析。
- **表面症状：** 修改后的代码不能正常运行，最初看起来像一个局部实现错误。
- **真正要判断的细微差异：** 三维到二维不是删除一个坐标；平面应力/平面应变、压头表示、gap 函数、接触边界测度、载荷单位和解析参照都会改变。
- **最小判别证据：** 先冻结二维物理假设与单位，独立画出 gap 和接触边界标签，再用一个解析或高分辨率基线比较接触宽度、压力分布与反力。
- **成功边界：** 修正变量名或让代码运行只证明局部错误消失；只有二维模型的量纲、接触表达与基准响应一致，才算迁移成立。
- **当前交付状态：** 已有来源与问题定义；没有把论坛代码当作本仓库已验证的二维 Hertz 实现。

## 场景 3：组合 bulk PDE 与 surface PDE

- **案例：** `FEN-012`
- **来源：** [Why is my program not converging for this system of coupled PDEs?](https://fenicsproject.discourse.group/t/why-is-my-program-not-converging-for-this-system-of-coupled-pdes/2284)
- **用户目标：** 耦合圆域内部的 bulk PDE 与圆周上的 surface PDE。
- **表面症状：** 两个解耦问题分别能解，耦合后即使用解析解初始化也不收敛。
- **真正要判断的细微差异：** trace 映射、bulk/surface 函数空间、耦合项符号、弱式一致性、零空间、分块装配与预条件必须作为一个整体成立。
- **最小判别证据：** 对每个耦合块做 manufactured-solution 残差检查，验证离散 trace 和符号；随后从零耦合参数连续增加，而不是直接把两个完整求解器拼接。
- **成功边界：** “两个子问题分别收敛”不是组合正确的证据；耦合残差、守恒量、解析解误差和网格收敛阶必须同时通过。
- **当前交付状态：** 原帖代码引用三个未附带的 `unitcircle*.xml`，因此当前只有来源级问题定义，不能冒充完整可运行的原始包。

## 与现有持久环境的关系

现有环境可以复用 Python、FEniCSx 0.11、PETSc、容器/本地入口、工件目录与基础验证框架。它当前实现的是张弦 benchmark 等执行底座，不包含以下研究模型：

- 相场断裂的历史变量、不可逆条件和裂纹演化；
- 三维 Hertz 示例向二维的完整物理重构；
- mixed-dimensional bulk/surface 装配与耦合求解；
- 从论坛原始代码清理出来的可再分发 reproducer。

因此这三个案例目前是**研究问题包**，不是“已经求解的三个 demo”。后续每个案例若进入执行阶段，都应保存问题清单、环境锁、原始来源收据、clean-room reproducer、逐轮决策 trace 和求解证据。

## 机器可读清单

[`research-user-triad.json`](research-user-triad.json) 对应本页。字段含义如下：

- `pack_schema`：问题包的契约版本。
- `selection_purpose`：为何从案例库选择这三个问题。
- `cases`：三个主场景，固定为一场景对应一个主要能力。
- `archetype`：待检验的研究决策类别。
- `decision_focus`：模型需要辨别的核心差异。
- `minimum_discriminating_evidence`：在形成结论前至少要执行或检查的证据。
- `delivery_status`：当前只有问题定义，还是已有可执行模型与求解结果。
- `environment_boundary`：现有持久环境可以复用什么、尚未实现什么。

Physics Forums 的三条桥梁问题不计入本 triad；它们作为来源信号保存在 [`physics-forums-supplement.md`](physics-forums-supplement.md)。

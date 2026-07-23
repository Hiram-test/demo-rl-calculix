# MeshPilot TQZ(XII) 支座构件族：真实 VM 结果

本文件由 `agent/meshpilot-tqz-support-family` 分支的完整 Ubuntu 24.04 / Gmsh / CalculiX 工作流验证后更新。

当前状态：**CalculiX 201 根因已确认并修复，六算例完整重跑已启动。**

根因不是荷载或 PSO，而是部分 OpenCASCADE/Gmsh 网格使用了稀疏节点标签，例如节点总数约 2300，但加载节点标签达到 2386。CalculiX 2.21 按节点记录数 `nk` 分配数组，因此报 `node ... is not defined`。修复在写输入文件前把节点和单元标签重排为连续的 `1..N` 与 `1..E`，并同步转换单元连接、边界集和集中荷载。

重跑仍使用原来的几何、材料、荷载合力、等效弯矩、六 patch PSO 和 32 次唯一 FE 硬预算，不通过改小荷载来掩盖输入兼容问题。

结果边界：局部三维线弹性算法基准，不是支座、垫石或整孔箱梁的工程验收计算。

# MeshPilot TQZ(XII) 支座构件族：真实 VM 结果

本文件由 `agent/meshpilot-tqz-support-family` 分支的完整 Ubuntu 24.04 / Gmsh / CalculiX 工作流验证后更新。

当前状态：**默认分支直跑与 PR 同步触发均已提交，等待 GitHub 分配 VM run。**

默认分支 runner 显式检出 `agent/meshpilot-tqz-support-family`，运行六个 TQZ 算例的 reference、coarse、cold PSO 和 guarded transfer PSO。完成并归档结果后，将清理临时 runner 并恢复分支关系。

结果边界：局部三维线弹性算法基准，不是支座、垫石或整孔箱梁的工程验收计算。

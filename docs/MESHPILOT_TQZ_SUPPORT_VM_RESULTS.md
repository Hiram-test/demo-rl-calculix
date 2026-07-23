# MeshPilot TQZ(XII) 支座构件族：真实 VM 结果

本文件由 `agent/meshpilot-tqz-support-family` 分支的完整 Ubuntu 24.04 / Gmsh / CalculiX 工作流验证后更新。

当前状态：**首轮六算例已完成诊断：1/6 通过，5/6 在 CalculiX reference 阶段返回 201。现已启动完整失败目录抓取。**

诊断工作流会保留每个失败算例的 `model.inp`、`model.msh`、`ccx.stdout.log`、`ccx.stderr.log`、`model.dat`、`model.sta` 和 `model.cvg`。拿到原始报错后，只修改被证实有问题的荷载、约束或网格映射，再重跑六算例。

结果边界：局部三维线弹性算法基准，不是支座、垫石或整孔箱梁的工程验收计算。

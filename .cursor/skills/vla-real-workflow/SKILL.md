---
name: vla-real-workflow
description: Real FEA meshing workflow for Visionamr VLA. Use when changing VLA, vision heads, size assignment, LP comparison, probes, or campaign accounting.
---

# VLA 按真实分析流程

现场不是「均匀探针 → 对着应力云图描区」。现场是两段。改 VLA、视觉头、尺寸、和 LP 对比时按这个 skill，不要退回旧顺序。

## 思维过程（就这一件事）

```
读图（零次求解）     图纸定相对粗细（不规则区 + 每区一个尺寸 + remainder）
预算尺度（零次求解） Gmsh 把整体尺度收到预算附近。排序不许改。
求解                 落在这张已画过的网格上
决策                 收到这一次的计算结果和资源存量 → 下一次分配
PSO                  只防止这次分配过度不可靠（超预算）。不是分配器。
再求解               直到决策停，或碰到求解帽
对照                 同预算一步 LP。轴：e_E、e_qoi、N/B、图纸排序还在不在。
```

画区 + 第一次给尺寸 **禁止** 调用 CalculiX / `FemRunner.solve` / `post.vm_*` / `eta2`。
没有 API。决策是代理自己写的（回放 JSON），不是提示词套本仓库的字段，也不是换一个模型就废的参数表。
**禁止** 用 PSO / `calibrate_measured` / κτ 代替决策。
**禁止** 再拟合 `η ~ h^q`、禁止 `fit_surrogate` / `Surrogate.predict` / `E_pred`。
多边形不重画。不许通信轮按 η 份额重谈（那是 AB5），不许分裂（那是 AB4），不许 LP 涂尺寸。

## 为什么

能求解还去画，就被 Li–Bettess / ZZ 局部预测吃掉。旧 VLA 用探针场涂尺寸，k=2 结构性输给 LP。
人的优势是图纸通道，加上每次算完看剩余资源再分配一次。
不是「人不用第一次求解」，也不是「PSO 在做第二次决策」。

## 文献座位（不要搅在一起）

| 方法 | 看什么 | 不看什么 |
|---|---|---|
| 读图 / GReFEM | CAD 视图 + 荷载文字 | 这一次的 \(\eta\)；GReFEM 无闭环 |
| LP | 这一次求解的 \(\eta\) | 图纸上还没显影的东西 |
| Dörfler | 多次 \(\eta\) + 标记 | 少求解冠军；渐近最优归它 |
| MeshingNet | 几何 → 冻结 \(h\) | 误差合同；标签是离线求解买的 |

禁止把「未见拓扑 / 冻结母族」写成科学结论。监督弱在标签循环和探针信息瓶颈。

## 实现锁

- `partitioner.propose` 只依赖 `Problem`。`post` / `eta2` 可留参数，但画区和初尺寸必须忽略它们。
- `run_vla`：先 `propose`，再用图纸尺寸场 `generate_mesh`，**然后** 第一次 `solve_mesh`（stage `first`）。
- 第一次求解之后必须走 `partitioner.revise`（计算结果 + 资源存量 → 新尺寸）。没有下一次决策就停，不要用残差 PSO 顶上。
- `run_vla` 只把 `project_feasible` 用在决策超预算的时候。`calibrate_measured` / `fit_surrogate` / `calibrate(` 不得出现在 `pipeline.py`。
- LLM 看 `render_drawing_png`（图纸），不看 `render_field_png`（von Mises）。没有 API 时不要去想 API。
- 区域是不规则多边形。盒子只留 AB2。
- 主方法：`allow_communication=False`，`allow_split=False`，`max_solves=2`。分裂只留 AB4，通信只留 AB5。
- 重网格用 `drawings_size_fn`（同一批多边形）。
- 对照至少报 e_E、e_qoi、N、N/B、图纸排序。不能只报一条误差曲线。

## 诚实

A2′ / H1–H4 数字在重跑落地前仍是改前战役。不要把新流程叙述套到旧表。
不要重启 2D RL 300×3。论文主文只报 3D。E1 不许说 v2 LP 单调。
不要提交 `solves/` 或 Grok-VM `locked.json`。

## 自检

```
grep -n propose visionamr/vla/pipeline.py   # 必须在第一次 solve_mesh 之前
grep revise visionamr/vla/pipeline.py       # 必须有
grep project_feasible visionamr/vla/pipeline.py
grep vm_node visionamr/vla/partition.py     # Scripted / LLM 画区路径不应读场
grep predicted_sizes visionamr/vla/pipeline.py  # 必须没有
grep fit_surrogate visionamr/vla/pipeline.py    # 必须没有
grep calibrate_measured visionamr/vla/pipeline.py  # 必须没有
```

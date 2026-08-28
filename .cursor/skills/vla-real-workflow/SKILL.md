---
name: vla-real-workflow
description: Real FEA meshing workflow for Visionamr VLA. Use when changing VLA, vision heads, size assignment, LP comparison, probes, or campaign accounting.
---

# VLA 按真实分析流程

现场不是「均匀探针 → 对着应力云图描区」。现场是两段。改 VLA、视觉头、尺寸、和 LP 对比时按这个 skill，不要退回旧顺序。

## 两段

```
读图（零次求解）     图纸定**相对粗细**（不规则区 + 每区一个尺寸 + remainder）
预算尺度（零次求解） Gmsh 单元数把整体尺度收到预算附近。排序不许改。
第一次求解           落在这张已画过的网格上（不是均匀 h0）
一次修订             (s,κ) PSO：**先验是眼睛的尺寸**
                     h_i = h_eye_i · exp(s + κ τ_i)
                     τ 来自实测残差密度。多边形不重画。
                     不许通信轮按 η 份额重谈，不许分裂，不许 LP 涂尺寸。
停                  一两轮定稿
对照                同预算一步 LP。轴：e_E、e_qoi、N/B、图纸排序还在不在。
                    不能只比误差。
```

画区 + 眼睛给尺寸 **禁止** 调用 CalculiX / `FemRunner.solve` / `post.vm_*` / `eta2`。
Gmsh 出网格不是求解。有了第一次求解之后，只用残差修订。
**最后一轮尺寸修订必须是 PSO**（`final_revision_pso` / `final_revision: "pso"`）。
PSO 已经写好了。适应度 = 上次实测 η² 份额 + 单元数缩放 N ~ h^{-d} + 硬预算（`calibrate_measured`）。
**禁止** 再拟合 `η ~ h^q`、禁止 `fit_surrogate` / `Surrogate.predict` / `E_pred`。那是 LP 的假设，不是 PSO 缺的一块。

## 为什么

能求解还去画，就被 Li–Bettess / ZZ 局部预测吃掉。旧 VLA 用探针场涂尺寸，k=2 结构性输给 LP。
人的优势是图纸通道（求解前、也在探针旁边），加上不把粗网格 \(\eta_K\) 当真理。
不是「人不用第一次求解」，也不是「人读云图比 LP 更会用 \(\eta\)」。

## 文献座位（不要搅在一起）

| 方法 | 看什么 | 不看什么 |
|---|---|---|
| 读图 / GReFEM | CAD 视图 + 荷载文字 | 这一次的 \(\eta\)；GReFEM 无闭环 |
| LP | 这一次求解的 \(\eta\) | 图纸上还没显影的东西 |
| Dörfler | 多次 \(\eta\) + 标记 | 少求解冠军；渐近最优归它 |
| MeshingNet | 几何 → 冻结 \(h\) | 误差合同；标签是离线求解买的 |

禁止把「未见拓扑 / 冻结母族」写成科学结论。监督弱在标签循环和探针信息瓶颈。

## 实现锁

- `partitioner.propose` 只依赖 `Problem`（几何、features、荷载说明）。`post` / `eta2` 可留参数，但画区和初尺寸必须忽略它们。
- `run_vla`：先 `propose`，再用图纸尺寸场 `generate_mesh`，**然后** 第一次 `solve_mesh`（stage `first`）。
- LLM 看 `render_drawing_png`（图纸），不看 `render_field_png`（von Mises）。
- 提示词写图纸，不写响应场 / 应力场。没画到的给 `remainder_fineness_fraction`。允许 `view=section`。
- 区域是不规则多边形，默认不是盒子。盒子只留 AB2。
- 第一次求解计入 k=1，可以成为交付解。不要把第一张网当成可丢的均匀探针。
- 主方法：`allow_communication=False`，`allow_split=False`，`max_solves=2`。分裂只留 AB4（`vla_ab4_split`），通信只留 AB5（`vla_ab5_comm`）。
- 修订 PSO 的 `h+` 必须是眼睛尺寸，不是通信轮输出，也不是「按 η 重新给的尺寸」。
- 重网格用 `drawings_size_fn`（同一批多边形），不要用旧网标签把图画糊掉。
- `run_vla` 只调用 `calibrate_measured`。`fit_surrogate` / `calibrate(` 不得出现在 `pipeline.py`。
- 对照至少报 e_E、e_qoi、N、N/B、图纸排序（rim 仍细于 remainder）。不能只报一条误差曲线。

## 诚实

A2′ / H1–H4 数字在重跑落地前仍是改前战役。不要把新流程叙述套到旧表。
不要重启 2D RL 300×3。论文主文只报 3D。E1 不许说 v2 LP 单调。
不要提交 `solves/` 或 Grok-VM `locked.json`。

## 自检

```
grep -n propose visionamr/vla/pipeline.py   # 必须在第一次 solve_mesh 之前
grep vm_node visionamr/vla/partition.py     # Scripted / LLM 画区路径不应读场
grep predicted_sizes visionamr/vla/pipeline.py  # 必须没有
grep fit_surrogate visionamr/vla/pipeline.py    # 必须没有
grep calibrate_measured visionamr/vla/pipeline.py  # 必须有
```

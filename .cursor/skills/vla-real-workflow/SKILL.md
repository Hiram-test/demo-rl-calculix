---
name: vla-real-workflow
description: Real FEA meshing workflow for Visionamr VLA. Use when changing VLA, vision heads, size assignment, LP comparison, probes, or campaign accounting.
---

# VLA 按真实分析流程

眼睛只做两件事：**判别**，和**调用工具**。
它不准，所以不能自己调参，也不能把尺寸委派出去。数字全是工具的。

循环是：判别 → 调工具 → 再判别 → 再调工具。

```
判别     画区；谁更密谁更疏（等级 1 最密 … 5 最疏，不是 h）
工具     先验映射（feats=None，评价 0）→ 出网 → 求解
判别     看占用和剩余；可改等级，仍不给 h
工具     一次闭式微调（评价 ≤1，不搜索）→ 出网 → 求解
```

第一次出网用 `GRADE_PRIOR`，**可以离谱超预算或欠预算**。
那不是要搜索掉的 bug，是下一次判别要看的东西。
不准的微调留给下一眼，像人一样。参数循环速度是核心。

`calibrate_grades` 第一次 `feats=None` 评价 = 0；有上次网格时评价 ≤ 1。
画区 + 判别 **禁止** 调用 CalculiX。视觉头默认走火山方舟 `doubao-seed-evolving`（OpenAI 兼容，配置见 README「视觉模型（VLA 视觉头）配置」：`ARK_API_KEY` 或 `VISIONAMR_ARK_CONFIG`）；没有 API 时回退脚本头、代理自己写判别 JSON。
**禁止** 眼睛输出 `fineness_fraction` / 缩放系数 / 连续尺寸。
**禁止** 眼睛委派参数（包括「交给 PSO 选 h」）。
**禁止** 把第一次 N 搜回预算：`scale_drawings_to_elem_budget` / Gmsh size-search / 粒子群不得出现在 live 路径或 `run_eye_vs_lp.py`。
**禁止** 再拟合 η ~ h^q、禁止 `fit_surrogate` / `E_pred`。
`calibrate_measured` 不得出现在 `pipeline.py`。
Scripted/LLM 没有 `revise`：第二次求解是同一套等级上的一次微调。改等级走 CachedDrawing 的 `revise` JSON。
多边形不重画。通信/分裂只留 AB5 / AB4。

对照：同预算一步 LP。轴：e_E、e_qoi、N/B、图纸排序还在不在。

## 诚实

A2′ / H1–H4 数字在重跑落地前仍是改前战役。不要把新流程叙述套到旧表。
不要重启 2D RL 300×3。论文主文只报 3D。E1 不许说 v2 LP 单调。
不要提交 `solves/` 或 Grok-VM `locked.json`。

## 自检

```
grep calibrate_grades visionamr/vla/pipeline.py
grep n_particles visionamr/vla/pso.py   # live calibrate_grades 不得用
grep calibrate_measured visionamr/vla/pipeline.py  # 必须没有
grep scale_drawings_to_elem_budget scripts/run_eye_vs_lp.py  # 必须没有
```

# 从论坛原始输入到真实求解反馈：DeepSeek 对 CalculiX 过盈接触不收敛的有限调用诊断

## 摘要

本文研究 DeepSeek 能否在不训练 DQN、不执行网格优化、不使用预设答案门禁的条件下，诊断一个真实 CalculiX 论坛过盈接触问题。实验取得并冻结原帖两个大型 C3D20R 输入文件，但因约一千一百六十万自由度、Pardiso 后端和标准 runner 资源限制，不冒充已执行原始大型模型。系统改用可快速运行的二次减缩积分接触代表模型，保留“边中间节点进入初始干涉后 cutback 失败”的机制，并让 DeepSeek在最多三次、同一增长消息历史中提出竞争假设、选择单因素实验、读取真实 ccx 反馈和形成最终答复。论文结论直接来自同一 trace 的 DeepSeek 最终决策，论文生成没有额外模型调用。

**关键词：** CalculiX；C3D20R；非线性接触；过盈；DeepSeek；反事实求解；证据追踪

## 1. 原始工程问题与附件证据

原帖为 [Interference Contact and mesh refinement](https://calculix.discourse.group/t/interference-contact-and-mesh-refinement/2747)。截至证据冻结时，帖子没有作者确认的最终修复。公开附件归档 SHA-256 为 `9AF9CE2F677384BF3AB2E44C188E92E71B47841D9C6C3C67C066DCF5AD8BDB9C`。

两个原始附件的关键事实如下：

| 输入 | 节点 | C3D20R 单元 | PIN2 接触面 | 步骤 | REMOVE/ADD PIN2 |
|---|---:|---:|---:|---:|---|
| `Shear_setup-INTER01-COARSE_PIN.inp` | 3836377 | 184896 | 1728 | 2 | False |
| `Shear_setup-INTER01-deactivate1thenreactivate.inp` | 3863869 | 232416 | 3344 | 3 | True |

这两个附件同时改变网格规模、PIN2 接触面数量和接触激活历史，不能直接作为“只因网格细化而失败”的因果对照。

## 2. 方法：应用级 Skill、真实求解和有限调用 loop

运行时使用 `nonlinear-contact-diagnosis` 应用级 Skill（SHA-256 `8E59BF60334F8CA14429F0DEB6F605177E86441715EA52215CA56C3DB986853B`）。Skill 只提供证据纪律：区分事实与解释、保留竞争假设、实验前声明正反预测、一次只改一个字段、把失败求解作为证据、保留适用边界。它不包含本案例的参数、工具顺序或答案。

整个运行只创建一个 DeepSeek client，SDK 重试为 0，模型 HTTP 请求上限为 3。每轮把模型原始 JSON 和工具结果追加到同一消息历史，既保留真实多轮上下文，也让稳定前缀具备缓存条件。非法 JSON 不触发付费修复请求。

## 3. 代表模型和初始可复现现象

代表模型来自 CalculiX 官方 `test/contact4.inp` 的两个二次实体接触块，统一为 C3D20R。正式对照使用 node-to-surface 以建立健康基线；这是机制实验替代，不是对原帖 surface-to-surface 的预设修复。对照与目标只改变上块接触面四个边中间节点的 z 坐标：1.00 对 0.95，四个角点均保持 z=1.00。

| SOLVE-ID | mid-z | 初始增量 | penalty | 接触 | 完成 | no convergence | 增量尝试 | 最大修正 |
|---|---:|---:|---:|---|---|---:|---:|---:|
| `SOLVE-F58C31B530EA` | 1.0 | 0.5 | default | NODE TO SURFACE | True | 5 | 2 | 0.00364587 |
| `SOLVE-DDCAEE99F171` | 0.95 | 0.5 | default | NODE TO SURFACE | False | 68 | 6 | 183725600000.0 |
| `SOLVE-F06394DCE1DB` | 0.95 | 0.5 | default | SURFACE TO SURFACE | True | 68 | 3 | 0.1837113 |

## 4. DeepSeek 决策链

### 第 1 轮

- 状态：`investigating`
- 证据判断：Control solve (mid_z=1.0, node‑to‑surface) converged; target solve (mid_z=0.95, node‑to‑surface, initial overclosure 0.05) failed with cutbacks and extreme displacement corrections. Both decks are identical except for edge‑midside node z‑coordinates. This isolates the effect of initial overclosure on the contact formulation.
- 选择动作：`run_surface_to_surface_pair`
- 正结果预测：Surface‑to‑surface solve converges (small corrections, normal force patterns); H1 strongly supported, H2/H3 become less likely.
- 负结果预测：Surface‑to‑surface solve also fails with similar extreme corrections; H1 is refuted, H2/H3 remain possible.
- 当前用户答复：Diagnosis in progress: representative model demonstrates that initial overclosure under node‑to‑surface contact diverges. Next step will run the same overclosure with surface‑to‑surface to test if the original contact formulation resolves the issue.
- 假设 `H1`，置信度 `0.5`：Initial overclosure causes divergence specifically under node‑to‑surface contact; switching to surface‑to‑surface (as in the original deck) handles the initial penetration numerically stably.
- 假设 `H2`，置信度 `0.3`：Penalty stiffness 69000 is too low for the initial overclosure of 0.05 length units, leading to large initial penetration and non‑convergence.
- 假设 `H3`，置信度 `0.2`：Initial increment 0.5 is too large for the sudden contact activation, causing failure; smaller increments allow gradual resolution of overclosure.

### 第 2 轮

- 状态：`investigating`
- 证据判断：Surface-to-surface solve (SOLVE-F06394DCE1DB) converged with 0.05 initial overclosure, while node-to-surface failed (SOLVE-DDCAEE99F171). This confirms that contact formulation alone can toggle convergence under the same geometric penetration. However, the original model already uses surface-to-surface, so the failure in the original must arise from other differences (geometry, mesh, penalty, increment, etc.). Original deck inspection is needed to reconcile.
- 选择动作：`inspect_original_deck_semantics`
- 正结果预测：Original deck reveals settings (e.g., node-to-surface mistakenly assigned, pressure-overclosure type differ, no auto-penetration adjustment) that explain why even surface-to-surface fails under refinement.
- 负结果预测：Original deck shows identical contact parameters to representative model; failure then likely due to larger overclosure magnitudes or mesh quality aspects not captured in the simplified geometry.
- 当前用户答复：Representative model evidence: with 0.05 overclosure, node-to-surface fails while surface-to-surface converges. However, your original deck already uses surface-to-surface, so the failure may come from other contact parameters or the refined geometry. I will inspect the original deck next to identify why surface-to-surface alone did not save your model.
- 假设 `H1`，置信度 `0.55`：In the representative model, initial overclosure causes divergence specifically under node-to-surface; surface-to-surface handles it numerically. Original model failure might involve similar penetrations but with different contact parameters (penalty, pressure-overclosure behavior, face matching) that also hinder convergence even with surface-to-surface.
- 假设 `H2`，置信度 `0.15`：Penalty stiffness 69000 is too low for the refined mesh overclosure in original model, causing non-convergence irrespective of contact type (surface-to-surface cannot compensate).
- 假设 `H3`，置信度 `0.1`：Initial increment 0.5 is too large for the refined mesh overclosure; a smaller increment would allow convergence even with node-to-surface or current surface-to-surface.

### 第 3 轮

- 状态：`narrowed_unresolved`
- 证据判断：Representative model experiments (SOLVE‑DDCAEE99F171 vs SOLVE‑F06394DCE1DB) demonstrate that under identical 0.05 initial overclosure, node‑to‑surface diverges while surface‑to‑surface converges. The original deck already uses surface‑to‑surface, yet the user reports failure after mesh refinement. The original attachments change mesh, contact faces and activation history simultaneously, preventing direct attribution. The failure in the original refined model likely arises from non‑uniform overclosure (e.g., corners and edges), face‑matching issues on curved geometry, or Pardiso‑specific contact iterations that are not captured by the representative experiment.
- 选择动作：`None`
- 正结果预测：None
- 负结果预测：None
- 当前用户答复：Diagnosis summary:
- In a simplified representative model with a flat rigid surface and an overclosure of 0.05 length units, node‑to‑surface contact fails with extreme cutbacks, while surface‑to‑surface converges without changing penalty, increment, or mesh.
- Your original model already uses surface‑to‑surface (SURFACE TO SURFACE), C3D20R elements, and a linear penalty stiffness of 69000. Despite these, mesh refinement in the cylindrical hole leads to non‑convergence.
- Because your attachments change mesh density, contact face count, and activation history simultaneously, it is impossible to isolate the root cause from the given files alone.
- Likely contributors: (1) non‑uniform initial overclosure (corners and edges may penetrate more than half an element width), (2) face‑matching difficulties on curved surfaces, (3) penalty stiffness 69000 being marginal for the larger penetrations after refinement, or (4) Pardiso solver‑specific contact iteration differences compared to the SPOOLES‑based representative test.
- Actionable steps for your original model: (a) Verify that the refined mesh does not introduce highly distorted elements or excessive local overclosure; (b) Increase the linear penalty stiffness (e.g., 1e5 or 1e6) and re‑run; (c) If the pressure‑overclosure curve is LINEAR, consider using a piece‑wise linear or exponential curve to better handle large penetrations; (d) Force a small initial increment (e.g., 0.01) and allow automatic cutbacks; (e) Check face orientation and ensure explicit face matching (e.g., using ADJUST=NO unless necessary).
- These suggestions are based on controlled experiments and known CalculiX behavior, but they are not a confirmed fix for your specific case. Please test them on the original model and report back.
- 假设 `H1`，置信度 `0.6`：Surface‑to‑surface contact handles initial overclosure more robustly than node‑to‑surface; original model failure is likely due to additional factors (excessive non‑uniform penetration, face mismatch on cylindrical surfaces, penalty stiffness).
- 假设 `H2`，置信度 `0.25`：Penalty stiffness 69000 is sufficient for 0.05 overclosure (representative model success proves it) but insufficient for the actual larger/non‑uniform overclosure in the refined original model.
- 假设 `H3`，置信度 `0.15`：Adaptive increment strategy (not initial increment) is the key; surface‑to‑surface representative model succeeded because it automatically cut back increments, while Pardiso/ original model may behave differently.

### 工具反馈

- 第 1 轮执行 `run_surface_to_surface_pair`，事件 `TOOL-959171E55942`，changed_fields=['contact_pair.type']。
- 第 2 轮执行 `inspect_original_deck_semantics`，事件 `TOOL-051D44A097DB`，changed_fields=[]。

## 5. DeepSeek 最终工程答复

**最终状态：** `narrowed_unresolved`

**以下答复为 trace 中 DeepSeek 最终原文，而非论文模板预写结论：**

Diagnosis summary:
- In a simplified representative model with a flat rigid surface and an overclosure of 0.05 length units, node‑to‑surface contact fails with extreme cutbacks, while surface‑to‑surface converges without changing penalty, increment, or mesh.
- Your original model already uses surface‑to‑surface (SURFACE TO SURFACE), C3D20R elements, and a linear penalty stiffness of 69000. Despite these, mesh refinement in the cylindrical hole leads to non‑convergence.
- Because your attachments change mesh density, contact face count, and activation history simultaneously, it is impossible to isolate the root cause from the given files alone.
- Likely contributors: (1) non‑uniform initial overclosure (corners and edges may penetrate more than half an element width), (2) face‑matching difficulties on curved surfaces, (3) penalty stiffness 69000 being marginal for the larger penetrations after refinement, or (4) Pardiso solver‑specific contact iteration differences compared to the SPOOLES‑based representative test.
- Actionable steps for your original model: (a) Verify that the refined mesh does not introduce highly distorted elements or excessive local overclosure; (b) Increase the linear penalty stiffness (e.g., 1e5 or 1e6) and re‑run; (c) If the pressure‑overclosure curve is LINEAR, consider using a piece‑wise linear or exponential curve to better handle large penetrations; (d) Force a small initial increment (e.g., 0.01) and allow automatic cutbacks; (e) Check face orientation and ensure explicit face matching (e.g., using ADJUST=NO unless necessary).
- These suggestions are based on controlled experiments and known CalculiX behavior, but they are not a confirmed fix for your specific case. Please test them on the original model and report back.

### 已实施动作

- run_surface_to_surface_pair
- inspect_original_deck_semantics

### 可使用范围

- Conclusive evidence from representative model that surface‑to‑surface contact robustly handles 0.05 initial overclosure that breaks node‑to‑surface.
- Original deck semantics confirming surface‑to‑surface, linear penalty, C3D20R, Pardiso – settings that should be stable but still fail in the user’s refined model.

### 不可使用范围

- Original large model verification – not run on Pardiso, full geometry, or with single‑factor changes.
- Claim that confirmed fix exists – forum did not confirm any solution before freeze.

### 模型引用的证据

- SOLVE-DDCAEE99F171
- SOLVE-F06394DCE1DB
- TOOL-959171E55942
- TOOL-051D44A097DB

## 6. 讨论与适用边界

本实验能评价 DeepSeek 是否会根据细微的 C3D20R 求解差异改判，不能证明原始大型圆柱孔模型已经修复。尤其需要同时考虑：原输入已经显式使用约 69000 的线性 penalty；代表模型的 penalty 结果可能只是数值 regularization；原附件还混入接触重新激活和接触面离散变化；当前 runner 后端不是原 deck 指定的 Pardiso。

只有在原几何、原 contact pair、批准后端和完整边界上复验位移、反力、接触压力及穿透后，候选措施才可升级为原问题的确认修复。

## 7. 调用、缓存与来源审计

- 实际 DeepSeek HTTP 请求：3 / 3
- API client 数量：1
- SDK 自动重试：0
- 缓存命中 token：15104
- 缓存未命中 token：4424
- 论文额外 DeepSeek 调用：0
- agent trace SHA-256：`B41AE78F1D550A450EE1E38E3EE1EE2B5504ABC7050E2DA2D62A0778C1F845A9`

## 参考资料

1. CalculiX Discourse, [Interference Contact and mesh refinement](https://calculix.discourse.group/t/interference-contact-and-mesh-refinement/2747).
2. CalculiX official repository, [test/contact4.inp](https://github.com/Dhondtguido/CalculiX/blob/master/test/contact4.inp).
3. CalculiX documentation, [Contact internals](https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node307.html).
4. CalculiX documentation, [*SURFACE INTERACTION](https://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/node249.html).

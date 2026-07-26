# AI 网格需求诊断与可执行 Skill：Model-aware MVP v0.3

## 目标

本原型不把“最大应力处继续加密”当作默认答案，也不把 PSO 当作所有网格问题的通用解法。它先把工程师的问题路由为可检验假设，再由模型、网格趋势、力/力矩平衡、能量一致性和重网格作用域证据进行保护层复核。

输出永远不是最终正确性裁决。`evidence_ledger.json` 只记录对明确用途与当前结论的有限证据，并允许后续证据推翻或修订。

## 六类需求与五类热点

需求族：

1. `resolution_convergence_budget`
2. `result_validity_and_extraction`
3. `topology_interface_load_transfer`
4. `geometry_layout_and_generation`
5. `global_local_and_model_fidelity`
6. `automation_and_repeatability`

热点语义：

1. `bounded_response_hotspot`
2. `qoi_sensitivity_or_error_hotspot`
3. `singular_or_artifact_hotspot`
4. `topology_or_geometry_event`
5. `none_or_unknown`

## 八个可执行 Skill

| Skill | 当前行为 |
| --- | --- |
| `bounded_hotspot_refinement` | 输出 Gmsh Box/Ball 背景尺寸场 |
| `qoi_guided_refinement` | 将 QoI 指标点输出为 Gmsh POS + PostView 尺寸场 |
| `singularity_guard` | 固定 QoI 已稳定而峰值持续增长时，禁止追逐原始峰值 |
| `topology_alignment` | 静态检查 CalculiX 节点、单元连通、重复坐标、未定义引用和点作用 |
| `geometry_mesh_repair` | 输出最小几何清理与重网格复核顺序 |
| `model_fidelity_switch` | 输出梁/壳/实体及全局—局部对照计划 |
| `mesh_replay_guard` | 比较 sets/loads/contacts/paths/supports/qoi_locations 是否漂移 |
| `convergence_verifier` | 评估固定 QoI 与原始峰值的受控网格趋势 |

## 方法选择边界

- 单个已经验证的有限热点：直接输出局部尺寸场，不运行搜索。
- 多个有效热点但无预算冲突：按确定性规则全部处理。
- 多热点竞争预算但区域近似独立：优先确定性分配。
- 只有“多个有效热点 + 真实预算冲突 + 已测得强耦合 + 固定 QoI”同时成立时，才输出 `external_hotspot_pso_job.json`。
- 奇异、连接、几何、模型层级或重网格作用域问题会阻止普通热点优化。

## 证据台账

状态仅允许：

- `supports_current_claim`
- `challenges_current_claim`
- `unresolved`
- `not_observed`

当前覆盖问题定义、AI/规则假设、模型连接与点作用、网格响应趋势、力/力矩平衡、能量一致性。每项都包含观察、局限和下一步检查。

## 本地验证

```bash
python -m pip install -r requirements-mesh-need.txt
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python scripts/validate_mesh_need_skills.py
```

当前自动测试不执行新的 Gmsh/CalculiX 求解；它验证路由、模型输入静态检查、保护层、控制文件生成、JSON 合同、浏览器/CLI 数据流和非最终证据台账。

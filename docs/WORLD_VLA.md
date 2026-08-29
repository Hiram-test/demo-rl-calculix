# World-model-guided VLA for bridge-component adaptive meshing

## Scope

This implementation adds an independent `wm_vla` path. It does not import or call `predicted_sizes()`, does not read local-prediction meshes, and does not use local-prediction trajectories as labels. The existing VLA and classical baselines remain unchanged.

The controller is deliberately narrow:


a one-shot visual head defines named structural regions and ordinal grades; a compact action-conditioned region-graph world model predicts the consequences of discrete grade changes; deterministic tools convert the selected action into numerical sizes, generate a candidate Gmsh mesh, and certify its exact free-equation count; CalculiX supplies the real transition used for online correction.

The visual model is outside the adaptive inner loop. It never emits a continuous mesh size. This design follows the cost lesson exposed by SimulCost: repeated LLM parameter adjustment can improve task completion, but it is slow and remains unreliable at high numerical accuracy. Counterfactual evaluation is therefore performed by a small local model, while numerical parameters remain owned by deterministic tools.

## State, action, and transition

For region `i`, the measured state contains:

- current grade and certified numerical size;
- regional ZZ error, element count, stress maximum and mean, realized element size, and physical volume;
- soft proximity to load, support, clamp, opening, and re-entrant-corner semantics;
- the region-adjacency graph;
- global estimator total, QoI error, exact equation count, solve index, and budget.

The only learned action is

\[
\Delta g_i\in\{-1,0,+1\},
\]

plus `STOP`. The world model predicts regional changes in `log(eta_i^2)`, regional element redistribution, and QoI-error change. The model is a mechanics-regularized bootstrap ensemble. It starts from signs and scaling consistent with refinement and `N~h^{-d}`, then updates only from real CalculiX transitions. Ensemble disagreement enters the planning objective as an uncertainty penalty.

The planner uses receding-horizon beam search. It may imagine several future solves, sparse multi-region refinement, and refine/coarsen resource transfers. Only the first action is executed. After the real solve, the imagined transition is replaced by measured evidence and the ensemble is refitted.

## Deterministic tool contract and MCP boundary

`tool_contract.py` exposes a strict JSON action schema suitable for an MCP wrapper. Undeclared fields and raw continuous parameters are rejected. The tool layer performs:

1. ordinal-grade validation;
2. deterministic grade-to-size mapping;
3. graph-gradation enforcement;
4. cheap resource projection;
5. Gmsh-only candidate generation;
6. exact free-equation counting from the generated topology and boundary conditions;
7. bounded global scale correction that preserves region ranking.

MCP can expose these functions without moving physical prediction into the LLM. Schema validity and exact parameter provenance are tool guarantees. Accuracy of the imagined physical consequence remains a world-model hypothesis and is checked by the next real solve.

## Medium-complexity bridge scenario

`bridge_pier_cap` is a three-dimensional concrete pier-cap subassembly containing:

- a cap beam fused to a pier column;
- two unequal bearing pressure patches;
- two longitudinal prestressing ducts;
- column-cap soffit re-entrant edges;
- fixed-base reaction edges.

The scenario is more informative than a single bearing block because load patches, duct rims, the column-cap junction, and the fixed base compete for a finite mesh budget. Actions in one region also alter neighboring mesh gradation and the global load-transfer solution. These cross-region effects are the intended test of the world model.

## Dörfler floor

The implementation does not declare superiority from one selected example. `dominance.py` pre-registers a non-inferiority release gate against the repository's exact element-level ZZ+Dörfler loop. Under the same equation cap and common real-solve horizon, release requires:

- final feasible energy error no more than 2% above Dörfler;
- geometric-mean matched-solve energy-curve error no more than 3% above Dörfler;
- final QoI error no more than 5% above Dörfler;
- every executed WM-VLA solve within the hard equation budget;
- at least two common measured solve points.

A failed gate remains a failed result. The guarded mode only adds an explicitly labelled region projection of Dörfler marking to the world-model action candidates; it is not presented as the pure method and is not equivalent to the exact element-level baseline.

## Reproducible command

```bash
python scripts/run_wm_vla_bridge3d.py \
  --budget 18000 \
  --max-solves 6 \
  --horizon 3 \
  --beam-width 24 \
  --guard off \
  --strict-gate \
  --out artifacts/wm_vla_bridge3d
```

The script runs WM-VLA, exact Dörfler, and local prediction in independent `FemRunner` instances. They share only the same reference solution. `comparison.json` contains every real solve, world state, imagined plan, selected action, mesh certificate, prediction residual, online update, and release-gate component.

# 3D action-support semantics benchmark

This pilot isolates one question: once refinement quality is reduced to a common accuracy-versus-resource objective, can a small LLM-generated action-support dictionary concentrate search on useful physical regions better than a fixed spatial patch dictionary?

## Mechanical model

The model is a 120 mm long, 40 mm × 40 mm three-dimensional cantilever with a transverse radius-8 mm through-hole centred at x = 35 mm. The x = 0 face is fully clamped. A total -1000 N z-direction load is applied through two fixed free-end CAD corner nodes on the positive-y side, producing bending plus torsion. The material is linear elastic with E = 210 GPa and ν = 0.30. Gmsh generates first-order tetrahedra and CalculiX solves one linear-static step.

The final v0.5 evaluation uses two mesh-invariant physical-point QoIs. The global QoI is vertical displacement `u_z` at the fixed material point (110, 0, 0) mm, away from the point-load singularities. The local QoI is the axial displacement difference `u_x(35,0,10)-u_x(35,0,-10)` across two fixed material points immediately above and below the hole. Both values are obtained by C3D4 barycentric interpolation, so the QoI sampling location does not change when the mesh changes.

## Frozen LLM supports

`llm_regions.json` contains six proposals generated before the numerical ranking was known: three originally conditioned on a global tip-displacement task and three on a local hole-deformation task. The model was given geometry, material, boundary conditions, loading, QoI wording and a one-action refinement budget. It was not given stress contours, element-wise error indicators, oracle rankings, the fine reference solution or element numbering. Each proposal is a mesh-independent union of one or more physical boxes plus a confidence and rationale. These six support geometries were kept frozen through all later numerical corrections.

The original selection prompt is retained in `llm_prompt.md` for provenance. The final physical-point QoIs differ slightly from the original nodal-average wording because v0.1 revealed that changing node sets confounded action-support quality with QoI sampling. The final benchmark therefore tests the robustness of the already-frozen region prior rather than claiming a clean estimate of QoI-conditioned LLM calibration.

## Common-resource comparison

The beam is divided into sixteen fixed 4 × 2 × 2 Cartesian atoms. v0.5 evaluates every single atom, all 120 unordered two-atom unions, and the same six frozen LLM supports. Before structural solution, every local support independently adjusts its local target mesh size until the generated mesh is as close as possible to a common target of 3000 translational DOF. The resulting candidate meshes are approximately resource matched; `results/budget_calibration.csv` records the local size and achieved DOF for every case. A globally fine 2.5 mm mesh with 33714 DOF supplies the numerical reference.

This design separates support geometry from refinement intensity. A large LLM region must use a coarser local target size than a small fixed patch to receive approximately the same resource budget.

## Current result

For the global interior vertical-displacement QoI, the best single fixed atom has relative error 0.1299, the best of all 120 fixed atom pairs has relative error 0.0910 at 2961 DOF, and `LLM_H2_hole_root_coupled` has relative error 0.0580 at 2994 DOF. The best frozen LLM support ranks first among the 120 exhaustive pairs plus six semantic candidates.

For the local axial-difference QoI, the best single fixed atom has relative error 0.0740, the best fixed atom pair has relative error 0.0286 at 3027 DOF, and `LLM_T2_root_band` has relative error 0.0249 at 2982 DOF. The best frozen LLM support again ranks first in the pair-plus-semantic comparison. Only 3 of 120 fixed pairs lie within 10% of the best fixed-pair error for the global QoI, and only 1 of 120 does so for the local QoI, indicating that high-value joint supports are sparse even in this small fixed dictionary.

The more interesting negative result is task calibration. The local-QoI winner is a proposal originally generated for the tip-displacement task, while the highest-confidence local proposal `LLM_H1_hole_local` performs substantially worse. The pilot therefore supports the weaker hypothesis that LLM structural priors can concentrate candidate supports toward useful coupled regions; it does not support the stronger claim that the model reliably understands which region is optimal for a stated QoI or that its self-reported confidence is calibrated.

## Interpretation and limits

The strongest recurring supports cover the root and the hole/transfer neighborhood jointly. Purely local hole refinement and the downstream eccentric-load-path candidate are less effective under the same DOF budget. In this example, the useful information appears to be a coarse relation about mechanically coupled action support, not simply hotspot location.

This remains a pilot. The fixed atom dictionary is coarse, the LLM regions use continuous box boundaries unavailable to that dictionary, only one geometry and load case is tested, and six frozen proposals are insufficient to estimate a true probability of near-optimal support recall. A probability claim requires repeated independent LLM sampling across perturbed but mechanically equivalent prompts and multiple geometries, followed by hit-rate estimation against an exhaustive or approximate support oracle.

## Files and reproduction

`run_experiment.py` is the original v0.1 driver. `run_experiment_v2.py` removes mesh-dependent QoI-node sets, `run_experiment_v3.py` defines the final fixed physical-point QoIs, `run_experiment_v4.py` normalizes every local support to a common DOF budget, and `run_experiment_v5.py` adds the exhaustive 120 two-atom controls. The current GitHub Actions workflow executes v0.5. Run `python experiments/action_support_3d/run_experiment_v5.py` on an Ubuntu environment with Gmsh and CalculiX installed to reproduce the final experiment. Compact numerical outputs are committed under `experiments/action_support_3d/results/`, including `results.csv`, `budget_calibration.csv`, `summary.md`, `manifest.json` and `pair_oracle_analysis.md`.

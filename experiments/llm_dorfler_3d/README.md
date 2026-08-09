# Minimal LLM semantic-gating experiment

This directory tests only one hypothesis: can an LLM-defined semantic support make an otherwise unchanged Dörfler AMR history more efficient for a fixed QoI?

## Mechanical model

The model is the same simple three-dimensional perforated cantilever used in the earlier exploratory work: length 120 mm, square 40 mm × 40 mm section, transverse through-hole of radius 8 mm centred at x = 35 mm, fully clamped at x = 0, and a total -1000 N z-load distributed over the positive-y half of the free end face. The material is linear elastic steel with E = 210 GPa and ν = 0.30. The evaluated QoI is the z-displacement at the fixed material point (110, 0, 0) mm.

## The only semantic intervention

Before any solve, the physical domain is described to the LLM using a fixed 4 × 2 × 2 Cartesian atom vocabulary. The LLM receives geometry, material, boundary condition, load, QoI, and a six-atom budget. It does not receive stress/error contours, element indicators, the reference solution, or later adaptive meshes. The frozen selection is stored in `llm_selection.json`.

The selected support is six of the sixteen atoms: the complete four-atom root slab x = 0..30 mm plus the positive-y two atoms of the x = 30..60 mm slab containing the hole. This is the only place where the LLM acts.

## Two methods

`global_dorfler` computes the same element indicator on every current tetrahedron and performs standard descending Dörfler bulk marking over the full mesh.

`llm_gated_dorfler` computes exactly the same element indicator on exactly the same solved state, but only elements whose centroids lie in the frozen six-atom semantic support are admitted to the Dörfler candidate pool. Dörfler then uses the same θ = 0.50 inside that pool.

Both methods use the same stress-jump element indicator, the same local remeshing operator, the same refinement factor, the same initial mesh, the same four mark-refine rounds, the same solver, and the same fine reference QoI. Therefore the intended independent variable is only the action support presented to Dörfler.

## Output

`results/history.csv` contains QoI error versus DOF for both adaptive histories. `results/marks.json` records the actual marked element identifiers at each round. `results/summary.md` places the two trajectories side by side. A useful semantic partition should move the error-versus-DOF trajectory down and/or left. No claim is made that the LLM predicts the optimal mesh or replaces the estimator.

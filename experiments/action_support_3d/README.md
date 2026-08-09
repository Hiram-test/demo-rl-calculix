# 3D action-support semantics benchmark

This experiment isolates one question: when the refinement objective is reduced to QoI accuracy versus computational resource, can an LLM propose better *objects of action* than element-scale or fixed spatial partitions?

## Mechanical model

The benchmark is a 120 mm long, 40 mm × 40 mm three-dimensional cantilever with a radius-8 mm transverse through-hole centred at x = 35 mm. The x = 0 face is fully clamped. A total -1000 N z-direction load is distributed over an eccentric patch on the x = 120 mm end, producing bending plus torsion. The material is linear elastic with E = 210 GPa and ν = 0.30. Gmsh generates first-order tetrahedra and CalculiX solves one linear-static step.

Two QoIs are evaluated from every identical solve. `tip_displacement` is the mean z-displacement of the eccentric load patch. `hole_opening` is the mean z-displacement of the upper hole arc minus the mean z-displacement of the lower hole arc. The first is global/compliance-like; the second is local and tied directly to the hole neighborhood.

## What the LLM is allowed to see

The frozen proposals in `llm_regions.json` were generated from geometry, material, boundary conditions, load, QoI and the fact that only one local refinement action is allowed. No stress contour, element-wise error indicator, oracle ranking, fine-mesh solution or element numbering is part of the semantic input. Each proposal is expressed as one or more overlapping mesh-independent geometric boxes plus a confidence and a physical rationale.

This separation is deliberate. If the LLM were given a stress/error map, success could collapse to reading a posterior hotspot. Here the test is whether task-conditioned structural relations can concentrate the candidate action space before an element-wise posterior selector is used.

## Baselines and numerical reference

The initial background target size is 10 mm. Every local action uses the same 4 mm target size inside its support. A globally fine 2.5 mm mesh supplies the numerical reference QoIs. Sixteen `oracle_atom` cases divide the full beam into a 4 × 2 × 2 Cartesian grid and refine one atom at a time; these atoms are evaluation probes and are never shown to the LLM.

The current v0.1 deliberately studies only one action at a time. The 16 atoms therefore provide a structured non-semantic sanity baseline, not a proof of the globally optimal subset over the full element power set. A later extension can enumerate atom pairs/triples or use branch-and-bound to approximate the budget-constrained support oracle.

## Output

`run_experiment.py` writes `results/results.csv`, `results/manifest.json` and `results/summary.md`. Every candidate reports node count, tetrahedron count, a three-DOF-per-node resource proxy, both QoIs, reference-relative errors and two Pareto flags. The useful first question is whether QoI-conditioned LLM proposals appear on or close to the error–DOF Pareto frontier more often than the spatial atoms, and whether the two QoIs induce different preferred supports.

A positive result is intentionally weaker than “the LLM found the optimal mesh.” It means that a small semantic candidate dictionary increased the chance of presenting a useful action support to a downstream AFEM, RL or explicit-search selector. A negative result is equally informative because the proposals are frozen before the numerical ranking is known.

## Reproduce locally

On Ubuntu, install Gmsh and CalculiX, then run `python experiments/action_support_3d/run_experiment.py` from any working directory. The GitHub Actions workflow performs the same command on `ubuntu-24.04` and uploads only the compact results plus solver logs needed for diagnosis.

# Annotation Schema

## Required fields

| Field | Meaning |
|---|---|
| Case ID | Stable identifier with solver-family prefix. |
| Source | Original forum thread. |
| Engineering intent | The physical or engineering analysis the user is trying to perform. |
| Surface symptom | The visible failure: mesh sensitivity, cutbacks, NaN, non-convergence, abnormal deformation, unsupported feature, or suspicious result. |
| User attempts | Changes already tried by the user or forum participants. |
| Deployment variables | Modeling decisions implicated by the discussion. |
| Deployment reading | Why the case cannot be faithfully reduced to a local debug pair. |
| Resolution status | Resolved, partially resolved, workaround reported, unresolved, or solver/version defect. |

## Preliminary taxonomy

### A. Physical abstraction

- dimensional reduction: 1D/2D/axisymmetric/3D
- shell, beam, membrane, or continuum representation
- rigid versus deformable body treatment
- quasi-static, implicit dynamic, explicit dynamic, eigenvalue buckling, or post-buckling analysis
- inclusion of geometric nonlinearity, plasticity, hyperelasticity, damage, fracture, flow, or multiphysics coupling

### B. Discretization

- element family and order
- reduced/full integration
- mixed finite-element spaces
- mesh topology, alignment, grading, refinement, and quality
- regularization length relative to mesh size
- geometry and boundary tagging

### C. Interaction and constitutive deployment

- contact pair topology and master/slave assignment
- node-to-surface versus surface-to-surface contact
- penalty stiffness, stick slope, friction, adjustment, overclosure, and initial interference
- constitutive parameters and admissibility
- incompressibility treatment and locking

### D. Boundary and loading design

- rigid-body suppression
- symmetry assumptions
- force versus displacement control
- load ramp, increment size, continuation, and initial guess
- cyclic symmetry, couplings, ties, MPCs, and rigid-body constraints

### E. Solver and execution route

- nonlinear algorithm and globalization
- linear solver and preconditioner
- parallelism and version-dependent behavior
- unsupported solver capability
- solution transfer after remeshing or adaptive refinement

### F. Verification

- analytical or benchmark comparison
- mesh-convergence behavior
- energy, equilibrium, reaction, and contact checks
- physical admissibility of the converged branch
- sensitivity to imperfections, mesh orientation, and solver configuration

## Suggested benchmark targets

A future model should be evaluated on more than whether it predicts the final forum fix. Useful targets include:

1. identifying which decisions are still under-specified;
2. separating local implementation defects from model-design uncertainty;
3. proposing a minimal discriminating experiment;
4. generating alternative deployments rather than one unconditional answer;
5. defining physical and numerical acceptance criteria before execution;
6. revising the model at the correct level after evidence is returned;
7. preserving unresolved uncertainty when the forum thread does not establish a validated solution.

## Research-user triad extension

The machine-readable [`research-user-triad.json`](research-user-triad.json) adds a deliberately small challenge layer above the source-case annotations:

| Field | Meaning |
|---|---|
| `pack_schema` | Version of the three-case challenge contract. |
| `selection_purpose` | The bounded generalization capability being evaluated. |
| `cases` | Exactly three primary cases; source supplements are not silently promoted into this set. |
| `case_id` | Stable reference to the corresponding `FEN-*` source record. |
| `archetype` | Phase-field physics, classic-to-nonclassic transfer, or multi-method composition. |
| `decision_focus` | The subtle modeling distinction an agent must identify before choosing an action. |
| `minimum_discriminating_evidence` | Evidence that must be generated or inspected before a conclusion is accepted. |
| `delivery_status` | Whether the record is only a source definition or has a complete runnable model and results. |
| `environment_boundary` | Persistent infrastructure that can be reused and research-model capabilities that remain unimplemented. |

This extension does not change the 30-case source corpus. It prevents a source pointer, a runnable reproducer, and a verified solution from being represented as the same maturity level.

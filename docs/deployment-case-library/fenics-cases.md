# FEniCS Cases

The entries below distinguish the forum user's visible symptom from a broader deployment-level reading. The interpretation is project annotation, not a claim about the terminology used by forum participants.

## FEN-001 — A 3D Hertz-contact example does not transfer directly to 2D

- **Source:** [Contact Mechanics — 2D problem](https://fenicsproject.discourse.group/t/contact-mechanics-2d-problem/4445)
- **Engineering intent:** Build a two-dimensional contact simulation by adapting a three-dimensional Hertzian rigid-indenter penalty example.
- **Surface symptom:** The modified code does not run, and the author initially asks what is wrong with the implementation.
- **User attempts:** Changed the domain, mesh mapping, boundary definitions, gap expression, and two-dimensional function spaces.
- **Deployment variables:** Plane-strain/plane-stress interpretation, analytical versus meshed indenter, gap definition, contact boundary measure, penalty formulation, and displacement constraints.
- **Deployment reading:** Dimensional reduction changes the mathematical contact model and its boundary representation; the task is not a mechanical translation of working code.
- **Resolution status:** Forum guidance focused on reformulating and simplifying the contact construction.

## FEN-002 — Hyperelastic contact diverges when deformation passes a mesh-dependent threshold

- **Source:** [Contact of hyperelastic with rigid body diverges by iteration 0](https://fenicsproject.discourse.group/t/contact-of-hyperelastic-with-rigid-body-diverges-by-iteration-0/6040)
- **Engineering intent:** Enforce contact between a hyperelastic body and planar rigid boundaries over large deformation.
- **Surface symptom:** PETSc SNES returns `DIVERGED_FNORM_NAN` at iteration zero after the imposed displacement exceeds a threshold related to nodal spacing.
- **User attempts:** Refined the mesh, changed body shape and boundary conditions, removed a symmetry condition, changed the linear solver, and eventually used PETSc TAO with a bound-constrained optimization solver.
- **Deployment variables:** Inequality/contact formulation, unstructured-mesh symmetry, displacement constraints, nonlinear solver class, admissible set, mesh geometry, and result-quality verification.
- **Deployment reading:** Switching from root finding to constrained optimization changes the mathematical deployment; a successful run still leaves the physical and numerical quality of the result to be established.
- **Resolution status:** Workaround reported with PETSc TAO; validation uncertainty remained explicit.

## FEN-003 — Darcy error decreases overall but rises on some refined meshes

- **Source:** [Error does not converge as mesh size is increased](https://fenicsproject.discourse.group/t/error-does-not-converge-as-mesh-size-is-increased-error-decreases-but-also-increases-sporadically/220)
- **Engineering intent:** Verify a Darcy-flow implementation by manufactured solutions and an L2 convergence study.
- **Surface symptom:** Error generally decreases with degrees of freedom but intermittently increases, preventing a clean convergence rate.
- **User attempts:** Varied mesh size, computed analytical/numerical derivatives, and inspected the variational formulation and error measure.
- **Deployment variables:** Manufactured forcing consistency, approximation space, differentiation/interpolation, quadrature, error norm, boundary conditions, and mesh sequence.
- **Deployment reading:** A convergence study is itself a designed numerical experiment; refinement alone cannot diagnose an inconsistent manufactured problem or measurement pipeline.
- **Resolution status:** Methodological debugging/verification discussion.

## FEN-004 — A phase-field fracture model works in DOLFIN but not after migration to DOLFINx

- **Source:** [Phase field model in dolfinx: did not converge](https://fenicsproject.discourse.group/t/phase-field-model-in-dolfinx-did-not-converge/7378)
- **Engineering intent:** Port a coupled elasticity/phase-field crack-propagation model from DOLFIN to DOLFINx.
- **Surface symptom:** The DOLFINx version fails to converge even though the earlier implementation runs.
- **User attempts:** Recreated Gmsh geometry, mesh tags, spaces, forms, and staggered/coupled solution steps in the new software stack.
- **Deployment variables:** API and form-semantics migration, mesh/tag ownership, function transfer, nonlinear iteration, boundary markers, and phase-field irreversibility handling.
- **Deployment reading:** Portability requires preserving the full simulation semantics and state-update protocol, not just translating syntax.
- **Resolution status:** Migration case with detailed community diagnosis; not reducible to a single mesh fix.

## FEN-005 — Elasto-plastic model converges under force control but fails under displacement control

- **Source:** [Not converging when applying Dirichlet BC instead of force](https://fenicsproject.discourse.group/t/not-converging-when-applying-dirichlet-bc-instead-of-force/10202)
- **Engineering intent:** Run a two-dimensional von Mises elasto-plastic test under prescribed displacement rather than traction.
- **Surface symptom:** The force-controlled model works, while an apparently equivalent displacement-controlled boundary condition does not converge.
- **User attempts:** Replaced the top-edge force with an incremented Dirichlet condition and retained the quadrature-based plasticity update.
- **Deployment variables:** Load control, incremental boundary-value updates, history-variable evolution, boundary application order, Newton linearization, and reaction interpretation.
- **Deployment reading:** Control mode is part of the constitutive integration path; replacing a load with a displacement is a model redesign, not a neutral input change.
- **Resolution status:** Forum diagnosis centered on consistent boundary/loading updates in the incremental algorithm.

## FEN-006 — A stiff inclusion makes a hyperelastic multi-material beam fail

- **Source:** [Subdomains with different materials, solver not converging](https://fenicsproject.discourse.group/t/subdomains-with-different-materials-solver-not-converging/14731)
- **Engineering intent:** Stretch a hyperelastic beam containing a square inclusion with different stiffness.
- **Surface symptom:** Newton reaches the maximum iteration count after assigning separate material behavior to subdomains.
- **User attempts:** Followed tutorial implementations for material subdomains and hyperelasticity, used a fine mesh, and assigned cell tags/material expressions.
- **Deployment variables:** Constitutive discontinuity, subdomain tagging, piecewise energy density, interface conformity, load magnitude, initial guess, and nonlinear globalization.
- **Deployment reading:** A correct multi-material deployment must preserve both variational consistency and interface/material mapping; the symptom does not identify which layer failed.
- **Resolution status:** Community troubleshooting case; requires checking the complete material/subdomain construction.

## FEN-007 — Non-Newtonian free-surface extrusion fails when pressure is added

- **Source:** [Don't understand why this solver is not converging](https://fenicsproject.discourse.group/t/dont-understand-why-this-solver-is-not-converging/13929)
- **Engineering intent:** Simulate die swell of molten polymer using a deforming mesh and a pressure condition that can be updated between iterations.
- **Surface symptom:** The model converges in a simpler setting but fails when the non-Newtonian option and pressure condition are combined.
- **User attempts:** Refined and smoothed the mesh, changed the initial guess, removed or modified velocity conditions, and tried different solvers and preconditioners.
- **Deployment variables:** ALE mesh motion, constitutive rheology, pressure/velocity compatibility, free-surface conditions, mixed spaces, nonlinear coupling, initialization, and continuation.
- **Deployment reading:** The engineering objective requires a consistent coupled formulation; generic mesh and solver changes cannot substitute for pressure-velocity-free-surface compatibility.
- **Resolution status:** Complex open diagnostic thread.

## FEN-008 — Large-strain poromechanics fails beyond a repeatable load level

- **Source:** [Newton Solver does not converge for nonlinear coupled field problem](https://fenicsproject.discourse.group/t/newton-solver-does-not-converge-for-nonlinear-coupled-field-problem/5141)
- **Engineering intent:** Solve a large-strain two-phase porous-medium problem coupling solid and fluid fields.
- **Surface symptom:** Small loads work, but beyond a repeatable load level the second Newton iteration deteriorates; reducing time step or changing mesh does not remove the threshold.
- **User attempts:** Used ramped stress loading, varied time step and mesh, and inspected intermediate fields for visible instability.
- **Deployment variables:** Coupled residual/Jacobian consistency, finite-strain kinematics, constitutive tangent, load continuation, field scaling, initial state, and block solver strategy.
- **Deployment reading:** A threshold tied to the coupled state suggests model-form, tangent, or branch-tracking issues rather than a generic lack of mesh density.
- **Resolution status:** Unresolved diagnosis in the available thread.

## FEN-009 — Adaptive phase-field fracture produces an unphysical crack path

- **Source:** [Challenges in reproducing phase-field fracture benchmark](https://fenicsproject.discourse.group/t/challenges-in-reproducing-phase-field-fracture-benchmark-unphysical-crack-trajectory-and-mesh-refinement-behavior/19720)
- **Engineering intent:** Reproduce a Mode-I adaptive phase-field fracture benchmark with a straight crack and controlled refinement band.
- **Surface symptom:** The crack becomes jagged or mixed-mode and the refined band is wider/larger than the reference result.
- **User attempts:** Changed regularization length, mesh size, boundary conditions, iteration method, mesh generation, curve tags, and refinement settings.
- **Deployment variables:** Gmsh boundary tagging, loading symmetry, phase-field regularization length, mesh-to-length-scale ratio, unstructured-mesh bias, AMR criterion, and benchmark interpretation.
- **Deployment reading:** The model can run and still realize the wrong fracture mode; deployment acceptance requires path physics and mesh-objective verification.
- **Resolution status:** Improved result reported after reformulation/reference implementation; residual asymmetry and mesh-resolution requirements were discussed.

## FEN-010 — A meshed rigid indenter requires a contact architecture not provided by simple demos

- **Source:** [Meshed rigid indenter in contact with a hyperelastic body in FEniCSx](https://fenicsproject.discourse.group/t/meshed-rigid-indenter-in-contact-with-a-hyperelastic-body-in-fenicsx/19698)
- **Engineering intent:** Push a geometrically arbitrary meshed rigid indenter into a hyperelastic body, later allowing cohesive penetration and newly created surfaces.
- **Surface symptom:** Existing examples cover analytical indenters, small deformation, tied interfaces, or contact methods that do not match the intended separation/penetration behavior.
- **User attempts:** Compared penalty demos, interface coupling, `dolfinx_mpc`, and third-medium contact concepts.
- **Deployment variables:** Analytical versus meshed geometry, deformable/rigid representation, finite-strain contact search, inequality enforcement, separation, cohesive-zone evolution, and evolving topology.
- **Deployment reading:** The central task is selecting or developing a feasible contact model architecture; no amount of local code repair supplies a missing mathematical capability.
- **Resolution status:** Requirement/formulation exploration; no turnkey implementation established.

## FEN-011 — Adaptive refinement breaks state continuity in phase-field simulation

- **Source:** [Error with Adaptive Mesh Refinement in Phase-Field Simulation](https://fenicsproject.discourse.group/t/error-with-adaptive-mesh-refinement-in-phase-field-simulation-fenics/17206)
- **Engineering intent:** Add adaptive remeshing to a hydrogen-assisted phase-field fracture simulation.
- **Surface symptom:** The first refinement changes the mesh, after which the solver raises a mesh/function-space key error.
- **User attempts:** Rebuilt function spaces after refinement but still faced problems with measures, boundary markers, and transferring previous solutions.
- **Deployment variables:** Mesh hierarchy, function-space recreation, state/history transfer, measure and tag reconstruction, crack irreversibility, and nonlinear continuation after remeshing.
- **Deployment reading:** Adaptive simulation deployment includes a state-migration protocol; generating a finer mesh is only one operation in that protocol.
- **Resolution status:** Open implementation/design question in the available discussion.

## FEN-012 — Coupled bulk/surface PDE fails even from the exact solution

- **Source:** [Why is my program not converging for this system of coupled PDEs?](https://fenicsproject.discourse.group/t/why-is-my-program-not-converging-for-this-system-of-coupled-pdes/2284)
- **Engineering intent:** Solve a mixed-dimensional system coupling a PDE on a circular boundary to a PDE in the interior.
- **Surface symptom:** Each decoupled problem works, but the coupled system does not converge even when initialized with the analytical solution.
- **User attempts:** Derived a weak form, used a mixed-dimensional branch, and verified the separate components.
- **Deployment variables:** Trace operators, bulk-surface function spaces, coupling signs, weak-form consistency, mixed-dimensional assembly, nullspaces, and solver block structure.
- **Deployment reading:** The exact initial guess cannot rescue an inconsistent coupled discretization; the coupling architecture must be validated independently.
- **Resolution status:** Variational-formulation investigation.

## FEN-013 — Phase-field crack boundary condition matches no mesh facets

- **Source:** [Phase Field Fracture](https://fenicsproject.discourse.group/t/phase-field-fracture/7476)
- **Engineering intent:** Initiate fracture from a prescribed pre-crack in a nontrivial polygonal specimen.
- **Surface symptom:** FEniCS warns that no facets match the crack boundary; after geometry changes, fracture still does not initiate from the intended tip.
- **User attempts:** Changed geometry, crack predicate, mesh resolution, and boundary conditions, and was advised to visualize tagged edges.
- **Deployment variables:** Geometric representation of an internal crack, facet versus point marking, tolerance, mesh conformity to the crack, phase-field initialization, and loading design.
- **Deployment reading:** The intended physical discontinuity was not actually represented in the discrete model; this is a deployment-semantic failure hidden behind a boundary-warning symptom.
- **Resolution status:** Partial diagnostic guidance; explicit visualization of tags and crack representation required.

## FEN-014 — Imported mesh causes a finite-strain model to stop converging

- **Source:** [Solver cannot converge](https://fenicsproject.discourse.group/t/solver-cannot-converge/14920)
- **Engineering intent:** Run a finite-strain structural model on a user-provided rod/fiber mesh.
- **Surface symptom:** A regular `BoxMesh` version works, while the imported mesh triggers nonlinear non-convergence; smaller load steps do not fix it.
- **User attempts:** Lowered load increments, shared the mesh, and compared with a regular structured mesh.
- **Deployment variables:** Imported element quality/orientation, boundary tagging, unit/scale consistency, finite-strain formulation, load application, and small-strain baseline testing.
- **Deployment reading:** Community advice first asks for a small-strain deployment to validate boundary and loading semantics before returning to finite strain, demonstrating staged model design.
- **Resolution status:** Diagnostic route proposed; imported-mesh cause not conclusively resolved in the available excerpt.

## FEN-015 — Hyperelastic tension fails only after many successful increments

- **Source:** [Newtonsolver does not converge — Hyperelasticity](https://fenicsproject.discourse.group/t/newtonsolver-does-not-converge-hyperelasticity/2949)
- **Engineering intent:** Perform a displacement-driven hyperelastic strain test in two or three dimensions.
- **Surface symptom:** The simulation progresses normally and then Newton fails after approximately the thirty-fourth load step.
- **User attempts:** Tested both 2D and 3D meshes and incrementally increased prescribed displacement.
- **Deployment variables:** Constitutive energy domain, element inversion, compressibility, load-step size, continuation, boundary constraints, and admissible deformation branch.
- **Deployment reading:** Late failure after a reproducible deformation level may indicate loss of admissibility or a bifurcation in the chosen formulation, not a generic solver bug.
- **Resolution status:** Nonlinear-model diagnostic case.

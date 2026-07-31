# CalculiX Cases

The entries below distinguish the forum user's visible symptom from a broader deployment-level reading. The interpretation is project annotation, not a claim about the terminology used by forum participants.

## CCX-001 — Interference contact becomes less solvable after refinement

- **Source:** [Interference Contact and mesh refinement](https://calculix.discourse.group/t/interference-contact-and-mesh-refinement/2747)
- **Engineering intent:** Resolve cylindrical interference fits, including large models with many spring/contact contributions.
- **Surface symptom:** A coarser pin mesh converges, while a slightly finer pin mesh can stall with negligible correction displacement and no useful residual evolution.
- **User attempts:** Changed incrementation, `NLGEOM`, master/slave order, contact activation, and node-to-surface versus surface-to-surface treatment.
- **Deployment variables:** Hex/quad quality, node alignment, initial overlap relative to element width, contact discretization, symmetry constraints, model reduction, and contact topology.
- **Deployment reading:** Refinement is not a monotonic repair operation because it changes how initial interference is represented and how contact constraints are assembled.
- **Resolution status:** Representative test cases isolated the behavior; no general validated fix was established in the thread.

## CCX-002 — Large nonlinear TET10 model reaches too many cutbacks

- **Source:** [Too many cutbacks — static analysis, NLGEOM on](https://calculix.discourse.group/t/too-many-cutbacks-static-analysis-ngleom-on/779)
- **Engineering intent:** Run a large-deformation structural analysis with more than 2.4 million nodes and quadratic tetrahedra.
- **Surface symptom:** A persistent high residual near one location and termination by too many cutbacks.
- **User attempts:** Strong local refinement, lower imposed strain, direct stepping, many increment limits, several time-step choices, and multiple sparse solvers.
- **Deployment variables:** Tetrahedral discretization, local singularity, MPC constraints, displacement imposition, geometric nonlinearity, model scale, solver memory, and load path.
- **Deployment reading:** The case cannot be reduced to tuning the nonlinear tolerance; the model representation, constraints, and loading strategy remain coupled and under-tested.
- **Resolution status:** Unresolved in the available discussion.

## CCX-003 — Snap-fit contact snags and produces excessive bending

- **Source:** [Snap-fit contact snagging problem](https://calculix.discourse.group/t/snap-fit-contact-snagging-problem/2277)
- **Engineering intent:** Simulate a deformable snap-fit sliding over a block and completing engagement.
- **Surface symptom:** The beam catches near contact transition and bends far more than expected, although a comparable Abaqus model can slide correctly.
- **User attempts:** Changed geometry, beam stiffness, friction, chamfer/radius, mesh density, element formulation, increment control, contact stiffness, stick slope, master/slave assignment, and rigid-body treatment.
- **Deployment variables:** Edge/contact topology, solver-specific contact capabilities, element formulation, mesh quality near the radius, friction regularization, control mode, and geometry idealization.
- **Deployment reading:** There is no unique local parameter defect; different complete deployments produce different contact paths and force predictions.
- **Resolution status:** Workarounds produced improved runs, but the discussion retained concerns about force accuracy and residual behavior.

## CCX-004 — Turbine shroud contact combined with cyclic symmetry

- **Source:** [Contact Modelling with Cyclic Symmetry for Turbine Blade Shroud Contacts](https://calculix.discourse.group/t/contact-modelling-with-cyclic-symmetry-for-turbine-blade-shroud-contacts/1491)
- **Engineering intent:** Reproduce a turbine-blade shroud contact model using a flying-face construction and cyclic symmetry.
- **Surface symptom:** An Abaqus modeling route produces warnings and first-increment convergence failure when transferred to CalculiX.
- **User attempts:** Combined C3D10 solids, copied nodes, membrane flying faces, contact pairs, friction, cyclic ties, and alternative split-volume ideas.
- **Deployment variables:** Solver feature support, overlapping constraints, membrane/continuum compatibility, cyclic boundary placement, automated model-generation constraints, and geometric redesign.
- **Deployment reading:** The main question is whether the original analysis concept is representable within CalculiX, or whether the model architecture must change.
- **Resolution status:** Alternative geometric arrangements were suggested; no direct flying-face equivalence was demonstrated.

## CCX-005 — Nonlinear buckling of an imperfect cylindrical shell

- **Source:** [Nonlinear buckling of a cylindrical shell](https://calculix.discourse.group/t/nonlinear-buckling-of-a-cylindrical-shell/1437)
- **Engineering intent:** Predict post-buckling collapse of an axially compressed imperfect cylindrical shell.
- **Surface symptom:** Static `NLGEOM` analyses often stop before completion or converge to a collapse pattern different from the expected one.
- **User attempts:** Force and displacement control, different imperfection amplitudes, plasticity, alternative boundary conditions, and comparison with Abaqus Riks results.
- **Deployment variables:** Imperfection shape, shell element choice, boundary constraints, post-limit-point solution method, plasticity, load control, and physical validation against analytical and commercial-solver results.
- **Deployment reading:** The target equilibrium branch depends on the entire analysis design; a converged solution can still represent the wrong branch or an artefact.
- **Resolution status:** Partially explored; CalculiX's lack of a Riks-type route remained a central limitation.

## CCX-006 — Reactor-skirt shell with openings fails in nonlinear compression

- **Source:** [Nonlinear Axial Compression (Buckling) of a Cylinder not Converging](https://calculix.discourse.group/t/nonlinear-axial-compression-buckling-of-a-cylinder-not-converging/4027)
- **Engineering intent:** Model nonlinear buckling of a reactor skirt with holes under axial compression.
- **Surface symptom:** Load- and displacement-controlled variants fail at roughly one percent axial shortening.
- **User attempts:** Linear-mode imperfection injection, load/displacement control, plasticity on/off, solver suspicion, and mesh review.
- **Deployment variables:** Regular versus geometry-derived mesh, shell normals, element type, imperfection field, top-structure stiffness, boundary conditions, GNIA/GMNIA staging, and unavailable stabilization or arc-length capabilities.
- **Deployment reading:** Forum advice explicitly rebuilds the analysis from a validated simple cylinder before adding holes and material complexity, which is design decomposition rather than local debugging.
- **Resolution status:** Ongoing; a staged validation route was proposed.

## CCX-007 — Rigid-body constraints destabilize nonlinear shell models

- **Source:** [Rigid-body constraint convergence problems](https://calculix.discourse.group/t/rigid-body-constraint-convergance-problems/369)
- **Engineering intent:** Apply a rigid coupling to a shell plate under geometrically nonlinear loading.
- **Surface symptom:** A solid model converges readily, while the analogous shell model is highly sensitive to reference-node position or fails.
- **User attempts:** Reduced the large model to three elements, compared shell and solid representations, and moved rigid-body reference nodes.
- **Deployment variables:** Shell expansion, rotational degrees of freedom, rigid coupling equations, reference-node placement, element representation, and geometric nonlinearity.
- **Deployment reading:** The choice of shell versus solid and the coupling construction determine the numerical model; the issue precedes any isolated error search.
- **Resolution status:** Minimal reproducer established the sensitivity; the thread investigates formulation behavior.

## CCX-008 — Soft elastomer/contact model fails while an unrealistically stiff model runs

- **Source:** [Model Convergence When Using Soft Material But not Hard](https://calculix.discourse.group/t/model-convergence-when-using-soft-material-but-not-hard/1946)
- **Engineering intent:** Simulate an elastomer stator with an interfering steel rotor and pressure-loaded cavities.
- **Surface symptom:** The model converges with a very high modulus but fails at rubber-like stiffness; a hyperelastic attempt also fails.
- **User attempts:** Changed Young's modulus and constitutive model while retaining interference, pressure loading, and constraints.
- **Deployment variables:** Hyperelastic constitutive choice, near-incompressibility, initial interference, contact stiffness, pressure path, rotor/stator constraints, and large-deformation equilibrium.
- **Deployment reading:** The unrealistic stiff material masks difficulties in the intended physical regime; success of the surrogate model does not validate the target deployment.
- **Resolution status:** Open diagnostic case in the available thread.

## CCX-009 — Shell-to-shell contact works only in an older CalculiX release

- **Source:** [Contact of two shell pipes — no convergence in newer ccx releases](https://calculix.discourse.group/t/contact-of-two-shell-pipes-no-convergence-in-newer-ccx-releases/2865)
- **Engineering intent:** Maintain a FreeCAD/CalculiX shell-contact regression example across solver versions.
- **Surface symptom:** A model that runs in CalculiX 2.16 terminates with too many cutbacks in later versions.
- **User attempts:** Coarse/fine, first/second-order, triangular/quadrilateral meshes; multiple contact options; forces versus prescribed displacement; and control changes.
- **Deployment variables:** Solver version, shell contact implementation, element order/topology, penalty stiffness, adjustment, contact type, and loading mode.
- **Deployment reading:** Some deployment choices are version-specific; an automated system must reason over a solver-capability/version matrix rather than treat the input deck as universally portable.
- **Resolution status:** Version regression/compatibility investigation.

## CCX-010 — Welding contact deck written without a clear contact model

- **Source:** [Contact of two sets](https://calculix.discourse.group/t/contact-of-two-sets/1584)
- **Engineering intent:** Establish contact between shelf and wall meshes in a welding-related model.
- **Surface symptom:** Immediate too-many-cutbacks failure; the author states that the input file was assembled without fully understanding the commands.
- **User attempts:** Defined surfaces, material, sections, boundary includes, and a surface-to-surface contact pair.
- **Deployment variables:** Correct surface-face selection, contact orientation, initial gap/penetration, constraint sufficiency, welding-process abstraction, and step design.
- **Deployment reading:** The missing object is a justified contact deployment, not merely the location of a syntax defect.
- **Resolution status:** Diagnostic forum case; requires reconstruction of model intent and contact definitions.

## CCX-011 — Spherical-shell buckling displays the mesh pattern

- **Source:** [Buckling of sphere](https://calculix.discourse.group/t/buckling-of-sphere/2774)
- **Engineering intent:** Obtain the buckling mode and pressure of a spherical shell.
- **Surface symptom:** Deformed/buckling patterns visibly follow the structured mesh for shell and solid sector models.
- **User attempts:** Half, eighth, and full models; shell and solid elements; cylindrical-coordinate boundaries; soft-spring supports; structured and proposed unstructured meshes.
- **Deployment variables:** Symmetry reduction, boundary constraints, soft stabilization, mesh orientation, shell/solid choice, mode multiplicity, and analytical critical-pressure comparison.
- **Deployment reading:** For a highly symmetric structure, the mesh and support strategy select among near-degenerate modes; avoiding an artefact requires experimental design and verification.
- **Resolution status:** Full-model/soft-spring and unstructured-mesh approaches were explored; mode-shape regularity remained sensitive.

## CCX-012 — Orthotropic shell refinement makes convergence worse

- **Source:** [Orthotropic Shell/Plate Analysis](https://calculix.discourse.group/t/orthotropic-shell-plate-analysis/1086)
- **Engineering intent:** Analyze an orthotropic shell/plate with nonlinear traction or pressure loading.
- **Surface symptom:** Quadratic shell midside nodes distort; refinement can make failure occur earlier unless the transverse modulus is increased.
- **User attempts:** Compared S8R and S4R behavior, varied transverse stiffness, refined the mesh, and changed traction versus pressure loading.
- **Deployment variables:** Admissible orthotropic constants, 3D stiffness positive-definiteness, shell element support, compression/buckling of midside geometry, load implementation, and through-thickness behavior.
- **Deployment reading:** Material-card validity and element formulation jointly define the simulation; finer meshes do not repair an incompatible constitutive/discretization deployment.
- **Resolution status:** Forum discussion identified formulation/material restrictions; no universal configuration was established.

## CCX-013 — Penalty contact yields NaN or negative Jacobians after adjustment

- **Source:** [NAN in Pardiso solver — penalty contact method](https://calculix.discourse.group/t/nan-in-pardiso-solver-the-penalty-contact-method/4045)
- **Engineering intent:** Replace an unsatisfactory tied interface with penalty contact in a large static model.
- **Surface symptom:** Pardiso/PaStiX report NaN; enabling contact adjustment can distort elements enough to produce negative Jacobians.
- **User attempts:** Switched from tie to penalty contact, varied adjustment tolerance over several orders of magnitude, and changed sparse solver.
- **Deployment variables:** Initial surface mismatch, mesh alignment, adjustment distance, contact versus tie semantics, element distortion, and solver robustness.
- **Deployment reading:** The interface geometry and intended kinematics must be redesigned or justified before solver substitution can help.
- **Resolution status:** Active thread; disabling or limiting adjustment was discussed, with no final validated model reported.

## CCX-014 — Parallel sparse solver gives inconsistent buckling factors

- **Source:** [Buckling analysis using Spooles](https://calculix.discourse.group/t/buckling-analysis-using-spooles/2051)
- **Engineering intent:** Compute repeatable eigenvalue buckling factors using parallel execution.
- **Surface symptom:** Repeated runs of the same input produce radically different first-mode factors when multithreaded Spooles is used.
- **User attempts:** Repeated the benchmark and varied processor count/solver configuration.
- **Deployment variables:** Sparse solver implementation, thread count, mixed precision, reproducibility, mode ordering, and solver/version selection.
- **Deployment reading:** Execution configuration is part of deployment provenance; identical FE input does not guarantee a stable computational experiment.
- **Resolution status:** Solver defect/known unsafe configuration; single-threaded Spooles or other solver configurations were recommended.

## CCX-015 — Result-driven automatic refinement destroys element quality

- **Source:** [Mesh Refinement based on results](https://calculix.discourse.group/t/mesh-refinement-based-on-results/2259)
- **Engineering intent:** Automate mesh refinement using CalculiX result/error-estimator fields through a Python/Cubit workflow.
- **Surface symptom:** Repeated refinement quickly creates poor-quality elements even though the error indicator changes.
- **User attempts:** Refined from result fields and compared with CalculiX's built-in tetrahedral refinement and geometry-based manual strategies.
- **Deployment variables:** Error indicator, remeshing algorithm, element-quality constraints, geometry-aware refinement zones, supported element families, and stopping criteria.
- **Deployment reading:** Mesh optimization requires a constrained design policy; following the largest indicator alone can degrade the discretization and invalidate later solves.
- **Resolution status:** Open methodology discussion; geometry-based and built-in alternatives were proposed.

# Cross-passage pure-torsion five-method benchmark

This isolated experiment compares five regional mesh-decision strategies on one normalized box-space-truss abstraction of a repeated catwalk cross-passage under prescribed pure twist: uniform mesh, pure discrete PSO, coarse-hotspot reduced PSO, graph multi-agent coordination, and online Double DQN with a two-layer GCN encoder.

The structural topology contains six transverse stations, four longitudinal chord lines, six rectangular station frames, and X bracing on both side faces plus the top and bottom faces. Sixteen graph regions control the member subdivisions: five chord-panel regions, six station-frame regions, and five bracing-panel regions. B31 beam elements are used so that curved-member refinement has bending and torsional stiffness and does not introduce the artificial mechanisms produced by subdivided pin-jointed truss elements.

The right end receives the small-rotation translation field `ux=-theta(z-zc)` and `uz=theta*x`; the left end is fixed in all six beam degrees of freedom. Every candidate is solved by real CalculiX. A hidden 16-subdivision-per-member reference evaluates torque error, total internal-energy error, central displacement-field error, and top-region hotspot recall. The final mesh of every method is restricted to the element count of the uniform four-subdivision mesh. Search methods receive 32 unique candidate CalculiX solves; duplicate configurations are cached and do not consume the budget.

All material, section, imperfection, and geometric dimensions are normalized benchmark parameters. The experiment supports algorithmic conclusions about resource allocation on a multi-truss graph; it does not publish engineering stress, stiffness, safety, or capacity values for the Zhangjinggao catwalk cross-passage.

The workflow commits `results/RESULTS.md`, `results/results.json`, CSV tables, figures, and every candidate input/result receipt back to the experiment branch after a successful run.

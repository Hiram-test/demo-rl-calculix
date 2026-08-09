# Frozen selection prompt specification

Use this specification when repeating the semantic-selection stage with another model. Do not provide solver contours, element indicators, fine-mesh results or oracle rankings.

You are selecting candidate supports for one local h-refinement action in a three-dimensional linear-elastic finite-element model. The structure is a 120 mm long cantilever with a 40 mm × 40 mm square cross-section and a transverse radius-8 mm through-hole centred at x = 35 mm. The x = 0 face is fully fixed. A total -1000 N load in z is distributed over a small patch on the x = 120 mm end that is offset toward positive y, so the response combines bending and torsion. Material: E = 210 GPa and ν = 0.30. The current mesh is deliberately coarse and approximately uniform. A local action refines its support to one common smaller target size. Resource cost grows with the volume and geometric complexity of the selected support.

For the specified QoI, propose exactly three candidate action supports ranked by probability of yielding a good accuracy/resource trade-off. Supports may overlap and may contain more than one connected box. Do not force a partition of the whole structure. Keep at least one lower-confidence alternative to preserve uncertainty. For each candidate return: `id`, `confidence` in [0,1], `rationale`, and one or more axis-aligned boxes `{xmin,xmax,ymin,ymax,zmin,zmax}` in millimetres. The boxes define where one refinement action acts; they do not identify elements.

QoI A: mean z-displacement over the eccentric load patch.

QoI B: mean z-displacement of the upper hole boundary arc minus mean z-displacement of the lower hole boundary arc.

Generate the three candidates for each QoI independently so that task conditioning is observable.

"""Vision-driven region-based adaptive mesh refinement research framework.

All remeshing goes through Gmsh: a decision (per-element target sizes or
per-region sizes) becomes a size field, and Gmsh regenerates the mesh.
CalculiX solves every mesh; no hand-written mesh manipulation exists in
this package.
"""

__version__ = "0.1.0"

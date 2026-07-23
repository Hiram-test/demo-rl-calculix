"""Launch the TQZ batch after normalizing Gmsh labels for CalculiX."""
from __future__ import annotations

import meshpilot_tqz_backend as backend
from meshpilot_tqz_dense_tags import renumber_support_mesh

_original_parse_msh2_tetra = backend.parse_msh2_tetra


def _parse_dense_mesh(filepath):
    return renumber_support_mesh(_original_parse_msh2_tetra(filepath))


# run_support_analysis resolves this module global at call time, so patching here
# fixes reference, coarse, cold and transfer analyses without changing PSO logic.
backend.parse_msh2_tetra = _parse_dense_mesh

from meshpilot_tqz_batch import main  # noqa: E402


if __name__ == "__main__":
    main()

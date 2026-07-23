"""Launch the TQZ batch with CalculiX-compatible mesh labels and cards."""
from __future__ import annotations

import meshpilot_tqz_backend as backend
from meshpilot_tqz_dense_tags import (
    renumber_support_mesh,
    write_calculix_deck,
)

_original_parse_msh2_tetra = backend.parse_msh2_tetra


def _parse_dense_mesh(filepath):
    return renumber_support_mesh(_original_parse_msh2_tetra(filepath))


# run_support_analysis resolves these module globals at call time.  The patch
# therefore applies to reference, coarse, cold and transfer analyses while the
# PSO and engineering resultants remain unchanged.
backend.parse_msh2_tetra = _parse_dense_mesh
backend._write_deck = write_calculix_deck

from meshpilot_tqz_batch import main  # noqa: E402


if __name__ == "__main__":
    main()

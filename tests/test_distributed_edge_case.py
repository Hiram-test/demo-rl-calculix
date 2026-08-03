from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_distributed_edge_comparison import build_distributed_nodal_loads  # noqa: E402


class DistributedEdgeLoadTests(unittest.TestCase):
    def test_resultant_is_preserved_for_all_meshes(self) -> None:
        for nx, ny in [(40, 8), (80, 16), (160, 32), (320, 64)]:
            with self.subTest(nx=nx, ny=ny):
                loads = build_distributed_nodal_loads(nx, ny)
                self.assertTrue(math.isclose(sum(loads.values()), -1000.0, abs_tol=1e-9))
                self.assertEqual(len(loads), int(round(2.5 / (20.0 / ny))) + 1)

    def test_coarsest_aligned_mesh_has_two_endpoint_forces(self) -> None:
        loads = build_distributed_nodal_loads(40, 8)
        self.assertEqual(len(loads), 2)
        self.assertEqual(sorted(loads.values()), [-500.0, -500.0])

    def test_rejects_non_aligned_segment(self) -> None:
        with self.assertRaises(ValueError):
            build_distributed_nodal_loads(20, 4)


if __name__ == "__main__":
    unittest.main()

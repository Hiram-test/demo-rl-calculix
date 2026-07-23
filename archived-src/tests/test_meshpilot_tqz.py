from pathlib import Path
import tempfile
import unittest

from meshpilot_pso import ObjectiveValue
from meshpilot_tqz_backend import (
    TQZCase,
    TQZMeshSpec,
    block_dimensions,
    compute_nodal_loads,
    load_resultants,
    parse_frd_displacements,
    parse_msh2_tetra,
    patch_boxes,
)
from meshpilot_tqz_batch import (
    FamilyRequest,
    normalized_descriptors,
    order_cases,
    warm_start_is_acceptable,
)


class TQZBackendTests(unittest.TestCase):
    def _case(self):
        return TQZCase(
            case_id="tqz4500_ag020",
            bearing_model="TQZ(XII)-4500-0.3P",
            nominal_vertical_capacity_kN=4500.0,
            ag=0.20,
            A=950.0,
            B=640.0,
            C=770.0,
            D=300.0,
            H=175.0,
        )

    def test_six_patch_partition_covers_effective_footprint(self):
        case = self._case()
        spec = TQZMeshSpec()
        boxes = patch_boxes(case, spec)
        self.assertEqual(len(boxes), 6)
        self.assertAlmostEqual(min(box[0] for box in boxes), -case.C / 2, places=3)
        self.assertAlmostEqual(max(box[1] for box in boxes), case.C / 2, places=3)
        self.assertAlmostEqual(min(box[2] for box in boxes), -case.D / 2, places=3)
        self.assertAlmostEqual(max(box[3] for box in boxes), case.D / 2, places=3)
        length, width, depth = block_dimensions(case, spec)
        self.assertGreater(length, case.A)
        self.assertGreater(width, case.B)
        self.assertEqual(depth, spec.block_depth_mm)

    def test_nodal_loads_reproduce_force_and_height_moment(self):
        case = self._case()
        nodes = {
            1: (-300.0, -100.0, 600.0),
            2: (-100.0, 100.0, 600.0),
            3: (100.0, -100.0, 600.0),
            4: (300.0, 100.0, 600.0),
        }
        loads = compute_nodal_loads(case, nodes, nodes)
        total_fx, total_fz, moment_y = load_resultants(loads, nodes)
        self.assertAlmostEqual(total_fx, case.horizontal_force_n, places=5)
        self.assertAlmostEqual(total_fz, -case.vertical_force_n, places=5)
        self.assertAlmostEqual(moment_y, -case.horizontal_force_n * case.H, places=2)
        self.assertTrue(all(value[2] < 0.0 for value in loads.values()))

    def test_msh_and_frd_parsers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            msh = root / "fixture.msh"
            msh.write_text(
                "\n".join(
                    [
                        "$MeshFormat",
                        "2.2 0 8",
                        "$EndMeshFormat",
                        "$Nodes",
                        "4",
                        "1 0 0 0",
                        "2 1 0 0",
                        "3 0 1 0",
                        "4 0 0 1",
                        "$EndNodes",
                        "$Elements",
                        "1",
                        "1 4 2 1 1 1 2 3 4",
                        "$EndElements",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            mesh = parse_msh2_tetra(msh)
            self.assertEqual(len(mesh.nodes), 4)
            self.assertEqual(mesh.tetrahedra[1], (1, 2, 3, 4))

            frd = root / "fixture.frd"
            frd.write_text(
                " -4  DISP\n -1 1 1.0E-03 2.0E-03 -3.0E-03\n -3\n",
                encoding="utf-8",
            )
            displacement = parse_frd_displacements(frd)
            self.assertEqual(displacement[1], (1.0e-3, 2.0e-3, -3.0e-3))


class TQZBatchTests(unittest.TestCase):
    def test_request_contract_and_order(self):
        request = FamilyRequest.from_json("examples/meshpilot_tqz_support_family.json")
        self.assertEqual(len(request.cases), 6)
        self.assertEqual(request.pso.max_level, 3)
        self.assertEqual(len(request.mesh.mesh_levels_mm), 4)
        descriptors = normalized_descriptors(request.cases)
        self.assertEqual(set(descriptors), {case.case_id for case in request.cases})
        for descriptor in descriptors.values():
            self.assertEqual(len(descriptor), 7)
            self.assertTrue(all(0.0 <= value <= 1.0 for value in descriptor))
        ordered = order_cases(request.cases, request.batch_order)
        self.assertEqual({case.case_id for case in ordered}, {case.case_id for case in request.cases})
        self.assertEqual(len(ordered), 6)

    def test_warm_guard_prefers_feasibility_then_bounded_degradation(self):
        coarse = ObjectiveValue((0,) * 6, 0.10, 0.08, 9000, True)
        good = ObjectiveValue((1,) * 6, 0.11, 0.07, 9500, True)
        bad = ObjectiveValue((2,) * 6, 0.14, 0.09, 9500, True)
        infeasible = ObjectiveValue((3,) * 6, 0.05, 0.03, 25000, False, 0.4)
        self.assertTrue(warm_start_is_acceptable(good, coarse, 1.20))
        self.assertFalse(warm_start_is_acceptable(bad, coarse, 1.20))
        self.assertFalse(warm_start_is_acceptable(infeasible, coarse, 1.20))
        self.assertTrue(warm_start_is_acceptable(coarse, infeasible, 1.20))


if __name__ == "__main__":
    unittest.main()

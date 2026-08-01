from __future__ import annotations  # Keep type annotations deferred on the workflow Python runtime.

import shutil  # Copy the version-controlled compatibility backend into the verified V4 payload.
import sys  # Read the one required payload-root argument.
from pathlib import Path  # Resolve source and target paths without shell-dependent string handling.

ASSET_ROOT = Path(__file__).resolve().parent  # Locate the version-controlled patch assets.
MARKER = "# ENTITY_CALCULIX_ENVIRONMENT_V1"  # Make every module override idempotent and auditable.


def _append_once(path: Path, block: str) -> None:  # Append one compatibility override without modifying original code bodies.
    text = path.read_text(encoding="utf-8")  # Read the exact unpacked V4 source.
    if MARKER in text:  # Treat an existing marker as an already-applied patch.
        return  # Keep repeated workflow application byte-stable.
    path.write_text(text.rstrip() + "\n\n" + block.strip() + "\n", encoding="utf-8")  # Append only the backend alias block.


def _replace_once(path: Path, old: str, new: str) -> None:  # Apply one exact compatibility correction inside the new backend only.
    text = path.read_text(encoding="utf-8")  # Read the installed backend source.
    if new in text:  # Treat the corrected implementation as already applied.
        return  # Keep repeated patch application idempotent.
    count = text.count(old)  # Count the exact legacy expression before replacing it.
    if count != 1:  # Refuse an ambiguous source drift.
        raise RuntimeError(f"expected one compatibility marker in {path}, found {count}")  # Preserve a precise packaging failure.
    path.write_text(text.replace(old, new, 1), encoding="utf-8")  # Replace only the intended numerical expression.


def _patch_backend_compatibility(path: Path) -> None:  # Adapt the new backend to current Gmsh, CalculiX, and NumPy contracts.
    _replace_once(  # Replace scalar planar cross product used for triangle orientation.
        path,  # Patch only the newly installed backend module.
        "        twice_area = float(np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0]))",  # Match the NumPy 2.4-compatible expression.
        "        first_edge = coordinates[1] - coordinates[0]\n        second_edge = coordinates[2] - coordinates[0]\n        twice_area = float(first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0])",  # Use the explicit two-dimensional determinant.
    )
    _replace_once(  # Replace vectorized planar cross products used for triangle areas.
        path,  # Patch only the newly installed backend module.
        "    return 0.5 * np.abs(np.cross(coordinates[:, 1] - coordinates[:, 0], coordinates[:, 2] - coordinates[:, 0]))",  # Match the deprecated vectorized expression.
        "    first_edges = coordinates[:, 1] - coordinates[:, 0]\n    second_edges = coordinates[:, 2] - coordinates[:, 0]\n    return 0.5 * np.abs(first_edges[:, 0] * second_edges[:, 1] - first_edges[:, 1] * second_edges[:, 0])",  # Use the explicit batched determinant.
    )
    _replace_once(  # Replace scalar planar cross product used by mesh-quality calculations.
        path,  # Patch only the newly installed backend module.
        "        area = 0.5 * abs(float(np.cross(coordinates[1] - coordinates[0], coordinates[2] - coordinates[0])))",  # Match the deprecated quality expression.
        "        first_edge = coordinates[1] - coordinates[0]\n        second_edge = coordinates[2] - coordinates[0]\n        area = 0.5 * abs(float(first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0]))",  # Preserve the same signed-area magnitude.
    )
    _replace_once(  # Bind the embedded crack curve to the single imported material surface explicitly.
        path,  # Patch only the new crack mesher.
        '            "Curve{crack_line} In Surface{domain[]};",',  # Match the list expression rejected by Gmsh 4.12.
        '            "Curve{crack_line} In Surface{domain[0]};",',  # Select the one exact intact-plate surface returned by the bounding-box query.
    )
    _replace_once(  # Keep every CalculiX node-set data line within its sixteen-entry parser limit.
        path,  # Patch only the new linear deck writer.
        '    if fixed_nodes:\n        lines.extend(["*NSET,NSET=FIXED", ",".join(str(node) for node in fixed_nodes)])',  # Match the unbounded single-line node-set writer.
        '    if fixed_nodes:\n        lines.append("*NSET,NSET=FIXED")\n        for start in range(0, len(fixed_nodes), 16):\n            lines.append(",".join(str(node) for node in fixed_nodes[start : start + 16]))',  # Emit deterministic sixteen-node chunks accepted by CalculiX 2.21.
    )


def main() -> int:  # Install the numerical environment behind the original V4 Skills.
    if len(sys.argv) != 2:  # Require one explicit hash-verified payload root.
        raise SystemExit("usage: apply_entity_calculix_environment.py <payload-root>")  # Reject ambiguous target directories.
    root = Path(sys.argv[1]).resolve()  # Normalize the unpacked V4 source root.
    if not (root / "bridge_agent" / "runtime.py").is_file():  # Confirm this is the original AgentRuntime payload.
        raise FileNotFoundError(root / "bridge_agent" / "runtime.py")  # Refuse to patch an unrelated directory.
    source = ASSET_ROOT / "entity_calculix_environment.py"  # Resolve the checked-in compatibility backend.
    if not source.is_file():  # Require the backend asset before changing any imports.
        raise FileNotFoundError(source)  # Preserve a clear packaging failure.
    backend_path = root / "bridge_agent" / "entity_calculix_environment.py"  # Resolve the installed backend target.
    shutil.copyfile(source, backend_path)  # Install exact Gmsh and CalculiX execution code.
    _patch_backend_compatibility(backend_path)  # Apply only environment-level compatibility corrections.
    _append_once(  # Redirect original geometry helpers while preserving every caller and Skill contract.
        root / "bridge_agent" / "meshers.py",  # Patch only the old numerical mesh implementation module.
        f'''{MARKER}
from .entity_calculix_environment import (
    central_crack_graded_q4_exact as central_crack_graded_q4,
    central_crack_q4_exact as central_crack_q4,
    circular_hole_q4_exact as circular_hole_q4,
    multi_hole_tri_mesh_exact as multi_hole_tri_mesh,
    rectangle_q4_exact as rectangle_q4,
)''',  # Preserve original public function names exactly.
    )
    _append_once(  # Redirect adaptive and budget-controlled mesh generation behind the original regional Skills.
        root / "bridge_agent" / "adaptive_mesh.py",  # Patch only the numerical meshing implementation module.
        f'''{MARKER}
from .entity_calculix_environment import (
    adaptive_region_tri_mesh_exact as adaptive_region_tri_mesh,
    crack_uniform_mesh_matching_budget_exact as crack_uniform_mesh_matching_budget,
    uniform_mesh_matching_budget_exact as uniform_mesh_matching_budget,
    uniform_tri_mesh_exact as uniform_tri_mesh,
)''',  # Keep PSO arguments, stopping depth, and Agent control untouched.
    )
    _append_once(  # Redirect the original linear finite-element call to a real CalculiX solve.
        root / "bridge_agent" / "fem.py",  # Patch only the solver implementation module.
        f'''{MARKER}
from .entity_calculix_environment import calculix_solve_linear as solve_linear''',  # Preserve the original solve_linear signature.
    )
    _append_once(  # Redirect the crack plastic sequence to the exact-seam CalculiX implementation.
        root / "bridge_agent" / "calculix_plastic.py",  # Patch only the crack solver implementation module.
        f'''{MARKER}
from .entity_calculix_environment import run_elastoplastic_crack_exact as run_elastoplastic_crack''',  # Preserve the original Skill-facing function name.
    )
    print(f"Applied exact Gmsh/OpenCASCADE and CalculiX environment to {root}")  # Record the only intended modification boundary.
    return 0  # Report success after all four backend aliases are installed.


if __name__ == "__main__":  # Run only when invoked by the isolated workflow.
    raise SystemExit(main())  # Propagate installation failures to GitHub Actions.

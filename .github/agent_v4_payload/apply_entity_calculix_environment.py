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


def main() -> int:  # Install the numerical environment behind the original V4 Skills.
    if len(sys.argv) != 2:  # Require one explicit hash-verified payload root.
        raise SystemExit("usage: apply_entity_calculix_environment.py <payload-root>")  # Reject ambiguous target directories.
    root = Path(sys.argv[1]).resolve()  # Normalize the unpacked V4 source root.
    if not (root / "bridge_agent" / "runtime.py").is_file():  # Confirm this is the original AgentRuntime payload.
        raise FileNotFoundError(root / "bridge_agent" / "runtime.py")  # Refuse to patch an unrelated directory.
    source = ASSET_ROOT / "entity_calculix_environment.py"  # Resolve the checked-in compatibility backend.
    if not source.is_file():  # Require the backend asset before changing any imports.
        raise FileNotFoundError(source)  # Preserve a clear packaging failure.
    shutil.copyfile(source, root / "bridge_agent" / "entity_calculix_environment.py")  # Install exact Gmsh and CalculiX execution code.
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

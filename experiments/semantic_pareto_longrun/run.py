from __future__ import annotations  # Enable modern postponed type annotations for a small wrapper module.
import importlib.util  # Load the validated full-domain semantic-partition benchmark without copying its numerical core.
from pathlib import Path  # Resolve repository-relative paths portably on the GitHub Actions runner.
import sys  # Register the dynamically loaded module and propagate explicit failures to the shell.

ROOT = Path(__file__).resolve().parent  # Anchor all long-run cases and compact results to this dedicated experiment directory.
BASE_PATH = ROOT.parent / "semantic_partition_dorfler_v2" / "run.py"  # Reuse the corrected full-domain semantic-partition implementation as the numerical base.
CASES_DIR = ROOT / "cases"  # Keep the many transient long-run meshes and solver files outside the compact result directory.
RESULTS_DIR = ROOT / "results"  # Store only long-run tables, plots, fits, and concise diagnostics here.
LONG_RUN_ROUNDS = 12  # Produce thirteen actual solved points per method, including the shared round-zero coarse state.
LONG_RUN_MIN_H = 1.50  # Allow substantially finer local refinement than the earlier 3.2 mm floor so the trajectories can continue evolving.
LONG_RUN_REFERENCE_H = 1.50  # Use a finer global reference consistent with the smallest local target used by the long-run adaptive histories.

spec = importlib.util.spec_from_file_location("semantic_partition_base", BASE_PATH)  # Build an import specification for the already validated benchmark implementation.
if spec is None or spec.loader is None:  # Reject an unexpected repository state before any Gmsh or CalculiX work starts.
    raise RuntimeError(f"Unable to load semantic-partition benchmark from {BASE_PATH}")  # Fail explicitly when the validated base implementation cannot be imported.
base = importlib.util.module_from_spec(spec)  # Create the isolated Python module object that will host the imported benchmark implementation.
sys.modules[spec.name] = base  # Register the module so its postponed annotations and dynamic core import resolve correctly.
spec.loader.exec_module(base)  # Execute the validated benchmark module without invoking its command-line main block.
base.CASES_DIR = CASES_DIR  # Redirect the base benchmark transient cases into the dedicated long-run experiment directory.
base.RESULTS_DIR = RESULTS_DIR  # Redirect the base benchmark compact outputs into the dedicated long-run result directory.
base.AMR_ROUNDS = LONG_RUN_ROUNDS  # Extend the corrected semantic and global histories from five adaptive rounds to twelve.
base.core.CASES_DIR = CASES_DIR  # Redirect the shared Gmsh and CalculiX numerical core to the long-run case directory as well.
base.core.RESULTS_DIR = RESULTS_DIR  # Redirect any shared-core compact outputs to the same long-run result directory.
base.core.AMR_ROUNDS = LONG_RUN_ROUNDS  # Keep the shared numerical core synchronized with the wrapper's extended round count.
base.core.MIN_LOCAL_H = LONG_RUN_MIN_H  # Lower the persistent local-refinement floor so later adaptive rounds can still create new mesh states.
base.core.REFERENCE_H = LONG_RUN_REFERENCE_H  # Refine the global numerical reference so late adaptive points are not judged against the earlier coarse reference.


def main() -> int:  # Execute the corrected full-domain semantic partition for a much longer adaptive horizon.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  # Ensure compact long-run outputs have a destination before the first native solver call.
    CASES_DIR.mkdir(parents=True, exist_ok=True)  # Ensure transient long-run cases have a destination before the first mesh is generated.
    return base.main()  # Run the same global and semantic algorithms with only the horizon, local floor, and reference resolution extended.


if __name__ == "__main__":  # Execute the long-run benchmark only when this wrapper file is called as the program entry point.
    try:  # Wrap the outermost call solely to keep GitHub Actions failure propagation explicit.
        raise SystemExit(main())  # Run the complete long-horizon numerical benchmark and return its shell status unchanged.
    except Exception as exc:  # Catch unexpected setup, meshing, solving, parsing, or remeshing failures at the command boundary.
        print(f"[fatal-longrun] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)  # Emit one concise failure reason into the persisted Actions console log.
        raise  # Re-raise the original exception so GitHub Actions records a real numerical failure rather than a false success.

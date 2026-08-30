#!/usr/bin/env python3  # Execute the frozen four-way runner with the active Python interpreter.
"""Run or dry-run the manifest-bound WMVLA-4WAY-P1 benchmark."""  # Describe the command's complete responsibility.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.
import argparse  # Import explicit command-line parsing.
import json  # Import strict machine-readable terminal output.
from pathlib import Path  # Import portable repository and campaign paths.
import sys  # Import local-package path injection and explicit exit codes.
ROOT = Path(__file__).resolve().parents[1]  # Locate the repository independently from the invocation directory.
sys.path.insert(0, str(ROOT))  # Import this checked-out package without installation or another commit.
from visionamr.vla.four_way_benchmark import ALL_METHODS, BUDGETS, BenchmarkRequest, FrozenInputError, run_benchmark  # Import the sole validated execution boundary.

def _parser() -> argparse.ArgumentParser:  # Build the complete controlled-shard command-line contract.
    parser = argparse.ArgumentParser(description="Run the frozen manifest-bound four-way bridge benchmark or perform a solve-free preflight.")  # Create a self-describing parser.
    parser.add_argument("--root", type=Path, default=ROOT / "results" / "wm_vla_four_way_p1", help="Campaign root containing protocol, training, references, and test outputs.")  # Permit an isolated campaign location without changing scientific settings.
    parser.add_argument("--manifest", type=Path, default=None, help="Validated case_manifest.json; defaults to <root>/protocol/case_manifest.json.")  # Permit explicit immutable-input selection.
    parser.add_argument("--frozen-config", type=Path, default=None, help="Frozen configuration; defaults to <root>/protocol/frozen_config.json.")  # Permit explicit freeze-document selection.
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test", help="Manifest split; blind test is the default.")  # Restrict selection to manifest-owned split labels.
    parser.add_argument("--case-id", action="append", default=[], help="Controlled shard case ID; repeat as needed; default is the complete selected split.")  # Permit exact case shards without resampling.
    parser.add_argument("--method", action="append", default=[], help=f"Method label; repeat as needed; rl expands three seeds; choices are {ALL_METHODS} or rl.")  # Permit exact method shards while retaining canonical order.
    parser.add_argument("--budget", type=int, action="append", default=[], help=f"Registered equation budget; repeat as needed; choices are {BUDGETS}.")  # Permit exact budget shards without new operating points.
    parser.add_argument("--dry-run", action="store_true", help="Validate all selected frozen inputs and print the exact plan without creating results or invoking solvers.")  # Provide the mandatory solve-free preflight mode.
    parser.add_argument("--resume", action="store_true", help="Skip only jobs with matching complete status markers; never overwrite failed or partial evidence.")  # Support controlled restart without duplicate trajectories.
    parser.add_argument("--allow-development-run", action="store_true", help="Permit a non-test solver smoke run; never relaxes blind-test freeze checks.")  # Require explicit authorization for train or validation execution.
    parser.add_argument("--allow-unqualified-references", action="store_true", help="Use operational non-qualified Reference B only when sealed frozen_config also explicitly allows it.")  # Require visible runtime acknowledgement of a frozen reference-threshold waiver.
    return parser  # Return the complete parser.

def main() -> int:  # Parse, validate, optionally execute, and report one invocation.
    parser = _parser()  # Construct the immutable command-line vocabulary.
    args = parser.parse_args()  # Parse caller-supplied paths and controlled filters.
    campaign_root = args.root.resolve()  # Normalize the campaign root for unambiguous artifacts.
    manifest = (args.manifest or campaign_root / "protocol" / "case_manifest.json").resolve()  # Resolve the exact manifest input.
    frozen_config = (args.frozen_config or campaign_root / "protocol" / "frozen_config.json").resolve()  # Resolve the exact freeze document.
    methods = tuple(args.method) if args.method else ALL_METHODS  # Default to WM, LP, supervised, all RL seeds, and independent Dörfler.
    budgets = tuple(args.budget) if args.budget else BUDGETS  # Default to all three preregistered resource caps.
    request = BenchmarkRequest(root=campaign_root, manifest_path=manifest, frozen_config_path=frozen_config, split=args.split, case_ids=tuple(args.case_id), methods=methods, budgets=budgets, dry_run=bool(args.dry_run), resume=bool(args.resume), development_run=bool(args.allow_development_run), allow_unqualified_references=bool(args.allow_unqualified_references))  # Assemble one immutable execution request with explicit reference-waiver intent.
    try:  # Convert preflight failures into concise machine-readable terminal evidence.
        result = run_benchmark(request)  # Perform complete preflight and optional method execution.
    except (FrozenInputError, FileNotFoundError, ValueError, json.JSONDecodeError) as exception:  # Catch configuration, artifact, manifest, and filter failures before uncontrolled tracebacks.
        payload = {"protocol_id": "WMVLA-4WAY-P1", "completed": False, "error_type": type(exception).__name__, "error": str(exception)}  # Build a bounded explicit failure result.
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), file=sys.stderr)  # Report failure without claiming any scientific result.
        return 2  # Signal preflight or request failure distinctly.
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))  # Emit the complete dry-run plan or execution summary.
    return 0 if result.get("failed_job_count", 0) == 0 else 3  # Distinguish a complete run from retained numerical method failures.

if __name__ == "__main__":  # Execute only when launched as a command.
    raise SystemExit(main())  # Propagate the explicit status to CI or the shell.

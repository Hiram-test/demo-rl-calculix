#!/usr/bin/env python3  # Execute the freeze guard with the active reviewed Python interpreter.
"""Create, dry-run, preflight, or verify the WMVLA-4WAY-P1 freeze bundle."""  # Describe the command's complete responsibility.
from __future__ import annotations  # Postpone annotation evaluation for supported Python runtimes.

import argparse  # Parse explicit campaign, Git, inventory, and receipt locations.
import json  # Emit strict machine-readable workflow evidence.
import os  # Read GitHub run metadata without exposing unrelated environment variables.
from pathlib import Path  # Resolve repository-relative defaults portably.
import sys  # Make the local package importable and report protocol failures on stderr.

ROOT = Path(__file__).resolve().parents[1]  # Locate the checked-out repository independently of launch directory.
sys.path.insert(0, str(ROOT))  # Import the reviewed local visionamr implementation without installation.

from visionamr.vla.four_way_freeze import FreezeError, create_freeze, preflight_report, seal_freeze_tag, verify_freeze  # Import the complete public freeze boundary and non-self-referential tag seal.


def _parser() -> argparse.ArgumentParser:  # Build a CLI that exposes operational paths but no scientific tuning knobs.
    parser = argparse.ArgumentParser(description="Create or verify the frozen WMVLA-4WAY-P1 pre-blind-test bundle.")  # Create a self-describing top-level parser.
    subparsers = parser.add_subparsers(dest="command", required=True)  # Require one explicit non-mutating or mutating mode.
    preflight = subparsers.add_parser("preflight", help="Scan for disclosure and report solve-free readiness without requiring completed training.")  # Define the PR-safe early guard.
    preflight.add_argument("--root", type=Path, default=ROOT / "results" / "wm_vla_four_way_p1", help="Campaign root to inspect without mutation.")  # Select only the campaign location.
    preflight.add_argument("--repository", type=Path, default=ROOT, help="Git repository containing the reviewed implementation.")  # Select the source-provenance worktree.
    preflight.add_argument("--implementation-commit", help="Optional complete expected implementation commit; defaults to current HEAD.")  # Permit workflow review of an explicitly selected SHA.
    preflight.add_argument("--receipt", type=Path, help="Optional path receiving the same strict-JSON readiness report.")  # Preserve CI dry-run evidence outside terminal logs.
    create = subparsers.add_parser("create", help="Validate explicit artifacts and create the one-shot immutable freeze bundle.")  # Define the complete freeze operation.
    create.add_argument("--root", type=Path, default=ROOT / "results" / "wm_vla_four_way_p1", help="Campaign root containing protocol, training, and partition inputs.")  # Select only the campaign location.
    create.add_argument("--repository", type=Path, default=ROOT, help="Git repository whose exact implementation commit is frozen.")  # Select the source-provenance worktree.
    create.add_argument("--config-source", type=Path, required=True, help="Reviewed JSON containing scientific settings and explicit model/evidence/cost artifact inventories.")  # Require explicit artifact selection without filesystem globbing.
    create.add_argument("--implementation-commit", required=True, help="Complete Git SHA that must exactly equal current HEAD during creation.")  # Bind the freeze to a committed implementation.
    create.add_argument("--dry-run", action="store_true", help="Perform all input, hash, Git, and disclosure checks without writing outputs.")  # Support full solve-free freeze rehearsal.
    create.add_argument("--receipt", type=Path, help="Optional path receiving the creation plan or completion record.")  # Preserve workflow evidence explicitly.
    verify = subparsers.add_parser("verify", help="Recompute every protected hash before blind execution.")  # Define the immutable-bundle verifier.
    verify.add_argument("--root", type=Path, default=ROOT / "results" / "wm_vla_four_way_p1", help="Frozen campaign root to authenticate.")  # Select only the campaign location.
    verify.add_argument("--repository", type=Path, default=ROOT, help="Git repository containing the dedicated freeze commit.")  # Select the source-provenance worktree.
    verify.add_argument("--require-committed", action="store_true", help="Require a clean descendant freeze commit with every protected artifact tracked.")  # Enforce the blind-launch Git boundary.
    verify.add_argument("--allow-postfreeze-test-references", action="store_true", help="Permit only authenticated cache JSON, ref_lNN.log, and failed-level ref_lNN.inp files under manifest test cases; the caller must immediately verify each cache.")  # Resolve independent post-freeze reference construction without permitting method results.
    verify.add_argument("--receipt", type=Path, help="Optional path receiving exact verification and workflow-run evidence.")  # Preserve the pre-blind receipt for artifact upload.
    seal = subparsers.add_parser("seal-tag", help="Create the fixed freeze tag after committing the complete freeze bundle.")  # Define the non-self-referential freeze-commit finalization step.
    seal.add_argument("--root", type=Path, default=ROOT / "results" / "wm_vla_four_way_p1", help="Committed frozen campaign root to authenticate and seal.")  # Select only the campaign location.
    seal.add_argument("--repository", type=Path, default=ROOT, help="Clean Git repository whose current HEAD is the dedicated freeze commit.")  # Select the exact commit receiving the immutable tag.
    seal.add_argument("--receipt", type=Path, help="Optional path receiving exact tag, HEAD, code, environment, and artifact evidence.")  # Preserve the finalization record for review.
    return parser  # Return the complete operational command contract.


def _github_metadata() -> dict[str, str | None]:  # Capture only non-secret GitHub Actions identifiers required by the protocol.
    names = ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_WORKFLOW", "GITHUB_WORKFLOW_REF", "GITHUB_JOB", "GITHUB_SHA", "GITHUB_REF")  # Enumerate auditable workflow and selected-revision identifiers.
    return {name.lower(): os.environ.get(name) for name in names}  # Preserve explicit absence outside GitHub Actions.


def _write_receipt(path: Path, payload: dict[str, object]) -> None:  # Persist one deterministic strict-JSON workflow receipt.
    path.parent.mkdir(parents=True, exist_ok=True)  # Create only the explicitly requested receipt directory.
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")  # Publish complete finite machine-readable evidence.


def main() -> int:  # Execute the selected freeze mode and emit one strict JSON result.
    args = _parser().parse_args()  # Parse operational inputs without scientific overrides.
    try:  # Convert protocol failures into a concise nonzero command result.
        if args.command == "preflight":  # Run the PR-safe solve-free readiness scan.
            result = preflight_report(args.root, args.repository, args.implementation_commit)  # Inspect Git, manifest readiness, and disclosure only.
        elif args.command == "create":  # Run the complete explicit artifact freeze or its exact dry-run.
            result = create_freeze(args.root, args.config_source, args.implementation_commit, args.repository, dry_run=bool(args.dry_run))  # Authenticate all inputs and optionally publish the one-shot bundle.
        elif args.command == "verify":  # Run the exact-byte verifier before a blind dispatch shard.
            result = verify_freeze(args.root, args.repository, require_committed=bool(args.require_committed), allow_postfreeze_test_references=bool(args.allow_postfreeze_test_references))  # Recompute every protected identity, optional Git boundary, and strict post-freeze reference exception.
        else:  # Authenticate a clean dedicated freeze commit and create its fixed lightweight tag once.
            result = seal_freeze_tag(args.root, args.repository)  # Eliminate commit-SHA self-reference while binding the immutable ref exactly to HEAD.
        result["github_actions"] = _github_metadata()  # Attach run and selected-revision identifiers without secrets.
        if args.receipt is not None:  # Preserve the exact terminal evidence when explicitly requested.
            _write_receipt(args.receipt, result)  # Write the same strict JSON payload used for automation.
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))  # Emit one transparent machine-readable completion record.
        if args.command == "preflight" and (result.get("TEST_NOT_RUN") is not True or result.get("implementation_matches_head") is not True):  # Fail PR dry-run on disclosure or an ambiguous/mismatched reviewed implementation identity, not unfinished training.
            return 2  # Signal a protocol-contaminated campaign to CI.
        return 0  # Signal successful validation, creation, or verification.
    except FreezeError as exc:  # Catch only expected protocol-integrity failures.
        print(f"freeze protocol error: {exc}", file=sys.stderr)  # Report an actionable failure without a noisy internal traceback.
        return 2  # Stop workflow execution before any blind solve.


if __name__ == "__main__":  # Execute only when launched as a command rather than imported by tests.
    raise SystemExit(main())  # Propagate the explicit protocol status to CI or the shell.

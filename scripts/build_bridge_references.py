#!/usr/bin/env python3  # Execute the formal Reference A/B campaign with the active reviewed interpreter.
"""Plan or build one complete frozen bridge-manifest reference split."""  # Describe the command's sole complete-split responsibility.
from __future__ import annotations  # Postpone annotation evaluation for supported Python runtimes.

import argparse  # Parse explicit split and operational evidence locations without scientific knobs.
import json  # Emit one strict machine-readable plan or terminal campaign record.
from pathlib import Path  # Resolve repository-relative defaults and caller-selected storage roots.
import sys  # Make the local package importable and report protocol failures concisely.

ROOT = Path(__file__).resolve().parents[1]  # Locate the reviewed repository independently from the launch directory.
sys.path.insert(0, str(ROOT))  # Import the checked-out implementation without requiring installation.

from visionamr.vla.four_way_freeze import FreezeError  # Catch committed-freeze authorization failures explicitly.
from visionamr.vla.four_way_reference_campaign import ALLOWED_SPLITS, ReferenceCampaignError, run_reference_campaign  # Import the no-subset formal driver and its public failure contract.


def _parser() -> argparse.ArgumentParser:  # Build the command surface with no case, seed, tolerance, or schedule selectors.
    parser = argparse.ArgumentParser(description="Plan or execute the complete Reference A/B campaign for exactly one frozen manifest split.")  # Create a self-describing top-level parser.
    parser.add_argument("--split", choices=ALLOWED_SPLITS, required=True, help="Exact complete manifest split; case subsets are intentionally unsupported.")  # Require an explicit train, validation, or test selection.
    parser.add_argument("--root", type=Path, default=ROOT / "results" / "wm_vla_four_way_p1", help="Formal campaign root containing protocol/case_manifest.json and references/.")  # Permit only the campaign storage location to vary.
    parser.add_argument("--repository", type=Path, default=ROOT, help="Reviewed Git worktree used for committed test-reference freeze authorization.")  # Select the source-provenance repository explicitly.
    parser.add_argument("--work-root", type=Path, help="Optional per-case native runner root; test paths must be outside the Git repository.")  # Allow operational relocation of bulky native evidence without changing science.
    parser.add_argument("--summary-dir", type=Path, help="Optional authenticated aggregate directory; test paths must be outside the Git repository.")  # Allow operational relocation of the split summary without changing science.
    parser.add_argument("--allow-unqualified", action="store_true", help="Explicitly permit user-authorized operational Reference B publication while retaining qualification=false and original 0.5% results.")  # Expose the nonblocking amendment only through an unmistakable opt-in.
    parser.add_argument("--expedited-levels", type=int, choices=range(2, 7), metavar="{2..6}", help="Use the first 2..6 registered levels; valid only together with --allow-unqualified.")  # Permit faster fixed-prefix execution without inventing new refinement scales.
    parser.add_argument("--dry-run", action="store_true", help="Validate the complete ordered plan and test freeze gate without creating a runner, mesh, solve, or file.")  # Expose the required solve-free rehearsal mode.
    return parser  # Return the complete parser with no hidden subsetting path.


def main(argv: list[str] | None = None) -> int:  # Execute one solve-free plan or resumable complete-split reference campaign.
    args = _parser().parse_args(argv)  # Parse only explicit split and operational paths.
    try:  # Convert expected protocol and path failures into a concise nonzero command result.
        if args.expedited_levels is not None and not args.allow_unqualified:  # Reject a shortened schedule before building even a solve-free plan.
            raise ReferenceCampaignError("--expedited-levels requires --allow-unqualified")  # Keep strict default execution independent from the post-registration amendment.
        result = run_reference_campaign(args.root, args.repository, args.split, dry_run=bool(args.dry_run), work_root=args.work_root, summary_directory=args.summary_dir, allow_unqualified=bool(args.allow_unqualified), expedited_levels=args.expedited_levels)  # Delegate ordering, disclosed amendment, freeze gating, cache reuse, verification, and aggregation.
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))  # Emit the complete strict-JSON plan or persisted terminal receipt.
        return 0 if result.get("status") in {"planned", "complete"} else 2  # Signal any retained per-case failure while preserving its written evidence.
    except (ReferenceCampaignError, FreezeError, ValueError, OSError) as exc:  # Catch expected manifest, resume, filesystem, and freeze-boundary failures.
        print(f"reference campaign error: {exc}", file=sys.stderr)  # Report an actionable concise failure without hiding its category.
        return 2  # Stop automation before a partial unauthorized campaign can be mistaken for success.


if __name__ == "__main__":  # Execute only when launched as a script rather than imported by focused tests.
    raise SystemExit(main())  # Propagate the explicit campaign status to the shell or workflow.

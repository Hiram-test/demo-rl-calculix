#!/usr/bin/env python3  # Execute the formal post-primary mechanism campaign with the active Python interpreter.
"""Run or dry-run the frozen WMVLA-4WAY-P1 B60000/K6 ablation campaign."""  # Describe the command's complete one-shot responsibility.
from __future__ import annotations  # Postpone annotation evaluation for repository runtime compatibility.

import argparse  # Parse the intentionally narrow formal command-line contract.
import json  # Emit finite machine-readable terminal evidence.
from pathlib import Path  # Resolve repository, campaign, manifest, and frozen-config paths portably.
import sys  # Import the local checkout and return explicit process status codes.

ROOT = Path(__file__).resolve().parents[1]  # Locate the repository independently from the invocation directory.
sys.path.insert(0, str(ROOT))  # Import this exact checked-out implementation without installation drift.

from visionamr.vla.four_way_ablations import AblationCampaignRequest, run_ablation_campaign  # Import the sole validated formal diagnostic boundary.


def _parser() -> argparse.ArgumentParser:  # Build the complete no-shard, no-resume command-line contract.
    parser = argparse.ArgumentParser(description="Run the one-shot post-primary WMVLA-4WAY-P1 mechanism ablations or perform a solve-free readiness check.")  # Create a self-describing parser.
    parser.add_argument("--root", type=Path, default=ROOT / "results" / "wm_vla_four_way_p1", help="Campaign root containing the validated freeze and complete primary test evidence.")  # Permit an isolated campaign location without changing scientific coordinates.
    parser.add_argument("--manifest", type=Path, default=None, help="Canonical case_manifest.json; defaults to <root>/protocol/case_manifest.json.")  # Permit explicit canonical-path spelling only.
    parser.add_argument("--frozen-config", type=Path, default=None, help="Canonical frozen_config.json; defaults to <root>/protocol/frozen_config.json.")  # Permit explicit canonical-path spelling only.
    parser.add_argument("--allow-unqualified-references", action="store_true", help="Acknowledge an operational non-qualified Reference B only when the same policy is sealed in frozen_config and TEST_STARTED.")  # Require explicit parity with any primary reference execution amendment.
    parser.add_argument("--dry-run", action="store_true", help="Revalidate freeze, primary completion, references, partitions, models, and exact phase order without writing or solving.")  # Provide the mandatory safe post-primary readiness mode.
    return parser  # Return the intentionally fixed B60000/K6 all-sixteen-case parser.


def main() -> int:  # Parse, validate, optionally execute, and report one formal invocation.
    args = _parser().parse_args()  # Parse caller-supplied paths and the sole solve-free control.
    campaign_root = args.root.resolve()  # Normalize the campaign boundary for unambiguous evidence paths.
    manifest_path = (args.manifest or campaign_root / "protocol" / "case_manifest.json").resolve()  # Resolve the sole canonical manifest input.
    frozen_config_path = (args.frozen_config or campaign_root / "protocol" / "frozen_config.json").resolve()  # Resolve the sole canonical frozen runtime input.
    request = AblationCampaignRequest(root=campaign_root, manifest_path=manifest_path, frozen_config_path=frozen_config_path, dry_run=bool(args.dry_run), allow_unqualified_references=bool(args.allow_unqualified_references))  # Assemble the immutable fixed-resource and reference-policy request.
    try:  # Convert readiness and fatal campaign failures into bounded terminal evidence while durable invalidation remains on disk.
        result = run_ablation_campaign(request)  # Perform complete preflight and optional 176-phase formal execution.
    except Exception as exception:  # Report configuration, integrity, native-boundary, API, and programming failures without scoring them as variants.
        payload = {"protocol_id": "WMVLA-4WAY-P1", "completed": False, "error_type": type(exception).__name__, "error": str(exception).replace("\x00", " ")[:2000]}  # Build a finite bounded failure result.
        print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False), file=sys.stderr)  # Emit concise machine-readable failure evidence.
        return 2  # Signal a refused or invalid formal campaign distinctly.
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))  # Emit the complete dry-run plan or durable execution index.
    return 0 if int(result.get("native_or_scored_failure_count", 0)) == 0 else 3  # Distinguish all-success evidence from a complete campaign retaining failed points.


if __name__ == "__main__":  # Execute only when launched as a command.
    raise SystemExit(main())  # Propagate the explicit process status to CI or the shell.

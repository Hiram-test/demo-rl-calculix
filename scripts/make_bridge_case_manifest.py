#!/usr/bin/env python3  # Execute the frozen manifest generator with the active Python interpreter.
"""Write the preregistered 48-case bridge manifest and SHA-256 sidecar."""  # Describe the command's complete responsibility.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.
import argparse  # Import the command-line parser for explicit output selection.
import json  # Import structured terminal reporting for automation and audit.
from pathlib import Path  # Import portable repository and output path handling.
import sys  # Import local-package path injection and process exit support.
ROOT = Path(__file__).resolve().parents[1]  # Locate the repository root independently of the launch directory.
sys.path.insert(0, str(ROOT))  # Make the checked-out visionamr package importable without installation.
from visionamr.bridge_case_manifest import write_case_manifest  # Import the sole validated manifest persistence entry point.

def _parser() -> argparse.ArgumentParser:  # Build the minimal frozen manifest command-line contract.
    parser = argparse.ArgumentParser(description="Write the frozen WMVLA-4WAY-P1 bridge case manifest.")  # Create a self-describing command-line parser.
    parser.add_argument("--output-dir", "--output", dest="output_dir", type=Path, default=ROOT / "results" / "wm_vla_four_way_p1" / "protocol", help="Directory receiving case_manifest.json and case_manifest.sha256.")  # Permit only artifact-location changes, never seed or sampling changes.
    return parser  # Return the complete parser without exposing scientific degrees of freedom.

def main() -> int:  # Generate, geometry-check, hash, persist, and report the frozen manifest artifacts.
    args = _parser().parse_args()  # Parse the output directory selected by the caller.
    manifest_path, checksum_path, digest = write_case_manifest(args.output_dir)  # Build and persist the unique validated 48-case design.
    summary = {"manifest": str(manifest_path), "checksum": str(checksum_path), "sha256": digest, "case_count": 48, "split_counts": {"train": 24, "validation": 8, "test": 16}}  # Assemble a concise machine-readable completion record.
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))  # Report exact artifact paths, digest, and frozen cardinalities.
    return 0  # Signal successful manifest generation.

if __name__ == "__main__":  # Execute only when the file is launched as a command.
    raise SystemExit(main())  # Propagate the explicit process status to CI or the shell.

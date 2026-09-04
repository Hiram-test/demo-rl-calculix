#!/usr/bin/env python3  # Package completed visual-decision evidence without reading its scientific outcomes.
"""Build a compact reproducible ZIP and SHA-256 manifests from frozen experiment artifacts."""  # Describe the purely mechanical packaging operation.
from __future__ import annotations  # Support current type annotations consistently.
import argparse  # Expose the output location and optional small upload-part size.
from collections import Counter  # Count artifact types and retained native evidence mechanically.
import hashlib  # Fingerprint exact source bytes and final upload artifacts.
import io  # Compact NPZ containers mechanically without loading or interpreting numerical arrays.
import json  # Write packaging metadata without interpreting experiment JSON.
from pathlib import Path  # Resolve the specified experiment and runtime artifact trees.
import subprocess  # Record the local Git source identity without changing the repository.
import zipfile  # Compress portable evidence while excluding large native duplicates.
ROOT = Path(__file__).resolve().parents[1]  # Resolve this script's current repository root.
RUN_DIRECTORIES = ("visual_wm_probe", "gpt_direct_901", "gpt_direct_902", "gpt_direct_903", "gpt_reference_check", "gpt_ranker_training", "gpt_holdout", "gpt_direct_1001", "gpt_direct_1002")  # Cover every requested completed or forthcoming experiment tree by fixed name.
EVIDENCE_SUFFIXES = frozenset((".json", ".npz", ".png", ".inp", ".log"))  # Retain metadata, original fields, actual viewed images, and native inputs and logs.
SOURCE_SCRIPTS = ("run_visual_world_experiment.py", "run_gpt_visual_action.py", "train_gpt_action_ranker.py", "check_gpt_reference_resolution.py", "render_gpt_quick_views.py", "package_gpt_visual_evidence.py")  # Preserve the scripts required to reproduce the packaged experiment and its packaging.
RUNTIME_FILES = ("runtime_manifest.json", "verify_native.py", "verification/native_verification.json", "verification/native_tension_patch.inp", "verification/native_tension_patch.log")  # Keep installation provenance and the authentic analytical verification without runtime binaries.
def digest(data):  # Compute a standard content fingerprint for bytes already read mechanically.
    return hashlib.sha256(data).hexdigest()  # Return the hexadecimal SHA-256 identity.
def write_json(path, value):  # Serialize packaging metadata with stable formatting.
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")  # Preserve deterministic readable manifests.
def git_text(*arguments):  # Read only local source-control metadata for reproducibility.
    process = subprocess.run(["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=False)  # Avoid network operations and repository mutations.
    return process.stdout.strip() if process.returncode == 0 else None  # Preserve an explicitly unavailable Git identity without preventing packaging.
def archive_entry(archive, name, data):  # Store one exact byte snapshot with reproducible ZIP metadata.
    info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))  # Use a fixed supported timestamp so identical content yields identical archives.
    info.create_system = 3  # Record a portable Unix-style metadata convention.
    info.external_attr = 0o100644 << 16  # Preserve data-file readability without inheriting host permissions.
    info.compress_type = zipfile.ZIP_DEFLATED  # Compress JSON, solver decks, and logs aggressively while retaining NPZ intact.
    archive.writestr(info, data, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)  # Write the exact bytes used for the manifest fingerprint.
def package_payload(name, source):  # Remove only reconstructible duplicate arrays from known solution containers.
    filename = Path(name).name  # Identify the container by its recorded artifact name rather than its numerical contents.
    is_field = filename.startswith("post_") and filename.endswith(".npz") or filename.startswith("reference") and filename.endswith(".npz")  # Compact only final fields while retaining initial and observation bytes bound by the sealed GPT provenance.
    if not is_field:  # Keep targets, observations, models, and all other evidence byte-identical.
        return source, "verbatim"  # Preserve the complete original artifact unchanged.
    with zipfile.ZipFile(io.BytesIO(source), "r") as original:  # Read only the NPZ container and opaque NPY members without evaluating array values.
        names = set(original.namelist())  # Inspect member names to recognize the existing nodes, cells, and displacement contract.
        required = ("nodes.npy", "cells.npy", "u.npy")  # Preserve the exact native geometry, connectivity, and saved displacement arrays.
        if not set(required).issubset(names):  # Leave unusual reference containers unchanged instead of guessing their reconstruction schema.
            return source, "verbatim"  # Preserve any unsupported container without dropping data.
        buffer = io.BytesIO()  # Hold the reduced portable NPZ container in memory.
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as reduced:  # Retain a standard NPZ-readable ZIP of original NPY byte streams.
            for member in required:  # Copy exactly the three required reconstruction arrays in a deterministic order.
                archive_entry(reduced, member, original.read(member))  # Preserve every original NPY byte without decoding numeric values.
        return buffer.getvalue(), "field_npz_nodes_cells_u_only"  # Mark the deliberate derived artifact so its hash is never confused with the original NPZ.
def collect_sources(runtime):  # Select evidence by fixed path and filename rules without interpreting results.
    selected, missing, omitted = {}, [], Counter()  # Track included artifacts, absent requested roots, and exclusion counts.
    for name in RUN_DIRECTORIES:  # Mechanically enumerate every requested experiment directory, including sealed holdout cases.
        directory = ROOT / "runs" / name  # Resolve the fixed run root without reading its metrics.
        if not directory.is_dir():  # Record experiments that have not yet produced an artifact directory.
            missing.append(str(directory.relative_to(ROOT)))  # Report absence in the package manifest without inventing content.
            continue  # Continue preparing all currently available evidence.
        for path in sorted(directory.rglob("*")):  # Enumerate files in a stable lexical order for reproducibility.
            if not path.is_file() or path.is_symlink():  # Keep only real files within the explicitly selected artifact trees.
                continue  # Exclude directories and external symlink targets mechanically.
            if "partial" in path.name.lower():  # Remove redundant checkpoint JSON and incomplete files by their existing names.
                omitted["partial_checkpoints"] += 1  # Count this explicit space-saving exclusion.
                continue  # Preserve the complete final summaries and decisions instead.
            if path.suffix.lower() not in EVIDENCE_SUFFIXES:  # Omit FRD, native scratch duplicates, runtime binaries, and unspecified formats.
                omitted[path.suffix.lower() or "no_suffix"] += 1  # Make every excluded file category visible in the receipt.
                continue  # Keep saved displacement NPZ as the authoritative field-reconstruction artifact.
            selected[str(path.relative_to(ROOT))] = (path, f"runs/{name}")  # Preserve the original repository-relative artifact identity.
    for name in SOURCE_SCRIPTS:  # Include the concrete execution scripts without unrelated experiments.
        path = ROOT / "scripts" / name  # Resolve each explicitly listed reproduction script.
        if path.is_file():  # Include scripts that exist at the final packaging snapshot.
            selected[str(path.relative_to(ROOT))] = (path, "source")  # Preserve source code under its original repository-relative path.
        else:  # Record an absent optional source script honestly.
            missing.append(f"scripts/{name}")  # Expose missing source provenance without modifying any experiment.
    for path in sorted((ROOT / "visionamr").rglob("*.py")):  # Preserve the numerical and visual modules used by the current scripts.
        if path.is_file() and not path.is_symlink():  # Exclude non-file paths and external links from the source snapshot.
            selected[str(path.relative_to(ROOT))] = (path, "source")  # Include plain source only, excluding compiled caches and model binaries.
    for name in ("requirements.txt", "pyproject.toml", "README.md"):  # Retain available dependency and repository documentation.
        path = ROOT / name  # Resolve each small reproduction aid.
        if path.is_file():  # Include only source files that actually exist.
            selected[name] = (path, "source")  # Keep the documentation at its original relative path.
    results_path = ROOT / "runs/gpt_visual_results.json"  # Include the parent-produced consolidated result receipt as opaque bytes when available.
    if results_path.is_file():  # Preserve the final result summary without reading its numerical conclusions.
        selected[str(results_path.relative_to(ROOT))] = (results_path, "consolidated_reports")  # Keep the original summary path and full byte identity.
    for path in sorted((ROOT / "docs").rglob("*")):  # Discover completed written reports and their figures mechanically at packaging time.
        if path.is_file() and not path.is_symlink() and path.suffix.lower() in (".md", ".pdf", ".png"):  # Include available report formats while excluding unrelated binary artifacts.
            selected[str(path.relative_to(ROOT))] = (path, "consolidated_reports")  # Preserve both reports automatically when the parent finishes writing them.
    for name in RUNTIME_FILES:  # Include the verified runtime's receipts and small analytical test inputs.
        path = runtime / name  # Resolve the independently installed runtime artifact.
        if path.is_file():  # Preserve existing native verification evidence without creating a substitute.
            selected[f"runtime_receipts/{name}"] = (path, "runtime_receipts")  # Map the runtime receipt tree into a portable package namespace.
        else:  # Record missing native-runtime provenance rather than silently claiming completeness.
            missing.append(f"runtime_receipts/{name}")  # Preserve the concrete absent receipt path.
    return sorted(selected.items()), missing, dict(sorted(omitted.items()))  # Return deterministic file selection without reading any result values.
def main():  # Build a compact evidence package after the parent finishes the held-out experiments.
    parser = argparse.ArgumentParser(description=__doc__)  # Expose only mechanical packaging choices.
    parser.add_argument("--output", type=Path, default=ROOT.parent / "deliverables/gpt_visual_evidence")  # Choose a separate local delivery directory without uploading anything.
    parser.add_argument("--runtime", type=Path, default=ROOT.parent / "ccx_runtime")  # Locate the previously verified native installation receipts.
    parser.add_argument("--max-part-mib", type=float, default=19.0)  # Keep any upload part below the requested twenty-megabyte target.
    args = parser.parse_args()  # Parse artifact locations and the upload-size preference.
    output = args.output.resolve()  # Normalize the package output path once.
    output.mkdir(parents=True, exist_ok=True)  # Prepare the local delivery folder without any external write.
    selected, missing, omitted = collect_sources(args.runtime.resolve())  # Enumerate the requested numerical evidence by filename rules alone.
    archive_path = output / "gpt_visual_evidence.zip"  # Name the complete portable ZIP artifact.
    manifest_path = output / "EVIDENCE_MANIFEST.json"  # Name the readable source-content and coverage receipt.
    content_sums_path = output / "CONTENT_SHA256SUMS.txt"  # Name the exact per-entry content fingerprint list.
    rows, coverage, content_sums = [], {}, []  # Accumulate deterministic source metadata and mechanical native-file counts.
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9, allowZip64=True) as archive:  # Build one portable archive before deciding whether transport parts are necessary.
        for name, (path, group) in selected:  # Read selected files only as opaque bytes for copying, hashing, and compression.
            source = path.read_bytes()  # Snapshot the exact file without interpreting scientific results, including sealed holdout outcomes.
            data, transformation = package_payload(name, source)  # Mechanically remove only reproducible stress and estimator duplicates from saved solution NPZ containers.
            fingerprint = digest(data)  # Bind the manifest entry to the bytes actually included in the ZIP.
            archive_entry(archive, name, data)  # Compress the exact source snapshot under its portable relative path.
            rows.append({"path": name, "bytes": len(data), "sha256": fingerprint, "archived_sha256": fingerprint, "source_bytes": len(source), "source_sha256": digest(source), "original_sha256": digest(source), "packaging": transformation, "group": group})  # Record explicitly different original and archived hash identities for reduced final-field containers.
            content_sums.append(f"{fingerprint}  {name}\n")  # Write a standard portable content-checksum entry.
            counts = coverage.setdefault(group, {"files": 0, "bytes": 0, "suffix_counts": {}, "native_inp_count": 0, "native_log_count": 0})  # Allocate purely mechanical coverage counts for the requested group.
            counts["files"] += 1  # Count the copied source artifacts.
            counts["bytes"] += len(data)  # Accumulate uncompressed source volume without interpreting content.
            suffix = Path(name).suffix.lower()  # Classify the artifact using its actual stored filename.
            counts["suffix_counts"][suffix] = counts["suffix_counts"].get(suffix, 0) + 1  # Retain JSON, NPZ, image, source, and native evidence counts.
            counts["native_inp_count"] += int(suffix == ".inp")  # Count retained native solver decks without inferring successful solves from them.
            counts["native_log_count"] += int(suffix == ".log")  # Count retained native logs while leaving solver-success semantics in their original receipts.
        manifest = {"format": "GPT visual AMR reproducible evidence package v1", "repository_head": git_text("rev-parse", "HEAD"), "repository_source_status": git_text("status", "--short", "--", "scripts", "visionamr", "requirements.txt", "pyproject.toml"), "requested_run_directories": [f"runs/{name}" for name in RUN_DIRECTORIES], "missing_requested_inputs": missing, "included_file_count": len(rows), "uncompressed_source_bytes": sum(row["source_bytes"] for row in rows), "uncompressed_packaged_payload_bytes": sum(row["bytes"] for row in rows), "coverage": coverage, "omitted_file_counts": omitted, "selection": {"evidence_suffixes": sorted(EVIDENCE_SUFFIXES), "excluded": ["partial checkpoints", "FRD files", "native temporary duplicates", "runtime binaries", "compiled Python caches", "reconstructible duplicate arrays in final solution NPZ"], "holdout_handling": "opaque byte copying only; NPZ member-name selection without loading numerical array values; no scientific-result interpretation", "sealed_initial_provenance": "initial.npz and observation.npz are archived byte-identically, preserving the initial SHA-256 identities in the sealed GPT records", "field_reconstruction": "post and reference NPZ retain the exact original nodes.npy, cells.npy, and u.npy byte streams; compute_post reconstructs stress and energy, zz_indicator reconstructs eta2", "compact_npz_hash_semantics": "original_sha256 identifies the unchanged original full-field NPZ; archived_sha256 identifies the compact archived bytes; these hashes intentionally differ for compacted NPZ", "compact_npz_verification_limit": "The original reference.npz or post NPZ byte hash cannot be independently verified using this compact package alone. The package preserves the exact original displacement and mesh array bytes and permits field reconstruction and repeated post-processing; original receipts still refer to the full original NPZ hash.", "native_count_semantics": "file counts only; successful native solves remain evidenced by original solver logs and experiment receipts"}, "files": rows}  # Describe the preserved sealed inputs and explicit compact-reference hash-verification limitation.
        manifest_bytes = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")  # Render stable internal and external manifests from the same source receipt.
        content_bytes = "".join(content_sums).encode("utf-8")  # Render the standard checksum list for all packaged source artifacts.
        archive_entry(archive, "EVIDENCE_MANIFEST.json", manifest_bytes)  # Embed the coverage and source-byte manifest in the ZIP.
        archive_entry(archive, "CONTENT_SHA256SUMS.txt", content_bytes)  # Embed per-file verification instructions in standard checksum form.
    manifest_path.write_bytes(manifest_bytes)  # Save the same readable manifest beside the ZIP for review before extraction.
    content_sums_path.write_bytes(content_bytes)  # Save source checksums independently of the archive itself.
    archive_size = archive_path.stat().st_size  # Measure the actual compressed artifact size.
    part_limit = max(1024 * 1024, int(args.max_part_mib * 1024 * 1024))  # Resolve the requested transport-part size with a practical one-megabyte minimum.
    for stale in output.glob("gpt_visual_evidence.zip.part[0-9][0-9][0-9]"):  # Remove only obsolete split parts produced by an earlier run of this same packager.
        stale.unlink()  # Avoid including outdated trailing parts when the regenerated archive needs fewer parts.
    upload_paths = [archive_path]  # Prefer one normal ZIP whenever it meets the requested size preference.
    if archive_size > part_limit:  # Split transport only when the actual compressed ZIP exceeds the upload target.
        upload_paths = []  # Upload raw ZIP parts instead of duplicating the full large archive.
        with archive_path.open("rb") as source:  # Stream the completed archive without reading its compressed scientific contents.
            index = 1  # Number transport parts deterministically from one.
            while True:  # Copy each bounded byte range until the complete ZIP has been preserved.
                block = source.read(part_limit)  # Keep each API-upload artifact within the chosen size preference.
                if not block:  # Finish after the exact final ZIP byte.
                    break  # Avoid creating an empty trailing part.
                part = output / f"gpt_visual_evidence.zip.part{index:03d}"  # Preserve lexical part order for a simple deterministic reconstruction.
                part.write_bytes(block)  # Store one contiguous range of the unchanged original ZIP.
                upload_paths.append(part)  # Record the exact pieces the parent should upload.
                index += 1  # Advance to the next transport part.
    archive_hash = digest(archive_path.read_bytes())  # Fingerprint the full ZIP so reassembled transport parts can be verified exactly.
    reconstruction_path = output / "ARCHIVE_RECONSTRUCTION.md"  # Save exact concatenation, checksum, and extraction instructions beside the transport pieces.
    ordered_parts = [path.name for path in upload_paths] if len(upload_paths) > 1 else []  # Preserve the exact lexical transport order without treating a single ordinary ZIP as a part.
    reconstruction = ["# GPT visual AMR evidence reconstruction", "", f"Full archive SHA-256: `{archive_hash}`", "", f"Full archive bytes: {archive_size}", "", "Transport part order:", "", *[f"{index}. `{name}`" for index, name in enumerate(ordered_parts, start=1)], "", "Run in the directory containing the downloaded parts and checksum receipts:", "", "```bash", "sha256sum --ignore-missing -c SHA256SUMS.txt # Verify the downloaded transport pieces and receipts."]  # Generate mechanical reconstruction guidance without including any scientific outcomes.
    if ordered_parts:  # Provide concatenation only when the complete ZIP was split for transport.
        reconstruction.append("cat gpt_visual_evidence.zip.part[0-9][0-9][0-9] > gpt_visual_evidence.zip # Reconstruct the exact complete archive in numeric part order.")  # Reassemble the original bytes with an explicitly bounded lexical filename pattern.
    reconstruction.extend(["sha256sum -c SHA256SUMS.txt # Verify all downloaded pieces and the complete reconstructed ZIP.", "unzip -t gpt_visual_evidence.zip # Check the ZIP's internal CRC integrity.", "unzip gpt_visual_evidence.zip -d gpt_visual_evidence # Extract the preserved relative source and evidence paths.", "(cd gpt_visual_evidence && sha256sum -c CONTENT_SHA256SUMS.txt) # Verify every archived scientific and source artifact.", "```", "", "Final-field post/reference NPZ containers preserve exact original nodes, cells, and displacement NPY bytes; their compact archive hashes differ from their recorded original complete NPZ hashes. Initial and observation NPZ bytes remain unchanged. See EVIDENCE_MANIFEST.json for the explicit verification limit.", ""])  # State both the exact extraction sequence and the intentional compact-field hash distinction.
    reconstruction_path.write_text("\n".join(reconstruction), encoding="utf-8")  # Save a standalone reconstruction document before its transport checksum is calculated.
    artifact_paths = [*upload_paths, manifest_path, content_sums_path, reconstruction_path]  # Include upload pieces, provenance receipts, and exact reconstruction instructions.
    artifact_rows = [{"file": path.name, "bytes": path.stat().st_size, "sha256": digest(path.read_bytes())} for path in artifact_paths]  # Record exact upload sizes and checksums without interpreting archive entries.
    checksums_path = output / "SHA256SUMS.txt"  # Name the final transport-artifact checksum list.
    checksum_lines = [f"{row['sha256']}  {row['file']}\n" for row in artifact_rows]  # List every actual transport artifact once in standard checksum format.
    if ordered_parts:  # Include the complete ZIP hash even when the parent uploads only its small byte ranges.
        checksum_lines.append(f"{archive_hash}  {archive_path.name}\n")  # Permit independent verification immediately after exact concatenation.
    checksums_path.write_text("".join(checksum_lines), encoding="utf-8")  # Save standard checksums for both each part and the complete archive.
    summary = {"archive": str(archive_path), "archive_bytes": archive_size, "archive_mib": archive_size / 1024 ** 2, "archive_sha256": archive_hash, "split_for_upload": len(upload_paths) > 1, "part_bytes_limit": part_limit, "parts_in_order": ordered_parts, "upload_artifacts": artifact_rows, "transport_checksums": str(checksums_path), "reassembly_command": "cat gpt_visual_evidence.zip.part[0-9][0-9][0-9] > gpt_visual_evidence.zip" if len(upload_paths) > 1 else None, "unzip_command": "unzip gpt_visual_evidence.zip -d gpt_visual_evidence", "reconstruction_instructions": str(reconstruction_path), "included_file_count": len(rows), "coverage": coverage, "missing_requested_inputs": missing, "uploaded": False}  # Record exact part order, whole-archive identity, and extraction instructions without implying publication.
    write_json(output / "PACKAGE_SUMMARY.json", summary)  # Save the complete local packaging receipt for the parent.
    print(json.dumps({"archive": str(archive_path), "archive_bytes": archive_size, "archive_mib": archive_size / 1024 ** 2, "upload_files": [{"file": row["file"], "bytes": row["bytes"]} for row in artifact_rows], "content_files": len(rows), "missing_requested_inputs": missing}, ensure_ascii=False, indent=2), flush=True)  # Report only coverage and file sizes without revealing held-out numerical outcomes.
if __name__ == "__main__":  # Execute packaging only after an explicit parent invocation.
    main()  # Build the frozen evidence package locally without uploading anything.

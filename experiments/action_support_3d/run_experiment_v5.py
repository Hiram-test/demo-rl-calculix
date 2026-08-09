from __future__ import annotations  # Enable modern type annotations without runtime evaluation.
import itertools  # Enumerate every unordered pair in the fixed sixteen-atom spatial dictionary.
import sys  # Preserve explicit command-line failure reporting for GitHub Actions.
import run_experiment as core  # Reuse the validated geometry, frozen LLM proposal loader, parsers, and original result schema.
import run_experiment_v4 as v4  # Reuse common-DOF calibration and the final mesh-invariant physical-QoI structural evaluator.

ORIGINAL_BUILD_CASES = core.build_cases  # Preserve the original case builder for provenance even though v0.5 expands its nonsemantic candidate dictionary.


def build_cases_v5() -> list[dict]:  # Build global baselines, all single atoms, all 120 unordered atom pairs, and the same six frozen LLM supports.
    document = core.load_llm_proposals()  # Read the committed frozen semantic proposals without modifying confidence, geometry, or ordering.
    atoms = core.evaluation_atoms()  # Recreate the sixteen fixed 4-by-2-by-2 spatial atoms used by all prior benchmark revisions.
    llm_supports = core.flatten_llm_proposals(document)  # Flatten the same three proposals per original QoI into executable local supports.
    cases: list[dict] = [  # Start with unchanged global baseline and numerical-reference meshes.
        {"id": "coarse_global", "kind": "global", "mesh_size": core.COARSE_H, "source": "baseline"},  # Preserve the deliberately coarse initial mesh.
        {"id": "reference_global", "kind": "global", "mesh_size": core.REFERENCE_H, "source": "reference"},  # Preserve the globally fine numerical reference mesh.
    ]  # Finish the unchanged global case list.
    for atom in atoms:  # Add every fixed single-patch support exactly as in v0.4.
        atom_case = dict(atom)  # Copy the atom metadata so the canonical atom dictionary is not mutated in place.
        atom_case["kind"] = "local"  # Mark the single atom as one locally refined action support.
        atom_case["source"] = "atom_single"  # Use an explicit source label that distinguishes fixed single patches from exhaustive pair combinations.
        cases.append(atom_case)  # Append the current single spatial patch to the common-budget candidate matrix.
    for first, second in itertools.combinations(atoms, 2):  # Enumerate all sixteen choose two unordered spatial-patch combinations exactly once.
        cases.append({  # Create one joint action support containing both selected fixed patches.
            "id": f"P_{first['id']}_{second['id']}",  # Encode the two constituent atom identifiers in a stable searchable pair identifier.
            "kind": "local",  # Mark the pair as one joint local refinement action for direct support-object comparison.
            "source": "atom_pair_exhaustive",  # Mark this support as part of the exhaustive nonsemantic two-patch oracle dictionary.
            "qoi_target": "both",  # Evaluate every spatial pair against both fixed physical QoIs without task-conditioned pruning.
            "confidence": None,  # Leave probability undefined because exhaustive spatial pairs are not model predictions.
            "regions": first["regions"] + second["regions"],  # Combine the two fixed boxes into one potentially disconnected joint refinement support.
        })  # Finish the current unordered atom-pair case.
    for support in llm_supports:  # Add the six semantic supports only after the exhaustive nonsemantic dictionary is complete.
        support_case = dict(support)  # Copy semantic metadata so the committed frozen proposal document remains immutable.
        support_case["kind"] = "local"  # Mark each frozen semantic proposal as one locally refined joint action support.
        cases.append(support_case)  # Append the current LLM support to the same common-budget structural experiment.
    return cases  # Return two global cases plus sixteen singles, 120 pairs, and six frozen LLM candidates.


def write_pair_analysis(rows: list[dict]) -> None:  # Summarize where six semantic candidates fall relative to the exhaustive fixed two-patch dictionary.
    pair_rows = [row for row in rows if row.get("source") == "atom_pair_exhaustive"]  # Isolate the complete nonsemantic two-patch candidate universe.
    single_rows = [row for row in rows if row.get("source") == "atom_single"]  # Isolate the fixed one-patch dictionary for reference.
    llm_rows = [row for row in rows if row.get("source") == "llm_semantic"]  # Isolate the six frozen semantic candidates for compression analysis.
    lines: list[str] = []  # Accumulate a compact Markdown interpretation table directly beside the numerical artifacts.
    lines.append("# Exhaustive fixed-pair comparison")  # Add the analysis title.
    lines.append("")  # Add one Markdown spacer line.
    lines.append(f"All {len(pair_rows)} unordered two-atom supports and all {len(single_rows)} single atoms were resource-normalized with the same target of {v4.TARGET_DOF} DOF before solving.")  # State the exact nonsemantic candidate universe and resource constraint.
    lines.append("")  # Add one Markdown spacer line.
    for title, error_key in [("Global interior vertical displacement", "tip_rel_error"), ("Local axial difference across the hole ligaments", "hole_rel_error")]:  # Analyze the two final fixed physical QoIs independently.
        best_pair = min(pair_rows, key=lambda row: row[error_key])  # Find the exhaustive two-patch spatial oracle for the current QoI.
        best_single = min(single_rows, key=lambda row: row[error_key])  # Find the best fixed single patch for the current QoI.
        best_llm = min(llm_rows, key=lambda row: row[error_key])  # Find the best of the six frozen semantic proposals for the current QoI.
        combined = sorted(pair_rows + llm_rows, key=lambda row: row[error_key])  # Rank semantic candidates inside the exhaustive pair-plus-semantic comparison pool.
        best_llm_rank = next(index for index, row in enumerate(combined, start=1) if row["id"] == best_llm["id"])  # Locate the best semantic candidate rank without assuming it beats the pair oracle.
        pair_near_count = sum(row[error_key] <= 1.10 * best_pair[error_key] for row in pair_rows)  # Count how many exhaustive pairs lie within ten percent of the best pair error.
        lines.append(f"## {title}")  # Add the current QoI section title.
        lines.append("")  # Add one Markdown spacer line.
        lines.append(f"Best single atom: `{best_single['id']}`, relative error {best_single[error_key]:.6e}, DOF {best_single['dof_proxy']}.")  # Report the strongest one-patch nonsemantic baseline.
        lines.append("")  # Add one Markdown spacer line.
        lines.append(f"Best exhaustive atom pair: `{best_pair['id']}`, relative error {best_pair[error_key]:.6e}, DOF {best_pair['dof_proxy']}; {pair_near_count} of 120 pairs lie within 10% of this pair optimum.")  # Report the exhaustive two-patch oracle and near-optimal pair multiplicity.
        lines.append("")  # Add one Markdown spacer line.
        lines.append(f"Best frozen LLM support: `{best_llm['id']}`, relative error {best_llm[error_key]:.6e}, DOF {best_llm['dof_proxy']}; rank {best_llm_rank} among 120 exhaustive pairs plus six semantic candidates.")  # Report semantic quality as a search-space compression result rather than an unconditional optimum claim.
        lines.append("")  # Add one Markdown spacer line.
        lines.append("| rank | candidate | source | relative error | DOF |")  # Add the compact top-candidate comparison table header.
        lines.append("| ---: | --- | --- | ---: | ---: |")  # Add the Markdown table separator.
        for rank, row in enumerate(combined[:10], start=1):  # Show the ten best pair-or-semantic candidates for the current QoI.
            lines.append(f"| {rank} | {row['id']} | {row['source']} | {row[error_key]:.6e} | {row['dof_proxy']} |")  # Add one ranked candidate row.
        lines.append("")  # Add one Markdown spacer line before the next QoI section.
    (core.RESULTS_DIR / "pair_oracle_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")  # Persist the exhaustive-pair compression analysis beside the standard result artifacts.


def write_outputs_v5(rows: list[dict], reference: dict) -> None:  # Preserve v0.4 outputs and add exhaustive-pair ranking diagnostics.
    v4.write_outputs_v4(rows, reference)  # Write standard errors, Pareto flags, manifest, summary, and common-DOF calibration audit table.
    write_pair_analysis(rows)  # Add the semantic-versus-exhaustive-two-patch search-compression report after relative errors are available.


def main() -> int:  # Install the expanded fixed-patch candidate universe around the unchanged v0.4 common-budget evaluator.
    v4.CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the shared mesh-only calibration directory exists before any resource probes are launched.
    core.build_cases = build_cases_v5  # Expand the nonsemantic support dictionary to every single atom and every unordered atom pair.
    core.solve_case = v4.solve_case_v4  # Keep the same approximately 3000-DOF normalization and final fixed physical QoIs for every local candidate.
    core.write_outputs = write_outputs_v5  # Preserve all standard outputs and add exhaustive pair-oracle compression analysis.
    return core.main()  # Execute the complete 144-case structural experiment under one unchanged reference solution and support budget.


if __name__ == "__main__":  # Execute the exhaustive pair-control experiment only when this file is invoked as the program entry point.
    try:  # Preserve explicit workflow failure reporting across calibration, meshing, solving, and pair analysis.
        raise SystemExit(main())  # Run the final exhaustive fixed-pair control and terminate with its returned shell status.
    except Exception as exc:  # Catch unexpected errors only at the outermost command-line boundary.
        print(f"[fatal-v5] {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)  # Emit a concise failure reason to the GitHub Actions log and persisted console tail.
        raise  # Re-raise the original exception so the workflow accurately records a failed exhaustive-control run.

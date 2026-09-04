#!/usr/bin/env python3  # Render exact measured GPT visual-AMR evidence.
"""Scientific plots from sealed GPT actions and the common finite-reference comparison."""  # Describe the evidence scope.
from __future__ import annotations  # Keep annotations compatible with the runtime.
import argparse  # Expose reproducible source and output paths.
import json  # Read existing experiment records without changing them.
import sys  # Make the repository's real mesh class importable.
from pathlib import Path  # Resolve experiment paths portably.
import numpy as np  # Compute exact chart values and geometric projections.
import matplotlib  # Select a headless scientific plotting backend.
matplotlib.use("Agg")  # Render in the shared execution environment without a display server.
import matplotlib.pyplot as plt  # Draw reproducible standard scientific figures.
from matplotlib.collections import LineCollection  # Plot actual generated boundary-triangle edges.
from matplotlib.colors import LogNorm, Normalize  # State the raster and size-ratio color scales explicitly.
from matplotlib.patches import Rectangle  # Overlay only the original GPT region coordinates.
ROOT = Path(__file__).resolve().parents[1]  # Resolve the source checkout independently of the current directory.
sys.path.insert(0, str(ROOT))  # Reuse the repository's mesh topology implementation.
from visionamr.mesher import Mesh  # Load true source and generated finite-element meshes.
#
METHODS = (("best_dorfler", "Best of 3 Doerfler (ex post)", "#6B7280"), ("analytic_density", "Analytic density", "#C57A23"), ("gpt", "GPT direct image decision", "#087F8C"), ("gpt_plus_old_world_model", "GPT + spatial WM", "#7044A5"))  # Keep the same method labels and colors in every panel.
#
def read_json(path):  # Read recorded numerical evidence without interpretation or mutation.
    return json.loads(Path(path).read_text())  # Preserve the stored exact floating-point values.
#
def method_result(case, name):  # Resolve baselines and sealed selections through one access path.
    return case[name] if name in ("best_dorfler", "analytic_density") else case["selections"][name]  # Keep retrospective baselines distinct from selected GPT actions.
#
def save_figure(fig, output, stem):  # Save shareable and print-ready copies of each scientific figure.
    fig.savefig(output / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")  # Write the requested raster artifact.
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", facecolor="white")  # Preserve vector text and plot geometry for reports.
    plt.close(fig)  # Release plotting resources between figures.
#
def comparison(data, output):  # Plot all five physical cases against the same finite reference factor.
    cases = data["cases"]  # Keep diagnostic and prospective cases in their recorded order.
    x = np.arange(len(cases), dtype=float)  # Allocate equal chart positions without implying temporal spacing.
    labels = [str(case["seed"]) + ("\npilot" if case["phase"] == "diagnostic_pilot" else "\nprospective") for case in cases]  # Make the evaluation phase visible beside every case.
    fig, axes = plt.subplots(3, 1, figsize=(12.8, 11.5), sharex=True, gridspec_kw={"height_ratios": [1.05, 1.1, 1.0]})  # Separate accuracy, relative comparison, and real resources.
    for method_index, (name, label, color) in enumerate(METHODS):  # Draw measured outcomes for each fixed method.
        values = [method_result(case, name) for case in cases]  # Read the true selected action or baseline outcome.
        offset = (method_index - 1.5) * 0.18  # Keep method observations separated within each case.
        errors = np.asarray([value["reference_error"] for value in values])  # Use the finite-reference field discrepancy exactly as recorded.
        dofs = np.asarray([value["n_equations"] for value in values])  # Use actual free displacement equations rather than nominal budgets.
        axes[0].scatter(x + offset, errors, s=64, color=color, label=label, zorder=3)  # Plot absolute discrepancy with no misleading zero-based bar area.
        difference = 100.0 * (errors / np.asarray([case["best_dorfler"]["reference_error"] for case in cases]) - 1.0)  # Compute exact percent differences from the retrospective classical envelope.
        if name != "best_dorfler":  # Keep the zero comparator as a line rather than duplicate bars.
            positions = x + (method_index - 2) * 0.23  # Center the three actual relative comparisons in each case.
            axes[1].bar(positions, difference, width=0.21, color=color, alpha=0.92)  # Display signed measured differences without claiming uniform success.
            for position, value in zip(positions, difference):  # Annotate every relative comparison for precise reading.
                axes[1].text(position, value + (0.09 if value >= 0 else -0.09), f"{value:+.2f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=8, color=color)  # State the exact displayed percent sign and rounding.
        axes[2].scatter(x + offset, dofs, s=55, color=color, zorder=3)  # Show actual budget utilization independently of accuracy.
        for position, count in zip(x + offset, dofs):  # Annotate measured resource counts directly.
            axes[2].text(position, count + (5 if method_index % 2 else -5), str(count), ha="center", va="bottom" if method_index % 2 else "top", fontsize=8, color=color)  # Preserve integer equation counts without normalization.
    for index, case in enumerate(cases):  # Draw the shared actual-equation cap for each case.
        axes[2].plot([index - 0.42, index + 0.42], [case["cap"]] * 2, color="#20242B", ls="--", lw=1.1)  # Show the cap without implying exact equal equations.
        axes[2].text(index, case["cap"] + 16, f"cap {case['cap']}", ha="center", fontsize=8, color="#20242B")  # Label the per-case resource bound explicitly.
    for axis in axes:  # Apply consistent restrained scientific styling.
        axis.grid(axis="y", color="#E4E7EC", lw=0.8, zorder=0)  # Make numerical comparisons readable without decorative backgrounds.
        axis.spines[["top", "right"]].set_visible(False)  # Remove redundant frame strokes.
        axis.axvline(2.5, color="#8F97A3", ls=":", lw=1.2)  # Separate the three diagnostic cases from two later prospective cases.
        axis.set_xlim(-0.6, len(cases) - 0.4)  # Retain clear margins around the grouped observations.
    axes[0].set_ylabel("Relative reference-field\nenergy discrepancy")  # Name the measured finite-reference quantity accurately.
    axes[0].set_title("A  Accuracy against the same finite reference", loc="left", fontweight="bold")  # Identify the first panel's concrete comparison.
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.28), ncol=2, frameon=False, fontsize=9)  # Give method identities once without covering measured points.
    axes[1].axhline(0, color="#30343B", lw=1.0)  # Make the signed comparison reference explicit.
    axes[1].set_ylabel("Difference vs best Doerfler (%)\nnegative = lower discrepancy")  # State the favorable direction without overstating generalization.
    axes[1].set_ylim(-4.0, 3.8)  # Leave enough room for all exact percent annotations.
    axes[1].set_title("B  Relative outcome; all comparisons are case specific", loc="left", fontweight="bold")  # Avoid implying one universal method ranking.
    axes[2].set_ylabel("Actual free displacement\nequations")  # Identify the measured resource unit.
    axes[2].set_title("C  Resource use; dashed lines show the common cap", loc="left", fontweight="bold")  # Explain why identical caps need not imply identical mesh counts.
    axes[2].set_ylim(min(method_result(case, name)["n_equations"] for case in cases for name, _, _ in METHODS) - 45, max(case["cap"] for case in cases) + 42)  # Fit actual count labels with explicit axis ticks.
    axes[2].set_xticks(x, labels)  # Display seed and evaluation phase together.
    fig.suptitle("GPT image decisions in real 3D CalculiX AMR", x=0.07, y=1.005, ha="left", fontsize=17, fontweight="bold")  # Name the implemented system and genuine solver.
    fig.text(0.07, -0.035, "Common finite reference: background h0/4; convergence is not established. Equal caps, unequal actual equations.\nThe added ranker matches the existing WM choices on both prospective cases; no separate prospective ranker gain is shown.", fontsize=9, color="#434B58", va="top")  # State limitations that materially affect the interpretation.
    fig.tight_layout(h_pad=2.8)  # Allocate adequate space for panel titles and annotations.
    save_figure(fig, output, "comparison")  # Save reproducible measured comparison artifacts.
#
def exact_table(data, output):  # Provide the exact values beside the higher-level chart.
    rows = []  # Collect phase-aware case rows.
    for case in data["cases"]:  # Retain all recorded completed cases.
        phase = "pilot" if case["phase"] == "diagnostic_pilot" else "prospective"  # Distinguish the two evidence phases.
        rows.append([f"{case['seed']}\n{phase}"] + [f"{method_result(case, name)['reference_error']:.6f}\nN = {method_result(case, name)['n_equations']}" for name, _, _ in METHODS])  # Pair each discrepancy with its actual equation count.
    fig, axis = plt.subplots(figsize=(12.8, 4.4))  # Allocate one compact exact-results artifact.
    axis.axis("off")  # Use a real table without irrelevant chart axes.
    table = axis.table(cellText=rows, colLabels=["Case / phase"] + [label for _, label, _ in METHODS], cellLoc="center", loc="center", colWidths=[0.13, 0.235, 0.18, 0.235, 0.22])  # Keep the method mapping explicit.
    table.auto_set_font_size(False)  # Use deliberate readable table typography.
    table.set_fontsize(10)  # Choose a stable font size across long method labels.
    table.scale(1.0, 2.8)  # Give two-line cells sufficient vertical room.
    for (row, column), cell in table.get_celld().items():  # Apply restrained header and phase styling.
        cell.set_edgecolor("#D7DCE2")  # Make table cell boundaries subtle but precise.
        if row == 0:  # Emphasize column headings once.
            cell.set_facecolor("#EDF1F5")  # Use a neutral header background.
            cell.set_text_props(weight="bold", fontsize=9)  # Keep long headings readable without overflow.
        elif row >= 4:  # Distinguish the two later prospective rows.
            cell.set_facecolor("#F1F8F7")  # Use a light semantic phase background.
    axis.set_title("Measured finite-reference discrepancy and actual free equations", loc="left", fontsize=15, fontweight="bold", pad=20)  # Describe the table values directly.
    fig.text(0.08, 0.015, "All five cases use the common h0/4 finite reference. These values are not proven errors against the exact solution.", fontsize=9, color="#434B58")  # Preserve the central reference limitation.
    save_figure(fig, output, "exact_results_table")  # Save precise companion values for review and reuse.
#
def action_overlay(runs, output, seed):  # Visualize a sealed GPT image decision and its actual compiled mesh action.
    direct = runs / f"gpt_direct_{seed}"  # Locate genuine direct-GPT branch evidence.
    decision = read_json(direct / "originaldecision.json")  # Read the pre-solve explicit region geometry.
    summary = read_json(direct / "summary.json")  # Locate the source observation and actual outcome files.
    source = Path(summary["provenance"]["case_dir"])  # Read the recorded source case location.
    if not source.exists():  # Support a later checkout at a different absolute workspace path.
        source = runs / ("gpt_holdout" if seed >= 1000 else "visual_wm_probe") / f"test_bearing_{seed}"  # Resolve the same named saved case within the current runs directory.
    observation, initial = np.load(source / "observation.npz"), np.load(source / "initial.npz")  # Load the genuine current-state raster and finite-element mesh.
    image, bbox = observation["image"], observation["bbox"]  # Preserve field-channel and physical-coordinate definitions.
    mesh = Mesh(initial["nodes"], initial["cells"], 3)  # Use the actual initial tetrahedral mesh.
    selected = decision["primary"]  # Display the image decision selected before future branch outcomes.
    action = next(candidate for candidate in decision["candidates"] if candidate["name"] == selected)  # Retrieve only its original explicit regions.
    target = np.load(direct / f"target_{selected}.npz")["executed_target"]  # Load the actual executed source-node target-size map after common budget correction and gradation.
    next_data = np.load(direct / f"post_{selected}.npz")  # Load the actual regenerated and solved mesh.
    next_mesh = Mesh(next_data["nodes"], next_data["cells"], 3)  # Reconstruct topology from saved genuine connectivity.
    width = bbox[3:] - bbox[:3]  # Retain the physical box dimensions used by the GPT normalized coordinates.
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 10.1), constrained_layout=True)  # Present the observation, decision, and actual realization together.
    for axis, dimensions, eliminate, panel in ((axes[0, 0], (0, 1), 2, "A  Current raster and original GPT regions: XY"), (axes[0, 1], (0, 2), 1, "B  The same sealed regions through depth: XZ")):  # Use complementary views to reveal the actual three-dimensional action.
        field = image[2].sum(axis=eliminate) * width[eliminate] / image.shape[1]  # Integrate the conserved current estimator density along the omitted physical axis.
        positive = field[field > 0]  # Determine a logarithmic plotting scale from true positive raster values.
        im = axis.imshow(field.T, origin="lower", extent=(0, 1, 0, 1), cmap="magma", norm=LogNorm(vmin=max(float(positive.min()), float(positive.max()) * 1e-4), vmax=float(positive.max())), aspect="equal")  # Display the actual numerical raster in normalized physical coordinates.
        for index, region in enumerate(action["regions"], 1):  # Overlay each original GPT region without changing any location.
            lower, upper = np.asarray(region["lo"]), np.asarray(region["hi"])  # Read the exact sealed three-dimensional box corners.
            axis.add_patch(Rectangle(lower[list(dimensions)], *(upper - lower)[list(dimensions)], fill=False, edgecolor="#47E8D1", lw=1.6))  # Draw the explicit normalized box projection.
        axis.set_title(panel, loc="left", fontsize=11, fontweight="bold")  # State that this is current numerical raster evidence with an explicit action overlay.
        axis.set_xlabel("Normalized " + "xyz"[dimensions[0]])  # Label the physical-coordinate normalization.
        axis.set_ylabel("Normalized " + "xyz"[dimensions[1]])  # Preserve depth direction in the XZ view.
        fig.colorbar(im, ax=axis, shrink=0.77, label="Integrated current eta-squared density")  # Avoid presenting the raster as a fabricated finite-element stress contour.
    nodes = (mesh.nodes - bbox[:3]) / width  # Normalize actual source nodes for a consistent action-map view.
    ratio = target / mesh.node_sizes  # Measure executed action strength relative to actual local source mesh size.
    colors = Normalize(vmin=float(ratio.min()), vmax=float(ratio.max()))  # Show the full genuine executed ratio range.
    scatter = axes[1, 0].scatter(nodes[:, 0], nodes[:, 1], c=ratio, cmap="viridis", norm=colors, s=16, linewidths=0, alpha=0.85)  # Plot actual source-node values with all Z levels explicitly projected.
    axes[1, 0].set_title("C  Executed nodal size ratio: all Z levels in XY", loc="left", fontsize=11, fontweight="bold")  # State the projection limitation directly.
    fig.colorbar(scatter, ax=axes[1, 0], shrink=0.77, label="Executed target h / current node h")  # Name the actual numerical action field.
    facets = next_mesh.boundary_facets  # Use the genuine boundary topology of the regenerated tetrahedral mesh.
    top = facets[np.all(np.isclose(next_mesh.nodes[facets, 2], bbox[5], atol=1e-7), axis=1)]  # Select only real triangles on the top physical surface.
    xy = (next_mesh.nodes[:, :2] - bbox[:2]) / width[:2]  # Normalize actual generated surface coordinates.
    edges = np.concatenate([top[:, [0, 1]], top[:, [1, 2]], top[:, [2, 0]]])  # Extract edges of those authentic surface triangles.
    edges = np.unique(np.sort(edges, axis=1), axis=0)  # Draw shared physical edges once.
    axes[1, 1].add_collection(LineCollection(xy[edges], colors="#253242", linewidths=0.6))  # Render the actual regenerated mesh instead of a schematic or generated contour.
    axes[1, 1].set_title(f"D  Actual generated top-surface mesh: {len(top)} triangles", loc="left", fontsize=11, fontweight="bold")  # Label only the displayed true surface subset.
    for axis in axes[1]:  # Align the two physical action and mesh views.
        axis.set(xlim=(0, 1), ylim=(0, 1), xlabel="Normalized x", ylabel="Normalized y", aspect="equal")  # Retain the same physical mapping as the raster overlays.
    fig.suptitle(f"Case {seed}: GPT image observation -> sealed regions -> real mesh\nPrimary action: {selected}; box dimensions {width[0]:g} x {width[1]:g} x {width[2]:g} mm", fontsize=15, fontweight="bold")  # State the concrete case, action, and physical scale.
    save_figure(fig, output, "gpt_action_overlay")  # Persist a reproducible causal action visualization.
#
def main():  # Execute the compact scientific plotting workflow.
    parser = argparse.ArgumentParser(description=__doc__)  # Explain the plotting scope in command help.
    parser.add_argument("--results", type=Path, default=ROOT / "runs/gpt_visual_results.json")  # Default to the unified finite-reference evidence table.
    parser.add_argument("--output", type=Path, default=ROOT / "runs/gpt_visual_figures")  # Save plots alongside the corresponding experiment evidence.
    parser.add_argument("--seed", type=int, default=1002)  # Default the action illustration to a prospectively held-out case.
    args = parser.parse_args()  # Read explicit reproducibility options.
    args.output.mkdir(parents=True, exist_ok=True)  # Create only the requested figure artifact directory.
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.labelcolor": "#253242", "text.color": "#253242", "savefig.facecolor": "white"})  # Use available scientific typography without relying on missing Chinese fonts.
    data = read_json(args.results)  # Read exact completed experimental outcomes.
    comparison(data, args.output)  # Render the phase-aware scientific comparison.
    exact_table(data, args.output)  # Render exact discrepancies paired with actual equations.
    action_overlay(args.results.parent, args.output, args.seed)  # Render the actual GPT action's geometric and mesh consequences.
    print(json.dumps({"figures": [str(args.output / f"{name}.png") for name in ("comparison", "exact_results_table", "gpt_action_overlay")], "cases": [case["seed"] for case in data["cases"]]}, indent=2))  # Report generated paths without claiming victory.
#
if __name__ == "__main__":  # Run only when the reusable plotting script is explicitly invoked.
    main()  # Produce measured scientific figure artifacts.

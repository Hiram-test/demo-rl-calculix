from __future__ import annotations  # Enable postponed annotations for this compact headless plotting script.
from pathlib import Path  # Resolve input tables and output figures relative to the experiment directory.
import pandas as pd  # Read the solved adaptive history and region-action diagnostics.
import matplotlib.pyplot as plt  # Draw the requested script-generated linear-axis figures.

ROOT = Path(__file__).resolve().parent  # Anchor all plotting inputs and outputs to the experiment directory.
RESULTS_DIR = ROOT / "results"  # Read compact numerical results and write final figures here.
HISTORY_PATH = RESULTS_DIR / "history.csv"  # Locate the actual solved precision-resource trajectory table.
ACTIONS_PATH = RESULTS_DIR / "region_actions.csv"  # Locate the exact per-region theta, rank, ratio, and marked-count diagnostics.
ERROR_DOF_PATH = RESULTS_DIR / "error_dof_linear.png"  # Define the main linear-coordinate Pareto-style figure path.
INTENSITY_PATH = RESULTS_DIR / "semantic_intensity.png"  # Define the explicit LLM rank-to-size-ratio diagnostic figure path.

history = pd.read_csv(HISTORY_PATH)  # Load every solved global and semantic mesh state.
global_df = history[history["method"] == "global_dorfler"].sort_values("dof").copy()  # Extract and resource-order the conventional global Dörfler states.
semantic_df = history[history["method"] == "semantic_ranked_intensity"].sort_values("dof").copy()  # Extract and resource-order the LLM-ranked semantic states.

fig, ax = plt.subplots(figsize=(10.5, 6.2))  # Create one uncluttered figure with ordinary linear coordinates on both axes.
ax.plot(global_df["dof"], 100.0 * global_df["qoi_rel_error"], marker="o", linewidth=1.8, label="Global Dörfler: q=0.8")  # Plot every actual global solver state as a connected scatter trajectory.
ax.plot(semantic_df["dof"], 100.0 * semantic_df["qoi_rel_error"], marker="s", linewidth=1.8, label="Semantic rank intensity: same theta")  # Plot every actual semantic solver state on the same resource-error axes.
ax.set_xlabel("DOF (resource)")  # Label the horizontal axis by actual finite-element resource cost.
ax.set_ylabel("QoI relative error (%)")  # Label the vertical axis by true reference-relative QoI error.
ax.set_title("Fixed theta, LLM-ranked refinement intensity vs global Dörfler")  # State the purified algorithmic difference directly in the figure title.
ax.grid(True, alpha=0.25)  # Add a light Cartesian grid without logarithmic scaling.
ax.legend()  # Identify both actual solved trajectories.
for _, row in global_df.iterrows():  # Traverse global states for adaptive-round labels.
    ax.annotate(f"r{int(row['round'])}", (row["dof"], 100.0 * row["qoi_rel_error"]), xytext=(5, 5), textcoords="offset points", fontsize=8)  # Label each global solver state with its round index.
for _, row in semantic_df.iterrows():  # Traverse semantic states for adaptive-round labels.
    ax.annotate(f"r{int(row['round'])}", (row["dof"], 100.0 * row["qoi_rel_error"]), xytext=(5, -12), textcoords="offset points", fontsize=8)  # Label each semantic solver state with its round index.
fig.tight_layout()  # Adjust margins so all labels remain visible in the saved artifact.
fig.savefig(ERROR_DOF_PATH, dpi=220, bbox_inches="tight")  # Save the main linear-axis precision-resource figure.
plt.close(fig)  # Release the headless plotting resources explicitly.

actions = pd.read_csv(ACTIONS_PATH)  # Load exact refinement actions used to generate the semantic meshes.
semantic_actions = actions[actions["method"] == "semantic_ranked_intensity"].copy()  # Keep only semantic region actions for the intensity diagnostic.
region_table = semantic_actions.groupby("region", as_index=False).agg(rank=("rank", "first"), size_ratio=("size_ratio", "first"), mean_marked=("marked_elements", "mean"))  # Summarize the frozen LLM level and observed mean marking count per region.
region_table = region_table.sort_values("rank", ascending=False)  # Order the diagnostic by semantic refinement intensity rather than alphabetically.
fig, ax = plt.subplots(figsize=(8.8, 5.8))  # Create one simple region-level intensity diagnostic figure.
ax.bar(region_table["region"], region_table["size_ratio"])  # Show the physical target-size ratio assigned by the frozen LLM ordinal rank.
ax.set_ylabel("Target size ratio h_new / h_current")  # State exactly what the semantic intensity parameter controls physically.
ax.set_xlabel("Semantic region")  # Label the three exhaustive semantic domains.
ax.set_title("Frozen LLM semantic refinement intensity")  # Make clear that the bars are a pre-solve semantic prior rather than fitted outputs.
ax.set_ylim(0.0, 1.0)  # Use the natural physical range for a refinement size ratio.
for index, row in region_table.reset_index(drop=True).iterrows():  # Annotate every region with its ordinal level and average detected element count.
    ax.annotate(f"rank={int(row['rank'])}\nmean marked={row['mean_marked']:.1f}", (index, row["size_ratio"]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)  # Separate semantic intensity from Dörfler detection cardinality explicitly.
fig.tight_layout()  # Adjust margins before saving the categorical diagnostic.
fig.savefig(INTENSITY_PATH, dpi=220, bbox_inches="tight")  # Save the region-intensity figure beside the main Pareto-style plot.
plt.close(fig)  # Release the second headless figure explicitly.
print(f"[plot] wrote {ERROR_DOF_PATH}")  # Record the main figure path in the Actions log.
print(f"[plot] wrote {INTENSITY_PATH}")  # Record the semantic intensity diagnostic path in the Actions log.

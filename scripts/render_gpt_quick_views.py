"""Render only current observations for direct GPT visual decisions."""  # Exclude all future outcomes from this renderer.
import argparse  # Select the uninspected initial case.
from pathlib import Path  # Manage image artifacts.
import numpy as np  # Read raw current-state raster arrays.
import matplotlib  # Configure headless scientific rendering.
matplotlib.use("Agg")  # Avoid interactive display dependencies.
import matplotlib.pyplot as plt  # Show actual numerical observations.
parser = argparse.ArgumentParser(description=__doc__)  # Expose a reusable observation entry point.
parser.add_argument("case_dir", type=Path)  # Accept a directory containing the already-solved initial state.
args = parser.parse_args()  # Read the case path.
data = np.load(args.case_dir / "observation.npz")  # Load only present-state images without accessing outcome records.
image = data["image"]  # Retain the recorded seven-channel observation.
fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")  # Present three orthogonal physical views.
for column, (axis, xlabel, ylabel) in enumerate([(2, "x/L", "y/D"), (1, "x/L", "z/H"), (0, "y/D", "z/H")]):  # Define normalized spatial coordinates.
    projected = np.sum(image[2], axis=axis)  # Project currently observed estimator density.
    artist = axes[0, column].imshow(projected.T, origin="lower", extent=(0, 1, 0, 1), aspect="auto", cmap="magma")  # Render the current spatial error pattern.
    fig.colorbar(artist, ax=axes[0, column], label="Current estimator projection")  # Show the numerical scale honestly.
    load = np.sum(image[5], axis=axis)  # Project actual force boundary marks.
    support = np.sum(image[6], axis=axis)  # Project actual constrained-node marks.
    rgb = np.ones((*load.shape, 3))  # Set a neutral physical-background map.
    rgb[:, :, 0] -= 0.85 * support / max(support.max(), 1e-30)  # Encode supports in cyan.
    rgb[:, :, 1] -= 0.85 * load / max(load.max(), 1e-30)  # Encode applied-load marks in red.
    rgb[:, :, 2] -= 0.85 * load / max(load.max(), 1e-30)  # Complete the red-channel load code.
    axes[1, column].imshow(np.transpose(rgb, (1, 0, 2)), origin="lower", extent=(0, 1, 0, 1), aspect="auto")  # Show physical boundary positions without suggested actions.
    axes[1, column].set_title("Red: load; cyan: constrained nodes")  # Explain the boundary image encoding.
    for row in range(2):  # Give every panel matching normalized geometry axes.
        axes[row, column].set_xlabel(xlabel)  # Label the horizontal physical direction.
        axes[row, column].set_ylabel(ylabel)  # Label the vertical physical direction.
        axes[row, column].set_xticks(np.linspace(0, 1, 6))  # Support reproducible visual region identification.
        axes[row, column].set_yticks(np.linspace(0, 1, 6))  # Support reproducible depth identification.
        axes[row, column].grid(alpha=0.15)  # Add a subdued coordinate reference.
fig.suptitle(f"GPT input only: {args.case_dir.name} | current fields, no future outcomes")  # Declare the information available for decision making.
destination = args.case_dir / "gpt_input_current.png"  # Preserve the exact displayed input for audit.
fig.savefig(destination, dpi=150)  # Write a inspectable raster observation.
plt.close(fig)  # Release plotting memory.
print(destination)  # Return the exact picture location.

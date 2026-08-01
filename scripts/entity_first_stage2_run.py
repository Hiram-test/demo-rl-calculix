"""Batch-mode launcher for the Stage 2 Gmsh entity builder."""  # Isolate the first discovered integration defect.
from __future__ import annotations  # Enable modern annotations on Actions Python.
from pathlib import Path  # Resolve generated Gmsh source paths.
import shutil  # Locate the installed Gmsh executable.
import subprocess  # Execute Gmsh without launching the GUI.
import sys  # Import the Stage 2 module and propagate its status.
import entity_first_stage2_gmsh as stage2  # Reuse the tested IMP, BREP, parser, and audit logic.
def run_gmsh_batch(source_path: Path) -> None:  # Execute geometry and mesh source in explicit batch mode.
    executable = shutil.which("gmsh")  # Resolve the mature Gmsh command installed by the workflow.
    if executable is None:  # Detect a missing system dependency before model generation.
        raise RuntimeError("gmsh executable is unavailable")  # Stop without creating partial evidence.
    dimension = "0" if source_path.name == "geometry.geo" else "2"  # Parse-only for BREP creation and two-dimensional batch mode for meshes.
    command = (executable, source_path.as_posix(), f"-{dimension}", "-nopopup")  # Build the explicit non-GUI command.
    completed = subprocess.run(command, cwd=source_path.parent, text=True, capture_output=True, check=False)  # Execute the generated source deterministically.
    if completed.returncode != 0:  # Detect any OpenCASCADE or meshing failure.
        source_text = source_path.read_text(encoding="utf-8")  # Preserve the exact generated Gmsh program for diagnosis.
        raise RuntimeError(f"gmsh command failed: {command}\nsource:\n{source_text}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}")  # Return complete reproducible evidence.
stage2.run_gmsh = run_gmsh_batch  # Replace only the process-launch adapter while preserving model logic.
if __name__ == "__main__":  # Run only when invoked directly by the isolated workflow.
    raise SystemExit(stage2.main())  # Execute the Stage 2 builder with the corrected batch adapter.
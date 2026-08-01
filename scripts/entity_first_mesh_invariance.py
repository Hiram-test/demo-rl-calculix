"""Change only mesh parameters and prove the CAD entity remains unchanged."""  # Isolate mesh effects from geometry effects.
from __future__ import annotations  # Enable modern Python type annotations.
from hashlib import sha256  # Fingerprint immutable geometry bytes.
from pathlib import Path  # Handle artifact paths safely.
import argparse  # Parse experiment inputs.
import json  # Write machine-readable invariance evidence.
import shutil  # Locate Gmsh executable.
import subprocess  # Execute Gmsh batch meshing.

def digest(path: Path) -> str:  # Calculate one deterministic file fingerprint.
    return sha256(path.read_bytes()).hexdigest()  # Return the SHA256 geometry identity.

def run_gmsh(source: Path) -> None:  # Run one mesh generation experiment.
    gmsh = shutil.which("gmsh")  # Resolve the installed Gmsh executable.
    if gmsh is None:  # Stop if the mature mesher is unavailable.
        raise RuntimeError("gmsh unavailable")  # Preserve explicit dependency failure.
    result = subprocess.run((gmsh, source.as_posix(), "-2", "-nopopup"), cwd=source.parent, text=True, capture_output=True)  # Execute batch meshing.
    if result.returncode != 0:  # Reject failed mesh generation.
        raise RuntimeError(result.stderr)  # Preserve Gmsh failure output.

def main() -> int:  # Execute the entity invariance experiment.
    parser = argparse.ArgumentParser()  # Create the argument parser.
    parser.add_argument("--brep", required=True)  # Receive the frozen BREP entity.
    parser.add_argument("--geo-template", required=True)  # Receive the mesh-only template.
    parser.add_argument("--output", required=True)  # Receive the receipt path.
    args = parser.parse_args()  # Parse command arguments.
    brep = Path(args.brep)  # Normalize BREP path.
    before = digest(brep)  # Record the entity identity before remeshing.
    records = []  # Collect different mesh-size experiments.
    for label, size in (("coarse", 40.0), ("medium", 20.0), ("fine", 8.0)):  # Change only mesh size.
        output_dir = brep.parent / label  # Create an isolated mesh experiment directory.
        output_dir.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists.
        geo = output_dir / "mesh.geo"  # Allocate the generated mesh program.
        template = Path(args.geo_template).read_text(encoding="utf-8")  # Read the mesh template.
        geo.write_text(template.replace("{{SIZE}}", str(size)).replace("{{BREP}}", brep.as_posix()), encoding="utf-8")  # Substitute only mesh parameters and frozen entity path.
        run_gmsh(geo)  # Generate one new mesh from the same entity.
        after = digest(brep)  # Verify the entity bytes after remeshing.
        records.append({"label": label, "mesh_size": size, "brep_unchanged": before == after, "brep_sha256": after})  # Record the invariance result.
    Path(args.output).write_text(json.dumps({"schema_version":"entity-first-mesh-invariance/1.0","initial_brep_sha256":before,"experiments":records,"all_entity_unchanged":all(item["brep_unchanged"] for item in records)}, indent=2), encoding="utf-8")  # Save the final evidence.
    return 0  # Finish successfully after all checks.

if __name__ == "__main__":  # Run only when called directly.
    raise SystemExit(main())  # Return the process status.
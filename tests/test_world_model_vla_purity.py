"""Static purity tests for the world-model VLA method boundary."""  # Describe the clean-comparison validation module.
from __future__ import annotations  # Postpone annotation evaluation for compatibility.
from pathlib import Path  # Import deterministic repository source inspection.

def _source_files() -> list[Path]:  # Collect the complete world-model VLA implementation surface.
    root = Path(__file__).resolve().parents[1]  # Locate the repository root.
    return sorted((root / "visionamr" / "vla" / "world").glob("*.py")) + [root / "scripts" / "run_world_model_bridge.py"]  # Return package and command sources only.

def test_world_vla_does_not_import_local_prediction_or_optimizers() -> None:  # Prevent accidental stacking with local prediction or black-box tuning.
    forbidden = ("predicted_sizes", "local_prediction", "eta_allow", "PSOConfig", "allow_pso", "scipy.optimize", "nelder", "particle_swarm")  # Define implementation tokens that would violate the clean method boundary.
    violations: list[str] = []  # Collect exact forbidden-token locations.
    for path in _source_files():  # Inspect every world-model VLA source file.
        text = path.read_text(encoding="utf-8")  # Read the committed source exactly.
        for token in forbidden:  # Check every forbidden dependency or optimizer token.
            if token.lower() in text.lower():  # Detect direct or differently cased references.
                violations.append(f"{path.name}:{token}")  # Record the offending file and token.
    assert not violations, "method-purity violations: " + ", ".join(violations)  # Require full separation from local prediction and black-box tuning.

def test_vision_partition_contains_no_mesh_size_action() -> None:  # Ensure semantic perception cannot tune continuous mesh parameters.
    root = Path(__file__).resolve().parents[1]  # Locate the repository root.
    text = (root / "visionamr" / "vla" / "world" / "vision_partition.py").read_text(encoding="utf-8")  # Read the cached-perception implementation.
    forbidden = ("refine_size_map", "NodalSizeField", "target_sizes", "extra_depth", "dorfler_mark")  # Define action and parameter APIs forbidden from the perception layer.
    violations = [token for token in forbidden if token in text]  # Detect any perception-to-action leakage.
    assert not violations, "vision layer contains mesh actions: " + ", ".join(violations)  # Require a region-only cached vision output.

def test_tool_gateway_is_the_only_parameter_materializer() -> None:  # Keep numerical mesh action construction inside the deterministic tool boundary.
    files = _source_files()  # Collect the implementation surface.
    materializers: list[str] = []  # Collect files that invoke the exact nodal target-field API.
    for path in files:  # Inspect every implementation file.
        text = path.read_text(encoding="utf-8")  # Read the committed source exactly.
        if "refine_size_map" in text or "NodalSizeField" in text:  # Detect exact mesh-parameter materialization APIs.
            materializers.append(path.name)  # Record the responsible source file.
    assert materializers == ["tool_gateway.py"]  # Require one deterministic parameter authority.

def test_world_actions_are_discrete_nonnegative_depths() -> None:  # Verify the action contract contains no continuous LLM-generated mesh size.
    root = Path(__file__).resolve().parents[1]  # Locate the repository root.
    text = (root / "visionamr" / "vla" / "world" / "model.py").read_text(encoding="utf-8")  # Read the action contract.
    assert "tuple[int, ...]" in text  # Require integer regional action depths.
    assert "depth < 0" in text  # Require explicit rejection of coarsening actions.
    assert "generic field regions cannot receive world-model depth" in text  # Require explicit exclusion of the generic field region.

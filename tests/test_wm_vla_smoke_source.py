# Source-contract tests for the native WM-VLA closure and bridge-specific visual partition.  # Test module purpose.
from pathlib import Path  # Locate and inspect the newly added Python sources.


def _assert_line_comments(path: Path) -> None:  # Enforce the user's per-line code-comment requirement on one source.
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):  # Inspect every physical source line.
        if not line.strip():  # Permit blank separators for readability.
            continue  # Move to the next physical line.
        assert "#" in line, f"{path}:{line_number} lacks a line-level comment"  # Require an explicit comment marker on every nonblank line.


def test_native_smoke_script_keeps_line_level_comments() -> None:  # Verify the reference-free native executable source.
    root = Path(__file__).resolve().parents[1]  # Locate the repository root deterministically.
    _assert_line_comments(root / "scripts" / "smoke_wm_vla_bridge3d.py")  # Inspect the native closure script.


def test_bridge_partition_keeps_line_level_comments() -> None:  # Verify the true three-dimensional geometry partition source.
    root = Path(__file__).resolve().parents[1]  # Locate the repository root deterministically.
    _assert_line_comments(root / "visionamr" / "vla" / "bridge_partition.py")  # Inspect the section-aware bridge partitioner.

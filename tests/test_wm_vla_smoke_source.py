# Source-contract test for the reference-free native WM-VLA closure script.  # Test module purpose.
from pathlib import Path  # Locate and inspect the newly added executable source.


def test_native_smoke_script_keeps_line_level_comments() -> None:  # Enforce the user's per-line code-comment requirement.
    root = Path(__file__).resolve().parents[1]  # Locate the repository root deterministically.
    path = root / "scripts" / "smoke_wm_vla_bridge3d.py"  # Select the new native closure script.
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):  # Inspect every physical source line.
        if not line.strip():  # Permit blank separators for readability.
            continue  # Move to the next physical line.
        assert "#" in line, f"{path}:{line_number} lacks a line-level comment"  # Require an explicit comment marker on every nonblank line.

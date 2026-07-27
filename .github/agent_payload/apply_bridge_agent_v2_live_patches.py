from __future__ import annotations

import sys
from pathlib import Path


def patch_skills(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "\ndef _artifact_payload(ctx: RunContext, kind: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:\n"
    helper = r'''

def _grid_pair_levels(value: Any) -> list[tuple[int, int]] | None:
    """Accept explicit [[n1,n2], ...] grid counts only when valid."""
    if not isinstance(value, (list, tuple)) or not value:
        return None
    pairs: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        try:
            a, b = int(item[0]), int(item[1])
        except (TypeError, ValueError):
            return None
        if a < 4 or b < 2:
            return None
        pairs.append((a, b))
    return pairs


def _bearing_levels(model: dict[str, Any], args: dict[str, Any]) -> list[tuple[int, int]]:
    """Normalize explicit grids or scalar target sizes from DeepSeek."""
    raw = args.get("levels")
    pairs = _grid_pair_levels(raw)
    if pairs:
        return pairs
    if isinstance(raw, (list, tuple)) and raw:
        try:
            sizes = [float(x) for x in raw]
        except (TypeError, ValueError):
            sizes = []
        if sizes and all(x >= 2.0 for x in sizes):
            length = float(model["length_mm"])
            height = float(model["height_mm"])
            return [
                (max(4, int(round(length / h))), max(2, int(round(height / h))))
                for h in sizes
            ]
    return [(40, 8), (80, 16), (160, 32)]


def _circular_levels(args: dict[str, Any]) -> list[tuple[int, int]]:
    """Resolve scalar level labels to the audited circular-hole mesh contract."""
    pairs = _grid_pair_levels(args.get("levels"))
    return pairs or [(48, 6), (96, 10), (160, 16)]
'''
    if "_bearing_levels(" not in text:
        if marker not in text:
            raise SystemExit("skills insertion marker not found")
        text = text.replace(marker, helper + marker, 1)

    old_b = 'm=ctx.case.model; levels=args.get("levels") or [[40,8],[80,16],[160,32]]'
    new_b = 'm=ctx.case.model; levels=_bearing_levels(m,args)'
    old_c = 'm=ctx.case.model; levels=args.get("levels") or [[48,6],[96,10],[160,16]]'
    new_c = 'm=ctx.case.model; levels=_circular_levels(args)'
    if old_b in text:
        text = text.replace(old_b, new_b, 1)
    elif new_b not in text:
        raise SystemExit("bearing replacement marker not found")
    if old_c in text:
        text = text.replace(old_c, new_c, 1)
    elif new_c not in text:
        raise SystemExit("circular replacement marker not found")
    path.write_text(text, encoding="utf-8")


def patch_tests(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    marker = "    def test_circular_focused_mesh(self) -> None:\n"
    test = '''    def test_model_mesh_level_shorthands_are_normalized(self) -> None:\n        bearing = self.ctx("bearing_load")\n        result = run_bearing_point_load_sequence(bearing, {"levels": ["20", "10", "5"]})\n        self.assertEqual(result.status, "completed")\n        rows = bearing.read_artifact(bearing.latest("solver_evidence").artifact_id)["levels"]\n        self.assertEqual([row["mesh"] for row in rows], [[50, 5], [100, 10], [200, 20]])\n\n        circular = self.ctx("circular_opening")\n        result = run_circular_baseline_sequence(circular, {"levels": [1, 2, 3, 4]})\n        self.assertEqual(result.status, "completed")\n        rows = circular.read_artifact(circular.latest("solver_evidence").artifact_id)["levels"]\n        self.assertEqual(len(rows), 3)\n        self.assertEqual(rows[0]["mesh"], {"ntheta": 48, "nr": 6})\n\n'''
    if "test_model_mesh_level_shorthands_are_normalized" not in text:
        if marker not in text:
            raise SystemExit("test insertion marker not found")
        text = text.replace(marker, test + marker, 1)
        path.write_text(text, encoding="utf-8")


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: apply_bridge_agent_v2_live_patches.py ROOT")
    root = Path(sys.argv[1])
    patch_skills(root / "bridge_agent" / "skills.py")
    patch_tests(root / "tests" / "test_bridge_agent.py")
    print("bridge-agent v2 live patches applied")


if __name__ == "__main__":
    main()

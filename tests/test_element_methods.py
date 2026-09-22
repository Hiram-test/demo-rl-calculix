"""Doerfler and local prediction must stay element-wise (never regional).

The contamination vector would be an import from ``visionamr.vla`` (the
partition/agent machinery), so the source checks are AST-based import
checks rather than brittle substring scans of comments and docstrings.
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    mods: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = "." * node.level
            mods.add(prefix + (node.module or ""))
    return mods


def _assert_no_vla_dependency(path: Path) -> None:
    mods = _imported_modules(path)
    offenders = [m for m in mods if "vla" in m.split(".")]
    assert not offenders, f"{path.name} imports VLA machinery: {offenders}"
    text = path.read_text()
    for symbol in ("Partition(", "Seed(", "ScriptedVisionPartitioner"):
        assert symbol not in text, f"{path.name} uses regional symbol {symbol!r}"


def test_dorfler_source_is_elementwise():
    _assert_no_vla_dependency(ROOT / "visionamr" / "baselines" / "dorfler.py")


def test_local_prediction_source_is_elementwise():
    _assert_no_vla_dependency(
        ROOT / "visionamr" / "baselines" / "local_prediction.py"
    )


def test_dorfler_mark_object_is_element():
    import numpy as np

    from visionamr.marking import dorfler_mark

    eta2 = np.array([100.0, 1e-6, 1e-6, 1e-6])
    assert list(dorfler_mark(eta2, 0.5)) == [0]


def test_predicted_sizes_are_per_element():
    import numpy as np

    from visionamr.baselines.local_prediction import predicted_sizes
    from visionamr.mesher import Mesh

    nodes = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    cells = np.array([[0, 1, 2], [0, 2, 3]])
    mesh = Mesh(nodes=nodes, cells=cells, dim=2)
    eta2 = np.array([10.0, 0.1])
    h = predicted_sizes(mesh, eta2, n_target=20)
    assert h.shape == (2,)
    assert h[0] < h[1]

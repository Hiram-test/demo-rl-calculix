"""Dörfler and local prediction must stay element-wise (never regional)."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_dorfler_source_has_no_partition():
    text = (ROOT / "visionamr" / "baselines" / "dorfler.py").read_text()
    assert "Partition" not in text
    assert "Seed" not in text
    assert "region" not in text.lower()


def test_local_prediction_source_has_no_partition():
    text = (ROOT / "visionamr" / "baselines" / "local_prediction.py").read_text()
    assert "Partition" not in text
    assert "Seed" not in text
    # the module may mention "region" only if someone sneaks it in
    assert "vla" not in text


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

import numpy as np
import pytest

from visionamr.marking import dorfler_mark, max_mark


def test_dorfler_minimal_cardinality():
    eta2 = np.array([4.0, 1.0, 3.0, 2.0])  # total 10
    marked = dorfler_mark(eta2, theta=0.5)
    # greedy: 4 (idx0) then 3 (idx2) -> sum 7 >= 5; minimal set has 2 elems
    assert set(marked) == {0, 2}


def test_dorfler_theta_one_marks_everything():
    eta2 = np.array([1.0, 2.0, 3.0])
    assert len(dorfler_mark(eta2, theta=1.0)) == 3


def test_dorfler_object_is_the_element():
    # marking must never aggregate: a single dominant element is enough
    eta2 = np.array([100.0, 1e-6, 1e-6, 1e-6])
    marked = dorfler_mark(eta2, theta=0.5)
    assert list(marked) == [0]


def test_dorfler_rejects_bad_theta():
    with pytest.raises(ValueError):
        dorfler_mark(np.ones(3), theta=0.0)


def test_max_mark():
    eta2 = np.array([1.0, 0.1, 0.5])
    assert set(max_mark(eta2, theta=0.25)) == {0, 2}

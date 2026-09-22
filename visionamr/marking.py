"""Marking strategies for element-wise AFEM baselines."""

from __future__ import annotations

import numpy as np


def dorfler_mark(eta2: np.ndarray, theta: float = 0.5) -> np.ndarray:
    """Doerfler (1996) bulk marking.

    Returns the minimal-cardinality element set M (greedy by decreasing
    indicator) with  sum_{K in M} eta_K^2 >= theta * sum_K eta_K^2.
    The marking object is the element, never an aggregate.
    """

    if not 0.0 < theta <= 1.0:
        raise ValueError("theta must be in (0, 1]")
    order = np.argsort(eta2)[::-1]
    csum = np.cumsum(eta2[order])
    total = csum[-1]
    if total <= 0.0:
        return np.array([], dtype=np.int64)
    k = int(np.searchsorted(csum, theta * total) + 1)
    return np.sort(order[:k])


def max_mark(eta2: np.ndarray, theta: float = 0.25) -> np.ndarray:
    """Maximum-criterion marking: eta_K^2 > theta * max eta^2."""

    m = eta2.max()
    if m <= 0:
        return np.array([], dtype=np.int64)
    return np.nonzero(eta2 > theta * m)[0]

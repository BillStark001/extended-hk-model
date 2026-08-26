"""Cell-integrated bounded-confidence geometry on a uniform opinion grid.

The state variables in the mesoscopic solvers are cell masses, not point
values.  Evaluating the hard confidence condition only at cell centers makes
the interaction domain jump whenever ``epsilon / dx`` crosses an integer.
This module integrates the condition over both cells and caches the resulting
geometry for reuse throughout a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import pairwise
from typing import Literal

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ConfidenceMode = Literal["cell_average", "center"]
_GL_NODES, _GL_WEIGHTS = leggauss(5)


@dataclass(frozen=True)
class ConfidenceGeometry:
    """Pair-cell moments of the bounded-confidence indicator.

    For source cell ``i`` and target cell ``j``, ``concordance[i, j]`` is
    ``P(|X-Y| <= epsilon)`` for independent uniform points in the two cells.
    ``target_first`` and ``target_second`` are the corresponding unnormalised
    moments of ``Y``; ``displacement`` is the unnormalised moment of ``Y-X``.
    """

    concordance: FloatArray
    target_first: FloatArray
    target_second: FloatArray
    displacement: FloatArray


def _validate_uniform_grid(x: FloatArray) -> float:
    if x.ndim != 1 or x.size < 2:
        raise ValueError("opinion grid must be a one-dimensional array")
    spacing = np.diff(x)
    if not np.all(np.isfinite(x)) or not np.allclose(
        spacing, spacing[0], rtol=1e-12, atol=1e-14
    ):
        raise ValueError("opinion grid must be finite and uniformly spaced")
    if spacing[0] <= 0:
        raise ValueError("opinion grid must be strictly increasing")
    return float(spacing[0])


@lru_cache(maxsize=64)
def _cached_confidence_geometry(
    centers: tuple[float, ...],
    epsilon: float,
    mode: ConfidenceMode,
) -> ConfidenceGeometry:
    x = np.asarray(centers, dtype=float)
    dx = _validate_uniform_grid(x)
    if not 0 < epsilon <= 2:
        raise ValueError("epsilon must lie in (0, 2]")
    if mode not in {"cell_average", "center"}:
        raise ValueError(f"unknown confidence mode: {mode}")

    delta = x[None, :] - x[:, None]
    if mode == "center":
        concordance = (np.abs(delta) <= epsilon).astype(float)
        target_first = concordance * x[None, :]
        target_second = concordance * x[None, :] ** 2
        displacement = concordance * delta
    else:
        lower = x - 0.5 * dx
        upper = x + 0.5 * dx
        size = x.size
        concordance = np.zeros((size, size), dtype=float)
        target_first = np.zeros_like(concordance)
        target_second = np.zeros_like(concordance)
        source_first = np.zeros_like(concordance)

        # The clipped target interval changes formula only at these four
        # breakpoints.  Five-point Gauss--Legendre integration is exact on
        # every resulting polynomial segment and is paid only once per grid.
        for source in range(size):
            source_lo = lower[source]
            source_hi = upper[source]
            for target in range(size):
                target_lo = lower[target]
                target_hi = upper[target]
                breakpoints = [source_lo, source_hi]
                for point in (
                    target_lo - epsilon,
                    target_hi - epsilon,
                    target_lo + epsilon,
                    target_hi + epsilon,
                ):
                    if source_lo < point < source_hi:
                        breakpoints.append(float(point))
                breakpoints = sorted(set(breakpoints))
                for left, right in pairwise(breakpoints):
                    half = 0.5 * (right - left)
                    midpoint = 0.5 * (right + left)
                    source_points = midpoint + half * _GL_NODES
                    clipped_lo = np.maximum(target_lo, source_points - epsilon)
                    clipped_hi = np.minimum(target_hi, source_points + epsilon)
                    active = clipped_hi > clipped_lo
                    length = np.where(active, clipped_hi - clipped_lo, 0.0)
                    weights = half * _GL_WEIGHTS / (dx * dx)
                    concordance[source, target] += np.sum(weights * length)
                    target_first[source, target] += np.sum(
                        weights
                        * np.where(
                            active,
                            0.5 * (clipped_hi**2 - clipped_lo**2),
                            0.0,
                        )
                    )
                    target_second[source, target] += np.sum(
                        weights
                        * np.where(
                            active,
                            (clipped_hi**3 - clipped_lo**3) / 3.0,
                            0.0,
                        )
                    )
                    source_first[source, target] += np.sum(
                        weights * source_points * length
                    )
        displacement = target_first - source_first
        concordance = np.clip(concordance, 0.0, 1.0)

    for array in (concordance, target_first, target_second, displacement):
        array.setflags(write=False)
    return ConfidenceGeometry(
        concordance=concordance,
        target_first=target_first,
        target_second=target_second,
        displacement=displacement,
    )


def confidence_geometry(
    x: FloatArray,
    epsilon: float,
    mode: ConfidenceMode = "cell_average",
) -> ConfidenceGeometry:
    """Return cached pair-cell confidence moments for a uniform grid."""

    values = np.asarray(x, dtype=float)
    _validate_uniform_grid(values)
    return _cached_confidence_geometry(
        tuple(float(value) for value in values), float(epsilon), mode
    )


__all__ = ["ConfidenceGeometry", "ConfidenceMode", "confidence_geometry"]

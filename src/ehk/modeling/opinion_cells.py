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
IntArray = NDArray[np.int64]
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
    displacement_second: FloatArray


@dataclass(frozen=True)
class CompromiseJumpGeometry:
    """Sparse pair-cell jump quadrature in padded destination slots.

    For each source/target cell pair, the non-negative values in ``weight``
    sum to one conditional on bounded-confidence eligibility. Corresponding
    ``destination_index`` entries identify the destination opinion cells.
    """

    destination_index: IntArray
    weight: FloatArray


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
        displacement_second = concordance * delta * delta
    else:
        lower = x - 0.5 * dx
        upper = x + 0.5 * dx
        size = x.size
        concordance = np.zeros((size, size), dtype=float)
        target_first = np.zeros_like(concordance)
        target_second = np.zeros_like(concordance)
        source_first = np.zeros_like(concordance)
        displacement_second = np.zeros_like(concordance)

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
                    displacement_second[source, target] += np.sum(
                        weights
                        * np.where(
                            active,
                            (
                                (clipped_hi - source_points) ** 3
                                - (clipped_lo - source_points) ** 3
                            )
                            / 3.0,
                            0.0,
                        )
                    )
        displacement = target_first - source_first
        concordance = np.clip(concordance, 0.0, 1.0)

    for array in (
        concordance,
        target_first,
        target_second,
        displacement,
        displacement_second,
    ):
        array.setflags(write=False)
    return ConfidenceGeometry(
        concordance=concordance,
        target_first=target_first,
        target_second=target_second,
        displacement=displacement,
        displacement_second=displacement_second,
    )


def _deposit_jump_mass(
    result: dict[int, float],
    destination: FloatArray,
    mass: FloatArray,
    x: FloatArray,
) -> None:
    """Accumulate positive quadrature mass with linear grid deposition."""

    dx = float(x[1] - x[0])
    coordinate = (destination - x[0]) / dx
    lower = np.clip(np.floor(coordinate).astype(np.int64), 0, x.size - 1)
    upper = np.minimum(lower + 1, x.size - 1)
    fraction = np.clip(coordinate - lower, 0.0, 1.0)
    for lower_index, upper_index, upper_weight, value in zip(
        lower, upper, fraction, mass, strict=True
    ):
        result[int(lower_index)] = result.get(int(lower_index), 0.0) + float(
            value * (1.0 - upper_weight)
        )
        result[int(upper_index)] = result.get(int(upper_index), 0.0) + float(
            value * upper_weight
        )


@lru_cache(maxsize=16)
def _cached_compromise_jump_geometry(
    centers: tuple[float, ...],
    epsilon: float,
    influence: float,
    mode: ConfidenceMode,
) -> CompromiseJumpGeometry:
    """Integrate the nonlocal Deffuant kernel once per numerical protocol."""

    x = np.asarray(centers, dtype=float)
    dx = _validate_uniform_grid(x)
    if not 0 <= influence <= 1:
        raise ValueError("influence must lie in [0, 1]")
    confidence = _cached_confidence_geometry(centers, epsilon, mode)
    lower_edge = x - 0.5 * dx
    upper_edge = x + 0.5 * dx
    pair_maps: list[list[dict[int, float]]] = [
        [{} for _ in range(x.size)] for _ in range(x.size)
    ]
    maximum_width = 1
    for source in range(x.size):
        for target in np.flatnonzero(confidence.concordance[source] > 1e-15):
            target = int(target)
            contributions = pair_maps[source][target]
            if mode == "center" or influence == 0:
                destination = np.asarray(
                    [x[source] + influence * (x[target] - x[source])]
                )
                _deposit_jump_mass(
                    contributions, destination, np.ones(1, dtype=float), x
                )
            else:
                source_lo = lower_edge[source]
                source_hi = upper_edge[source]
                target_lo = lower_edge[target]
                target_hi = upper_edge[target]
                breakpoints = [source_lo, source_hi]
                for point in (
                    target_lo - epsilon,
                    target_hi - epsilon,
                    target_lo + epsilon,
                    target_hi + epsilon,
                ):
                    if source_lo < point < source_hi:
                        breakpoints.append(float(point))
                for left, right in pairwise(sorted(set(breakpoints))):
                    source_half = 0.5 * (right - left)
                    source_midpoint = 0.5 * (right + left)
                    source_points = source_midpoint + source_half * _GL_NODES
                    source_weights = source_half * _GL_WEIGHTS / dx
                    for source_point, source_weight in zip(
                        source_points, source_weights, strict=True
                    ):
                        clipped_lo = max(target_lo, source_point - epsilon)
                        clipped_hi = min(target_hi, source_point + epsilon)
                        if clipped_hi <= clipped_lo:
                            continue
                        target_half = 0.5 * (clipped_hi - clipped_lo)
                        target_midpoint = 0.5 * (clipped_hi + clipped_lo)
                        target_points = target_midpoint + target_half * _GL_NODES
                        joint_weight = source_weight * target_half * _GL_WEIGHTS / dx
                        destination = x[source] + influence * (
                            target_points - source_point
                        )
                        _deposit_jump_mass(contributions, destination, joint_weight, x)
            total = sum(contributions.values())
            if total <= 0:
                raise FloatingPointError("eligible jump pair has zero quadrature mass")
            for destination in contributions:
                contributions[destination] /= total
            maximum_width = max(maximum_width, len(contributions))

    destination_index = np.full((x.size, x.size, maximum_width), -1, dtype=np.int64)
    weight = np.zeros((x.size, x.size, maximum_width), dtype=float)
    for source, rows in enumerate(pair_maps):
        for target, contributions in enumerate(rows):
            for slot, (destination, value) in enumerate(sorted(contributions.items())):
                destination_index[source, target, slot] = destination
                weight[source, target, slot] = value
    destination_index.setflags(write=False)
    weight.setflags(write=False)
    return CompromiseJumpGeometry(destination_index, weight)


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


def compromise_jump_geometry(
    x: FloatArray,
    epsilon: float,
    influence: float,
    mode: ConfidenceMode = "cell_average",
) -> CompromiseJumpGeometry:
    """Return a cached cell-integrated nonlocal compromise kernel."""

    values = np.asarray(x, dtype=float)
    _validate_uniform_grid(values)
    return _cached_compromise_jump_geometry(
        tuple(float(value) for value in values),
        float(epsilon),
        float(influence),
        mode,
    )


__all__ = [
    "CompromiseJumpGeometry",
    "ConfidenceGeometry",
    "ConfidenceMode",
    "compromise_jump_geometry",
    "confidence_geometry",
]

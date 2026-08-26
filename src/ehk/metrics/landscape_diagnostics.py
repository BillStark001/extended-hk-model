"""Quantitative diagnostics for time-dependent effective potentials."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from ehk.common.numerics import estimate_potential_from_force

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class LandscapeSnapshot:
    """The dominant two-well structure of one potential curve."""

    double_well: bool
    barrier_height: float
    left_depth: float
    right_depth: float
    left_position: float
    right_position: float
    barrier_position: float
    well_separation: float
    left_curvature: float
    right_curvature: float
    barrier_curvature: float


@dataclass(frozen=True)
class LandscapeSeries:
    """Snapshot diagnostics and persistent first-passage times."""

    time: FloatArray
    double_well: BoolArray
    barrier_height: FloatArray
    left_depth: FloatArray
    right_depth: FloatArray
    left_position: FloatArray
    right_position: FloatArray
    barrier_position: FloatArray
    well_separation: FloatArray
    left_curvature: FloatArray
    right_curvature: FloatArray
    barrier_curvature: FloatArray
    formation_time: float
    barrier_thresholds: FloatArray
    barrier_crossing_times: FloatArray
    half_final_barrier_time: float


def potential_from_force(
    x: FloatArray,
    force: FloatArray,
    *,
    center: bool = True,
) -> FloatArray:
    """Integrate one or more force curves along their final axis."""

    x_values = np.asarray(x, dtype=float)
    force_values = np.asarray(force, dtype=float)
    if x_values.ndim != 1 or force_values.shape[-1] != x_values.size:
        raise ValueError("force curve's final axis must match one-dimensional x")
    flat_force = force_values.reshape((-1, x_values.size))
    flat_potential = np.stack(
        [estimate_potential_from_force(x_values, row) for row in flat_force]
    )
    potential = flat_potential.reshape(force_values.shape)
    if center:
        potential -= np.mean(potential, axis=-1, keepdims=True)
    return np.asarray(potential, dtype=float)


def _quadratic_curvature(
    x: FloatArray,
    potential: FloatArray,
    index: int,
    radius: int,
) -> float:
    left = max(index - radius, 0)
    right = min(index + radius + 1, x.size)
    if right - left < 3:
        return float("nan")
    local_x = x[left:right] - x[index]
    coefficient = np.polyfit(local_x, potential[left:right], deg=2)[0]
    return float(2.0 * coefficient)


def _minimum_indices(potential: FloatArray, prominence: float) -> NDArray[np.int64]:
    minima, _ = find_peaks(-potential, prominence=prominence)
    candidates = [int(index) for index in minima]
    if potential[0] + prominence < potential[1]:
        candidates.insert(0, 0)
    if potential[-1] + prominence < potential[-2]:
        candidates.append(potential.size - 1)
    return np.asarray(sorted(set(candidates)), dtype=np.int64)


def quantify_landscape(
    x: FloatArray,
    potential: FloatArray,
    *,
    min_prominence: float = 1e-8,
    relative_prominence: float = 1e-3,
    curvature_radius: int = 2,
) -> LandscapeSnapshot:
    """Extract the strongest barrier-separated pair of stable wells.

    The effective barrier is the smaller of the two well depths.  This
    conservative definition requires both basins to be separated from the
    intervening maximum, and is therefore robust to a strongly tilted
    potential with one merely shoulder-like side.
    """

    x_values = np.asarray(x, dtype=float)
    values = np.asarray(potential, dtype=float)
    if x_values.ndim != 1 or values.shape != x_values.shape or x_values.size < 5:
        raise ValueError("x and potential must be equal one-dimensional arrays")
    if not np.all(np.isfinite(x_values)) or not np.all(np.isfinite(values)):
        raise ValueError("x and potential must be finite")
    if not np.all(np.diff(x_values) > 0):
        raise ValueError("x must be strictly increasing")
    if curvature_radius < 1:
        raise ValueError("curvature_radius must be positive")

    prominence = max(
        float(min_prominence), float(relative_prominence) * float(np.ptp(values))
    )
    minima = _minimum_indices(values, prominence)
    best: tuple[float, int, int, int, float, float] | None = None
    for position, left in enumerate(minima[:-1]):
        for right in minima[position + 1 :]:
            if int(right) - int(left) < 2:
                continue
            relative_barrier = int(np.argmax(values[left : right + 1]))
            barrier = int(left) + relative_barrier
            if barrier in {int(left), int(right)}:
                continue
            left_depth = float(values[barrier] - values[left])
            right_depth = float(values[barrier] - values[right])
            effective_depth = min(left_depth, right_depth)
            if effective_depth < prominence:
                continue
            candidate = (
                effective_depth,
                int(left),
                int(right),
                barrier,
                left_depth,
                right_depth,
            )
            if best is None or candidate[0] > best[0]:
                best = candidate

    if best is None:
        missing = float("nan")
        return LandscapeSnapshot(
            False,
            0.0,
            missing,
            missing,
            missing,
            missing,
            missing,
            missing,
            missing,
            missing,
            missing,
        )

    height, left, right, barrier, left_depth, right_depth = best
    return LandscapeSnapshot(
        True,
        height,
        left_depth,
        right_depth,
        float(x_values[left]),
        float(x_values[right]),
        float(x_values[barrier]),
        float(x_values[right] - x_values[left]),
        _quadratic_curvature(x_values, values, left, curvature_radius),
        _quadratic_curvature(x_values, values, right, curvature_radius),
        _quadratic_curvature(x_values, values, barrier, curvature_radius),
    )


def first_persistent_time(
    time: FloatArray,
    condition: BoolArray,
    persistence: int = 3,
) -> float:
    """Return the start of the first run lasting ``persistence`` records."""

    times = np.asarray(time, dtype=float)
    flags = np.asarray(condition, dtype=bool)
    if times.ndim != 1 or flags.shape != times.shape:
        raise ValueError("time and condition must be equal one-dimensional arrays")
    if persistence < 1:
        raise ValueError("persistence must be positive")
    if persistence > flags.size:
        return float("nan")
    counts = np.convolve(flags.astype(int), np.ones(persistence, dtype=int), mode="valid")
    found = np.flatnonzero(counts == persistence)
    return float(times[found[0]]) if found.size else float("nan")


def quantify_landscape_series(
    x: FloatArray,
    time: FloatArray,
    potential: FloatArray,
    *,
    barrier_thresholds: tuple[float, ...] = (0.01, 0.05),
    persistence: int = 3,
    min_prominence: float = 1e-8,
    relative_prominence: float = 1e-3,
    curvature_radius: int = 2,
) -> LandscapeSeries:
    """Quantify a potential time series and its persistent first passages."""

    times = np.asarray(time, dtype=float)
    values = np.asarray(potential, dtype=float)
    if times.ndim != 1 or values.ndim != 2 or values.shape[0] != times.size:
        raise ValueError("potential must have shape (time.size, x.size)")
    if values.shape[1] != np.asarray(x).size:
        raise ValueError("potential's final axis must match x")
    thresholds = np.asarray(barrier_thresholds, dtype=float)
    if np.any(thresholds < 0) or not np.all(np.isfinite(thresholds)):
        raise ValueError("barrier thresholds must be finite and non-negative")

    snapshots = [
        quantify_landscape(
            x,
            row,
            min_prominence=min_prominence,
            relative_prominence=relative_prominence,
            curvature_radius=curvature_radius,
        )
        for row in values
    ]
    double_well = np.asarray([item.double_well for item in snapshots], dtype=bool)
    barrier_height = np.asarray(
        [item.barrier_height for item in snapshots], dtype=float
    )
    crossing_times = np.asarray(
        [
            first_persistent_time(
                times,
                double_well & (barrier_height >= threshold),
                persistence,
            )
            for threshold in thresholds
        ],
        dtype=float,
    )
    final_barrier = float(barrier_height[-1])
    half_final_time = (
        first_persistent_time(
            times,
            double_well & (barrier_height >= 0.5 * final_barrier),
            persistence,
        )
        if final_barrier > 0
        else float("nan")
    )

    def field(name: str) -> FloatArray:
        return np.asarray([getattr(item, name) for item in snapshots], dtype=float)

    return LandscapeSeries(
        time=times.copy(),
        double_well=double_well,
        barrier_height=barrier_height,
        left_depth=field("left_depth"),
        right_depth=field("right_depth"),
        left_position=field("left_position"),
        right_position=field("right_position"),
        barrier_position=field("barrier_position"),
        well_separation=field("well_separation"),
        left_curvature=field("left_curvature"),
        right_curvature=field("right_curvature"),
        barrier_curvature=field("barrier_curvature"),
        formation_time=first_persistent_time(times, double_well, persistence),
        barrier_thresholds=thresholds,
        barrier_crossing_times=crossing_times,
        half_final_barrier_time=half_final_time,
    )


__all__ = [
    "LandscapeSeries",
    "LandscapeSnapshot",
    "first_persistent_time",
    "potential_from_force",
    "quantify_landscape",
    "quantify_landscape_series",
]

"""Quantitative diagnostics for time-dependent effective potentials."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linear_sum_assignment
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


@dataclass(frozen=True)
class MultiwellSnapshot:
    """All one-dimensional wells and adjacent barriers at one time."""

    well_position: FloatArray
    well_potential: FloatArray
    basin_mass: FloatArray
    basin_left: FloatArray
    basin_right: FloatArray
    barrier_position: FloatArray
    barrier_height: FloatArray
    barrier_left_depth: FloatArray
    barrier_right_depth: FloatArray
    barrier_left_mass: FloatArray
    barrier_macro_score: FloatArray
    robust_well_count: int
    robust_barrier_count: int
    effective_well_count: float
    dominant_barrier_index: int


@dataclass(frozen=True)
class MultiwellSeries:
    """Padded multibarrier spectra and tracked well identities over time."""

    time: FloatArray
    well_count: NDArray[np.int64]
    well_position: FloatArray
    well_potential: FloatArray
    basin_mass: FloatArray
    basin_left: FloatArray
    basin_right: FloatArray
    well_id: NDArray[np.int64]
    barrier_count: NDArray[np.int64]
    barrier_position: FloatArray
    barrier_height: FloatArray
    barrier_macro_score: FloatArray
    barrier_left_id: NDArray[np.int64]
    barrier_right_id: NDArray[np.int64]
    robust_well_count: NDArray[np.int64]
    robust_barrier_count: NDArray[np.int64]
    effective_well_count: FloatArray
    dominant_barrier_height: FloatArray
    dominant_macro_score: FloatArray
    dominant_barrier_position: FloatArray
    dominant_left_id: NDArray[np.int64]
    dominant_right_id: NDArray[np.int64]
    dominant_switch: BoolArray
    well_birth_count: NDArray[np.int64]
    well_death_count: NDArray[np.int64]
    peak_time: float
    peak_barrier_height: float
    final_barrier_height: float
    overshoot_absolute: float
    overshoot_relative: float
    overshoot_class: str


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
    counts = np.convolve(
        flags.astype(int), np.ones(persistence, dtype=int), mode="valid"
    )
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


def _basin_masses(
    rho: FloatArray,
    minima: NDArray[np.int64],
    saddles: NDArray[np.int64],
) -> tuple[FloatArray, NDArray[np.int64], NDArray[np.int64]]:
    """Partition cell mass at adjacent saddle cells, splitting separators."""

    masses = np.zeros(minima.size, dtype=float)
    left = np.zeros(minima.size, dtype=np.int64)
    right = np.full(minima.size, rho.size - 1, dtype=np.int64)
    for index in range(minima.size):
        if index > 0:
            left[index] = saddles[index - 1]
        if index < saddles.size:
            right[index] = saddles[index]
        masses[index] = float(np.sum(rho[left[index] : right[index] + 1]))
        if index > 0:
            masses[index] -= 0.5 * float(rho[left[index]])
        if index < saddles.size:
            masses[index] -= 0.5 * float(rho[right[index]])
    total = float(np.sum(masses))
    if total <= 0:
        raise ValueError("basin masses must have positive total mass")
    return masses / total, left, right


def quantify_multiwell(
    x: FloatArray,
    potential: FloatArray,
    rho: FloatArray,
    *,
    min_basin_mass: float = 0.02,
    min_barrier_height: float = 1e-6,
    min_prominence: float = 1e-8,
    relative_prominence: float = 1e-3,
) -> MultiwellSnapshot:
    """Extract all wells, adjacent barriers, and a mass-balanced macro cut."""

    x_values = np.asarray(x, dtype=float)
    values = np.asarray(potential, dtype=float)
    masses = np.asarray(rho, dtype=float)
    if (
        x_values.ndim != 1
        or values.shape != x_values.shape
        or masses.shape != x_values.shape
        or x_values.size < 5
    ):
        raise ValueError("x, potential, and rho must be equal one-dimensional arrays")
    if (
        not np.all(np.isfinite(x_values))
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(masses))
        or np.any(masses < 0)
    ):
        raise ValueError("multiwell inputs must be finite with non-negative rho")
    if not np.all(np.diff(x_values) > 0) or float(np.sum(masses)) <= 0:
        raise ValueError("x must increase and rho must have positive mass")
    if not 0 <= min_basin_mass < 0.5 or min_barrier_height < 0:
        raise ValueError("invalid basin-mass or barrier-height threshold")

    prominence = max(
        float(min_prominence), float(relative_prominence) * float(np.ptp(values))
    )
    minima = _minimum_indices(values, prominence)
    if not minima.size:
        minimum = float(np.min(values))
        tied = np.flatnonzero(np.isclose(values, minimum, rtol=0, atol=1e-15))
        minima = np.asarray([tied[np.argmin(np.abs(x_values[tied]))]], dtype=np.int64)
    saddles = np.asarray(
        [
            int(left) + int(np.argmax(values[left : right + 1]))
            for left, right in pairwise(minima)
        ],
        dtype=np.int64,
    )
    basin_mass, basin_left, basin_right = _basin_masses(
        masses / np.sum(masses), minima, saddles
    )

    left_depth = values[saddles] - values[minima[:-1]]
    right_depth = values[saddles] - values[minima[1:]]
    barrier_height = np.maximum(np.minimum(left_depth, right_depth), 0.0)
    left_mass = np.cumsum(basin_mass)[:-1]
    macro_score = 4.0 * left_mass * (1.0 - left_mass) * barrier_height
    valid_barrier = (
        (left_mass >= min_basin_mass)
        & ((1.0 - left_mass) >= min_basin_mass)
        & (barrier_height >= min_barrier_height)
    )
    if np.any(valid_barrier):
        candidates = np.flatnonzero(valid_barrier)
        dominant = int(candidates[np.argmax(macro_score[candidates])])
    else:
        dominant = -1
    positive_mass = basin_mass[basin_mass > 0]
    effective_count = float(np.exp(-np.sum(positive_mass * np.log(positive_mass))))
    return MultiwellSnapshot(
        well_position=x_values[minima],
        well_potential=values[minima],
        basin_mass=basin_mass,
        basin_left=x_values[basin_left],
        basin_right=x_values[basin_right],
        barrier_position=x_values[saddles],
        barrier_height=np.asarray(barrier_height, dtype=float),
        barrier_left_depth=np.asarray(left_depth, dtype=float),
        barrier_right_depth=np.asarray(right_depth, dtype=float),
        barrier_left_mass=np.asarray(left_mass, dtype=float),
        barrier_macro_score=np.asarray(macro_score, dtype=float),
        robust_well_count=int(np.sum(basin_mass >= min_basin_mass)),
        robust_barrier_count=int(np.sum(valid_barrier)),
        effective_well_count=effective_count,
        dominant_barrier_index=dominant,
    )


def _track_well_ids(
    snapshots: list[MultiwellSnapshot],
    max_well_displacement: float,
) -> tuple[list[NDArray[np.int64]], NDArray[np.int64], NDArray[np.int64]]:
    ids: list[NDArray[np.int64]] = []
    births = np.zeros(len(snapshots), dtype=np.int64)
    deaths = np.zeros(len(snapshots), dtype=np.int64)
    next_id = 0
    for time_index, current in enumerate(snapshots):
        current_ids = np.full(current.well_position.size, -1, dtype=np.int64)
        if time_index == 0:
            current_ids[:] = np.arange(next_id, next_id + current_ids.size)
            births[time_index] = current_ids.size
            next_id += current_ids.size
            ids.append(current_ids)
            continue

        previous = snapshots[time_index - 1]
        previous_ids = ids[-1]
        matched_previous: set[int] = set()
        matched_current: set[int] = set()
        if previous_ids.size and current_ids.size:
            distance = np.abs(
                previous.well_position[:, None] - current.well_position[None, :]
            )
            overlap = np.maximum(
                0.0,
                np.minimum(previous.basin_right[:, None], current.basin_right[None, :])
                - np.maximum(previous.basin_left[:, None], current.basin_left[None, :]),
            )
            union = np.maximum(
                previous.basin_right[:, None], current.basin_right[None, :]
            ) - np.minimum(previous.basin_left[:, None], current.basin_left[None, :])
            overlap_fraction = np.divide(
                overlap,
                union,
                out=np.zeros_like(overlap),
                where=union > 0,
            )
            cost = distance / max_well_displacement + 1.0 - overlap_fraction
            previous_match, current_match = linear_sum_assignment(cost)
            for old, new in zip(previous_match, current_match, strict=True):
                if distance[old, new] <= max_well_displacement:
                    current_ids[new] = previous_ids[old]
                    matched_previous.add(int(old))
                    matched_current.add(int(new))
        for current_index in range(current_ids.size):
            if current_index not in matched_current:
                current_ids[current_index] = next_id
                next_id += 1
                births[time_index] += 1
        deaths[time_index] = previous_ids.size - len(matched_previous)
        ids.append(current_ids)
    return ids, births, deaths


def _track_dominant_barriers(
    snapshots: list[MultiwellSnapshot],
    tracked_ids: list[NDArray[np.int64]],
    *,
    min_basin_mass: float,
    min_barrier_height: float,
    score_margin: float,
    switch_persistence: int,
) -> tuple[NDArray[np.int64], BoolArray]:
    """Track one dominant barrier without switching between near-tied pairs.

    The instantaneous macro-score maximum is only a challenger. While the
    incumbent pair remains robust, a challenger must lead it by ``score_margin``
    for ``switch_persistence`` consecutive records before its identity is
    accepted. Exact and near ties therefore retain the incumbent. If the
    incumbent barrier ceases to be robust, the best remaining pair takes over
    immediately; a record with no robust barrier resets the tracker.
    """

    dominant = np.full(len(snapshots), -1, dtype=np.int64)
    switches = np.zeros(len(snapshots), dtype=bool)
    incumbent_pair: tuple[int, int] | None = None
    pending_pair: tuple[int, int] | None = None
    pending_count = 0

    for time_index, (snapshot, ids_at_time) in enumerate(
        zip(snapshots, tracked_ids, strict=True)
    ):
        left_mass = snapshot.barrier_left_mass
        valid = np.flatnonzero(
            (left_mass >= min_basin_mass)
            & ((1.0 - left_mass) >= min_basin_mass)
            & (snapshot.barrier_height >= min_barrier_height)
        )
        if valid.size == 0:
            incumbent_pair = None
            pending_pair = None
            pending_count = 0
            continue

        pairs = {
            (int(ids_at_time[index]), int(ids_at_time[index + 1])): int(index)
            for index in valid
        }
        challenger_index = int(valid[np.argmax(snapshot.barrier_macro_score[valid])])
        challenger_pair = (
            int(ids_at_time[challenger_index]),
            int(ids_at_time[challenger_index + 1]),
        )

        if incumbent_pair is None:
            incumbent_pair = challenger_pair
        elif incumbent_pair not in pairs:
            switches[time_index] = challenger_pair != incumbent_pair
            incumbent_pair = challenger_pair
            pending_pair = None
            pending_count = 0
        else:
            incumbent_index = pairs[incumbent_pair]
            incumbent_score = float(snapshot.barrier_macro_score[incumbent_index])
            challenger_score = float(snapshot.barrier_macro_score[challenger_index])
            scale = max(
                abs(incumbent_score),
                abs(challenger_score),
                np.finfo(float).tiny,
            )
            lead = (challenger_score - incumbent_score) / scale
            if challenger_pair == incumbent_pair or lead <= score_margin:
                pending_pair = None
                pending_count = 0
            else:
                if challenger_pair == pending_pair:
                    pending_count += 1
                else:
                    pending_pair = challenger_pair
                    pending_count = 1
                if pending_count >= switch_persistence:
                    incumbent_pair = challenger_pair
                    switches[time_index] = True
                    pending_pair = None
                    pending_count = 0

        dominant[time_index] = pairs[incumbent_pair]

    return dominant, switches


def quantify_multiwell_series(
    x: FloatArray,
    time: FloatArray,
    potential: FloatArray,
    rho: FloatArray,
    *,
    min_basin_mass: float = 0.02,
    min_barrier_height: float = 1e-6,
    min_prominence: float = 1e-8,
    relative_prominence: float = 1e-3,
    max_well_displacement: float = 0.2,
    dominant_score_margin: float = 0.05,
    dominant_switch_persistence: int = 3,
    overshoot_tolerance: float = 1e-4,
) -> MultiwellSeries:
    """Quantify, pad, and track a time-dependent multibarrier landscape.

    Dominant-barrier identities use a tie-aware hysteresis tracker. A robust
    challenger must exceed the incumbent macro score by
    ``dominant_score_margin`` for ``dominant_switch_persistence`` consecutive
    records. This prevents symmetric barriers from generating label chatter.
    """

    times = np.asarray(time, dtype=float)
    values = np.asarray(potential, dtype=float)
    densities = np.asarray(rho, dtype=float)
    if (
        times.ndim != 1
        or values.shape != (times.size, np.asarray(x).size)
        or densities.shape != values.shape
    ):
        raise ValueError("potential and rho must have shape (time.size, x.size)")
    if max_well_displacement <= 0:
        raise ValueError("tracking displacement must be positive")
    if not np.isfinite(dominant_score_margin) or not 0 <= dominant_score_margin < 1:
        raise ValueError("dominant score margin must lie in [0, 1)")
    if dominant_switch_persistence < 1:
        raise ValueError("dominant switch persistence must be positive")
    if overshoot_tolerance < 0:
        raise ValueError("overshoot tolerance must be non-negative")
    snapshots = [
        quantify_multiwell(
            x,
            potential_row,
            rho_row,
            min_basin_mass=min_basin_mass,
            min_barrier_height=min_barrier_height,
            min_prominence=min_prominence,
            relative_prominence=relative_prominence,
        )
        for potential_row, rho_row in zip(values, densities, strict=True)
    ]
    tracked_ids, births, deaths = _track_well_ids(snapshots, max_well_displacement)
    dominant_indices, switches = _track_dominant_barriers(
        snapshots,
        tracked_ids,
        min_basin_mass=min_basin_mass,
        min_barrier_height=min_barrier_height,
        score_margin=dominant_score_margin,
        switch_persistence=dominant_switch_persistence,
    )
    well_width = max(snapshot.well_position.size for snapshot in snapshots)
    barrier_width = max(
        1, max(snapshot.barrier_position.size for snapshot in snapshots)
    )
    time_size = times.size

    def padded_float(width: int) -> FloatArray:
        return np.full((time_size, width), np.nan, dtype=float)

    well_position = padded_float(well_width)
    well_potential = padded_float(well_width)
    basin_mass = padded_float(well_width)
    basin_left = padded_float(well_width)
    basin_right = padded_float(well_width)
    well_id = np.full((time_size, well_width), -1, dtype=np.int64)
    barrier_position = padded_float(barrier_width)
    barrier_height = padded_float(barrier_width)
    barrier_macro_score = padded_float(barrier_width)
    barrier_left_id = np.full((time_size, barrier_width), -1, dtype=np.int64)
    barrier_right_id = np.full((time_size, barrier_width), -1, dtype=np.int64)
    dominant_height = np.zeros(time_size, dtype=float)
    dominant_score = np.zeros(time_size, dtype=float)
    dominant_position = np.full(time_size, np.nan, dtype=float)
    dominant_left_id = np.full(time_size, -1, dtype=np.int64)
    dominant_right_id = np.full(time_size, -1, dtype=np.int64)
    for time_index, (snapshot, ids_at_time) in enumerate(
        zip(snapshots, tracked_ids, strict=True)
    ):
        wells = snapshot.well_position.size
        barriers = snapshot.barrier_position.size
        well_position[time_index, :wells] = snapshot.well_position
        well_potential[time_index, :wells] = snapshot.well_potential
        basin_mass[time_index, :wells] = snapshot.basin_mass
        basin_left[time_index, :wells] = snapshot.basin_left
        basin_right[time_index, :wells] = snapshot.basin_right
        well_id[time_index, :wells] = ids_at_time
        barrier_position[time_index, :barriers] = snapshot.barrier_position
        barrier_height[time_index, :barriers] = snapshot.barrier_height
        barrier_macro_score[time_index, :barriers] = snapshot.barrier_macro_score
        if barriers:
            barrier_left_id[time_index, :barriers] = ids_at_time[:-1]
            barrier_right_id[time_index, :barriers] = ids_at_time[1:]
        dominant = int(dominant_indices[time_index])
        if dominant >= 0:
            dominant_height[time_index] = snapshot.barrier_height[dominant]
            dominant_score[time_index] = snapshot.barrier_macro_score[dominant]
            dominant_position[time_index] = snapshot.barrier_position[dominant]
            pair = (int(ids_at_time[dominant]), int(ids_at_time[dominant + 1]))
            dominant_left_id[time_index], dominant_right_id[time_index] = pair

    peak_index = int(np.argmax(dominant_height))
    peak_height = float(dominant_height[peak_index])
    final_height = float(dominant_height[-1])
    overshoot = max(peak_height - final_height, 0.0)
    relative_overshoot = (
        overshoot / peak_height if peak_height >= min_barrier_height else float("nan")
    )
    peak_pair = (int(dominant_left_id[peak_index]), int(dominant_right_id[peak_index]))
    final_pair = (int(dominant_left_id[-1]), int(dominant_right_id[-1]))
    if peak_height < min_barrier_height:
        overshoot_class = "no_robust_barrier"
    elif final_pair == (-1, -1):
        overshoot_class = "barrier_annihilation"
    elif final_pair != peak_pair or np.any(switches[peak_index + 1 :]):
        overshoot_class = "dominant_pair_switch"
    elif overshoot > overshoot_tolerance:
        overshoot_class = "same_pair_relaxation"
    else:
        overshoot_class = "no_overshoot"

    return MultiwellSeries(
        time=times.copy(),
        well_count=np.asarray(
            [snapshot.well_position.size for snapshot in snapshots], dtype=np.int64
        ),
        well_position=well_position,
        well_potential=well_potential,
        basin_mass=basin_mass,
        basin_left=basin_left,
        basin_right=basin_right,
        well_id=well_id,
        barrier_count=np.asarray(
            [snapshot.barrier_position.size for snapshot in snapshots],
            dtype=np.int64,
        ),
        barrier_position=barrier_position,
        barrier_height=barrier_height,
        barrier_macro_score=barrier_macro_score,
        barrier_left_id=barrier_left_id,
        barrier_right_id=barrier_right_id,
        robust_well_count=np.asarray(
            [snapshot.robust_well_count for snapshot in snapshots], dtype=np.int64
        ),
        robust_barrier_count=np.asarray(
            [snapshot.robust_barrier_count for snapshot in snapshots], dtype=np.int64
        ),
        effective_well_count=np.asarray(
            [snapshot.effective_well_count for snapshot in snapshots], dtype=float
        ),
        dominant_barrier_height=dominant_height,
        dominant_macro_score=dominant_score,
        dominant_barrier_position=dominant_position,
        dominant_left_id=dominant_left_id,
        dominant_right_id=dominant_right_id,
        dominant_switch=switches,
        well_birth_count=births,
        well_death_count=deaths,
        peak_time=float(times[peak_index]),
        peak_barrier_height=peak_height,
        final_barrier_height=final_height,
        overshoot_absolute=overshoot,
        overshoot_relative=relative_overshoot,
        overshoot_class=overshoot_class,
    )


__all__ = [
    "LandscapeSeries",
    "LandscapeSnapshot",
    "MultiwellSeries",
    "MultiwellSnapshot",
    "first_persistent_time",
    "potential_from_force",
    "quantify_landscape",
    "quantify_landscape_series",
    "quantify_multiwell",
    "quantify_multiwell_series",
]

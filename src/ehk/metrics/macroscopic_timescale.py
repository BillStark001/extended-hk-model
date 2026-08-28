"""Operator-resolved macro-level time-scale diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ehk.metrics.density_indices import (
    DensityIndexCalculator,
    IndexSeries,
    calculate_index_series,
)
from ehk.metrics.homophily import uniform_concordance_probability
from ehk.modeling.mesoscopic import (
    KineticTrajectory,
    advance_frozen_opinion_density,
)
from ehk.modeling.mesoscopic.solver import FloatArray


@dataclass(frozen=True)
class ChannelContributionSeries:
    """One-step channel responses and their early integrated ratio."""

    time: FloatArray
    opinion_polarization_rate: FloatArray
    rewiring_homophily_rate: FloatArray
    gamma: FloatArray
    progress_threshold: float
    window_end: float
    opinion_drive: float
    rewiring_drive: float
    gamma_integrated: float
    opinion_rate_at_window: float
    rewiring_rate_at_window: float
    gamma_at_window: float
    probe_dt: float


@dataclass(frozen=True)
class ChannelWindowSummary:
    """Integrated positive channel responses up to a state-space threshold."""

    progress_threshold: float
    window_end: float
    opinion_drive: float
    rewiring_drive: float
    gamma_integrated: float
    opinion_rate_at_window: float
    rewiring_rate_at_window: float
    gamma_at_window: float


@dataclass(frozen=True)
class ChannelProgressSnapshot:
    """Operator-resolved rates at one interpolated macro-progress level.

    Progress is ``u = I_p + I_h``.  The two channel rates are evaluated only
    at the recorded states bracketing ``u`` and linearly interpolated in
    progress, so callers need not probe every state in a long trajectory.
    """

    progress_threshold: float
    time: float
    opinion_polarization_rate: float
    rewiring_homophily_rate: float
    gamma: float
    reached: bool
    lower_record: int
    upper_record: int
    interpolation_fraction: float


def _ratio(numerator: float, denominator: float, floor: float) -> float:
    if denominator > floor:
        return numerator / denominator
    return float("inf") if numerator > floor else float("nan")


def _channel_rates_at_records(
    trajectory: KineticTrajectory,
    indices: IndexSeries,
    records: np.ndarray,
    *,
    probe_dt: float,
    ratio_floor: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Evaluate frozen operator responses at selected trajectory records."""

    params = trajectory.parameters
    calculator = DensityIndexCalculator(
        trajectory.x,
        params.epsilon,
        params.mean_degree,
        params.confidence_mode,
    )
    opinion_rate = np.empty(records.size, dtype=float)
    for output_index, record in enumerate(records):
        structural_score = (
            trajectory.structural_score[record]
            if trajectory.structural_score is not None
            else None
        )
        counterfactual = advance_frozen_opinion_density(
            params,
            trajectory.x,
            trajectory.rho[record],
            trajectory.edge[record],
            dt=probe_dt,
            structural_score=structural_score,
        )
        opinion_rate[output_index] = (
            calculator.polarization(counterfactual)
            - indices.polarization[record]
        ) / probe_dt

    baseline = uniform_concordance_probability(params.epsilon)
    if baseline >= 1.0:
        rewiring_rate = np.zeros(records.size, dtype=float)
    else:
        rewiring_rate = np.einsum(
            "tij,ij->t",
            trajectory.rewiring_flux[records],
            calculator.concordant,
            optimize=True,
        ) / (params.mean_degree * (1.0 - baseline))
    gamma = np.asarray(
        [
            _ratio(max(float(rewiring), 0.0), max(float(opinion), 0.0), ratio_floor)
            for opinion, rewiring in zip(opinion_rate, rewiring_rate, strict=True)
        ],
        dtype=float,
    )
    return opinion_rate, rewiring_rate, gamma


def calculate_channel_progress_snapshots(
    trajectory: KineticTrajectory,
    indices: IndexSeries | None = None,
    *,
    progress_thresholds: tuple[float, ...] = (0.0, 0.1),
    probe_dt: float = 1.0,
    ratio_floor: float = 1e-12,
) -> tuple[ChannelProgressSnapshot, ...]:
    """Return channel-rate ratios at requested ``I_p + I_h`` levels.

    A threshold not reached within the simulated horizon is evaluated at the
    final record and marked ``reached=False``.  This keeps the numerical value
    auditable without silently treating a censored trajectory as a crossing.
    """

    if probe_dt <= 0 or ratio_floor <= 0:
        raise ValueError("probe_dt and ratio_floor must be positive")
    if not progress_thresholds:
        raise ValueError("at least one progress threshold is required")
    thresholds = np.asarray(progress_thresholds, dtype=float)
    if np.any(~np.isfinite(thresholds)) or np.any(thresholds < 0):
        raise ValueError("progress thresholds must be finite and non-negative")
    if indices is None:
        indices = calculate_index_series(trajectory)
    if not np.array_equal(indices.time, trajectory.time):
        raise ValueError("index and trajectory recording times differ")

    progress = indices.polarization + indices.homophily
    brackets: list[tuple[int, int, float, bool]] = []
    required_records: set[int] = set()
    for threshold in thresholds:
        found = np.flatnonzero(progress >= threshold)
        if not found.size:
            lower = upper = progress.size - 1
            fraction = 0.0
            reached = False
        else:
            upper = int(found[0])
            if upper == 0:
                lower = 0
                fraction = 0.0
            else:
                lower = upper - 1
                before = float(progress[lower])
                after = float(progress[upper])
                fraction = (
                    (float(threshold) - before) / (after - before)
                    if after > before
                    else 1.0
                )
                fraction = float(np.clip(fraction, 0.0, 1.0))
            reached = True
        brackets.append((lower, upper, fraction, reached))
        required_records.update((lower, upper))

    records = np.asarray(sorted(required_records), dtype=int)
    opinion_rate, rewiring_rate, _ = _channel_rates_at_records(
        trajectory,
        indices,
        records,
        probe_dt=probe_dt,
        ratio_floor=ratio_floor,
    )
    record_lookup = {int(record): index for index, record in enumerate(records)}
    snapshots = []
    for threshold, (lower, upper, fraction, reached) in zip(
        thresholds, brackets, strict=True
    ):
        lower_index = record_lookup[lower]
        upper_index = record_lookup[upper]
        time = float(
            trajectory.time[lower]
            + fraction * (trajectory.time[upper] - trajectory.time[lower])
        )
        opinion = float(
            opinion_rate[lower_index]
            + fraction * (opinion_rate[upper_index] - opinion_rate[lower_index])
        )
        rewiring = float(
            rewiring_rate[lower_index]
            + fraction * (rewiring_rate[upper_index] - rewiring_rate[lower_index])
        )
        snapshots.append(
            ChannelProgressSnapshot(
                progress_threshold=float(threshold),
                time=time,
                opinion_polarization_rate=opinion,
                rewiring_homophily_rate=rewiring,
                gamma=_ratio(
                    max(rewiring, 0.0), max(opinion, 0.0), ratio_floor
                ),
                reached=reached,
                lower_record=lower,
                upper_record=upper,
                interpolation_fraction=fraction,
            )
        )
    return tuple(snapshots)


def _integrate_to_progress(
    time: FloatArray,
    progress: FloatArray,
    first_rate: FloatArray,
    second_rate: FloatArray,
    threshold: float,
) -> tuple[float, float, float, float, float]:
    """Integrate two rates until the first linearly interpolated crossing."""

    found = np.flatnonzero(progress >= threshold)
    if found.size and found[0] == 0:
        return (
            0.0,
            0.0,
            float(time[0]),
            float(first_rate[0]),
            float(second_rate[0]),
        )
    if found.size:
        stop = int(found[0])
        before = float(progress[stop - 1])
        after = float(progress[stop])
        fraction = (
            (threshold - before) / (after - before)
            if after > before
            else 1.0
        )
        fraction = float(np.clip(fraction, 0.0, 1.0))
        boundary_time = time[stop - 1] + fraction * (
            time[stop] - time[stop - 1]
        )
        selected_time = np.concatenate((time[:stop], [boundary_time]))
        selected_first = np.concatenate(
            (
                first_rate[:stop],
                [
                    first_rate[stop - 1]
                    + fraction * (first_rate[stop] - first_rate[stop - 1])
                ],
            )
        )
        selected_second = np.concatenate(
            (
                second_rate[:stop],
                [
                    second_rate[stop - 1]
                    + fraction * (second_rate[stop] - second_rate[stop - 1])
                ],
            )
        )
    else:
        selected_time = time
        selected_first = first_rate
        selected_second = second_rate
        boundary_time = float(time[-1])
    first_integral = float(
        np.trapezoid(np.maximum(selected_first, 0.0), selected_time)
    )
    second_integral = float(
        np.trapezoid(np.maximum(selected_second, 0.0), selected_time)
    )
    return (
        first_integral,
        second_integral,
        float(boundary_time),
        float(selected_first[-1]),
        float(selected_second[-1]),
    )


def calculate_channel_contributions(
    trajectory: KineticTrajectory,
    indices: IndexSeries | None = None,
    *,
    probe_dt: float = 1.0,
    progress_threshold: float = 0.25,
    ratio_floor: float = 1e-12,
) -> ChannelContributionSeries:
    """Separate opinion-to-polarization and rewiring-to-homophily responses.

    At each recorded state the opinion channel is advanced for ``probe_dt``
    with its frozen velocity and zero diffusion, while rewiring's homophily
    response is evaluated exactly from the stored conservative edge flux.
    The scalar ``gamma_integrated`` is the ratio of positive channel responses
    accumulated until ``I_p + I_h`` first reaches ``progress_threshold``.
    """

    if probe_dt <= 0 or not 0 < progress_threshold < 1:
        raise ValueError("probe_dt must be positive and threshold must lie in (0, 1)")
    if ratio_floor <= 0:
        raise ValueError("ratio_floor must be positive")
    if indices is None:
        indices = calculate_index_series(trajectory)
    if not np.array_equal(indices.time, trajectory.time):
        raise ValueError("index and trajectory recording times differ")

    records = np.arange(trajectory.time.size, dtype=int)
    opinion_rate, rewiring_rate, gamma = _channel_rates_at_records(
        trajectory,
        indices,
        records,
        probe_dt=probe_dt,
        ratio_floor=ratio_floor,
    )
    window = summarize_channel_window(
        trajectory.time,
        indices.polarization,
        indices.homophily,
        opinion_rate,
        rewiring_rate,
        progress_threshold=progress_threshold,
        ratio_floor=ratio_floor,
    )
    return ChannelContributionSeries(
        time=trajectory.time.copy(),
        opinion_polarization_rate=opinion_rate,
        rewiring_homophily_rate=rewiring_rate,
        gamma=gamma,
        progress_threshold=progress_threshold,
        window_end=window.window_end,
        opinion_drive=window.opinion_drive,
        rewiring_drive=window.rewiring_drive,
        gamma_integrated=window.gamma_integrated,
        opinion_rate_at_window=window.opinion_rate_at_window,
        rewiring_rate_at_window=window.rewiring_rate_at_window,
        gamma_at_window=window.gamma_at_window,
        probe_dt=probe_dt,
    )


def summarize_channel_window(
    time: FloatArray,
    polarization: FloatArray,
    homophily: FloatArray,
    opinion_polarization_rate: FloatArray,
    rewiring_homophily_rate: FloatArray,
    *,
    progress_threshold: float,
    ratio_floor: float = 1e-12,
) -> ChannelWindowSummary:
    """Re-summarize one channel series at a different pre-ordering window."""

    arrays = tuple(
        np.asarray(values, dtype=float)
        for values in (
            time,
            polarization,
            homophily,
            opinion_polarization_rate,
            rewiring_homophily_rate,
        )
    )
    if any(values.ndim != 1 or values.shape != arrays[0].shape for values in arrays):
        raise ValueError("all channel-window inputs must be equal one-dimensional arrays")
    if not 0 < progress_threshold < 1 or ratio_floor <= 0:
        raise ValueError("invalid progress threshold or ratio floor")
    (
        opinion_drive,
        rewiring_drive,
        window_end,
        opinion_rate_at_window,
        rewiring_rate_at_window,
    ) = _integrate_to_progress(
        arrays[0],
        arrays[1] + arrays[2],
        arrays[3],
        arrays[4],
        progress_threshold,
    )
    return ChannelWindowSummary(
        progress_threshold=progress_threshold,
        window_end=window_end,
        opinion_drive=opinion_drive,
        rewiring_drive=rewiring_drive,
        gamma_integrated=_ratio(rewiring_drive, opinion_drive, ratio_floor),
        opinion_rate_at_window=opinion_rate_at_window,
        rewiring_rate_at_window=rewiring_rate_at_window,
        gamma_at_window=_ratio(
            max(rewiring_rate_at_window, 0.0),
            max(opinion_rate_at_window, 0.0),
            ratio_floor,
        ),
    )


__all__ = [
    "ChannelContributionSeries",
    "ChannelProgressSnapshot",
    "ChannelWindowSummary",
    "calculate_channel_contributions",
    "calculate_channel_progress_snapshots",
    "summarize_channel_window",
]

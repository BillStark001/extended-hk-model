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
    advance_opinion_density,
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


def _ratio(numerator: float, denominator: float, floor: float) -> float:
    if denominator > floor:
        return numerator / denominator
    return float("inf") if numerator > floor else float("nan")


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

    params = trajectory.parameters
    calculator = DensityIndexCalculator(
        trajectory.x,
        params.epsilon,
        params.mean_degree,
        params.confidence_mode,
    )
    dx = float(trajectory.x[1] - trajectory.x[0])
    opinion_rate = np.empty(trajectory.time.size, dtype=float)
    for index, (rho, velocity) in enumerate(
        zip(trajectory.rho, trajectory.velocity, strict=True)
    ):
        counterfactual = advance_opinion_density(
            rho,
            velocity,
            dx=dx,
            dt=probe_dt,
            diffusion=0.0,
        )
        opinion_rate[index] = (
            calculator.polarization(counterfactual) - indices.polarization[index]
        ) / probe_dt

    baseline = uniform_concordance_probability(params.epsilon)
    if baseline >= 1.0:
        rewiring_rate = np.zeros(trajectory.time.size, dtype=float)
    else:
        rewiring_rate = np.einsum(
            "tij,ij->t",
            trajectory.rewiring_flux,
            calculator.concordant,
            optimize=True,
        ) / (params.mean_degree * (1.0 - baseline))
    positive_opinion = np.maximum(opinion_rate, 0.0)
    positive_rewiring = np.maximum(rewiring_rate, 0.0)
    gamma = np.asarray(
        [
            _ratio(float(rewiring), float(opinion), ratio_floor)
            for opinion, rewiring in zip(
                positive_opinion, positive_rewiring, strict=True
            )
        ],
        dtype=float,
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
    "ChannelWindowSummary",
    "calculate_channel_contributions",
    "summarize_channel_window",
]

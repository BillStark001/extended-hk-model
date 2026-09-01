"""Operator-resolved initial channel-rate diagnostic.

The opinion response is measured by a separate one-step Go ``measure`` run
with rewiring and background diffusion disabled. The rewiring response is the
concordant mass of the Go solver's signed rewiring source at the same initial
state. This module computes a metric; it does not advance a mesoscopic state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ehk.metrics.density_indices import DensityIndexCalculator, calculate_index_series
from ehk.metrics.homophily import uniform_concordance_probability
from ehk.modeling.mesoscopic.go_kinetic import KineticTrajectory


@dataclass(frozen=True)
class ChannelProgressSnapshot:
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


def calculate_initial_channel_snapshot(
    trajectory: KineticTrajectory,
    opinion_only_probe: KineticTrajectory,
    *,
    ratio_floor: float = 1e-12,
) -> ChannelProgressSnapshot:
    """Return ``Gamma(0)`` from two Go measure trajectories.

    ``trajectory`` supplies the initial rewiring source at the requested
    rewiring rate. ``opinion_only_probe`` starts from the same uniform state
    and contains one step with ``q=D0=0``.
    """

    if ratio_floor <= 0:
        raise ValueError("ratio_floor must be positive")
    if trajectory.time.size < 1 or opinion_only_probe.time.size < 2:
        raise ValueError("initial trajectory and two-point opinion probe required")
    if not np.allclose(trajectory.rho[0], opinion_only_probe.rho[0], atol=1e-12):
        raise ValueError("channel probes do not start from the same node state")

    probe_indices = calculate_index_series(opinion_only_probe)
    probe_dt = float(opinion_only_probe.time[1] - opinion_only_probe.time[0])
    opinion_rate = max(
        float(probe_indices.polarization[1] - probe_indices.polarization[0])
        / probe_dt,
        0.0,
    )

    params = trajectory.parameters
    calculator = DensityIndexCalculator(
        trajectory.x, params.epsilon, params.mean_degree, params.confidence_mode
    )
    baseline = uniform_concordance_probability(params.epsilon)
    rewiring_rate = max(
        float(np.sum(trajectory.rewiring_flux[0] * calculator.concordant))
        / (params.mean_degree * max(1.0 - baseline, 1e-15)),
        0.0,
    )
    return ChannelProgressSnapshot(
        progress_threshold=0.0,
        time=0.0,
        opinion_polarization_rate=opinion_rate,
        rewiring_homophily_rate=rewiring_rate,
        gamma=_ratio(rewiring_rate, opinion_rate, ratio_floor),
        reached=True,
        lower_record=0,
        upper_record=0,
        interpolation_fraction=0.0,
    )


__all__ = ["ChannelProgressSnapshot", "calculate_initial_channel_snapshot"]

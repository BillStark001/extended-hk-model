"""Density-based counterparts of the full model's pathway indices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from ehk.metrics.homophily import normalize_homophily
from ehk.modeling.mesoscopic.go_kinetic import FloatArray, KineticTrajectory
from ehk.modeling.opinion_cells import ConfidenceMode, confidence_geometry


@dataclass
class IndexSeries:
    time: FloatArray
    polarization: FloatArray
    subjective: FloatArray
    homophily: FloatArray
    homophily_raw: FloatArray
    pathway: float


def _fast_trapz(values: FloatArray, axis: FloatArray) -> float:
    return float(np.trapezoid(values, axis))


def _js_distance(p: FloatArray, q: FloatArray, axis: FloatArray) -> float:
    p_safe = np.maximum(p, 1e-13)
    q_safe = np.maximum(q, 1e-13)
    middle = 0.5 * (p_safe + q_safe)
    integrand = 0.5 * (
        p_safe * (np.log(p_safe) - np.log(middle))
        + q_safe * (np.log(q_safe) - np.log(middle))
    )
    return float(np.sqrt(max(_fast_trapz(integrand, axis), 0.0)))


def _distance_mass(
    pair_mass: FloatArray,
    distance_bins: NDArray[np.int64] | None = None,
) -> FloatArray:
    size = pair_mass.shape[0]
    if distance_bins is None:
        bins = np.arange(size)
        distance_bins = np.abs(bins[:, None] - bins[None, :]).ravel()
    result = np.bincount(
        distance_bins,
        weights=pair_mass.ravel(),
        minlength=size,
    ).astype(float)
    total = result.sum()
    return result / total if total > 0 else result


def _bandwidth(
    distances: FloatArray,
    weights: FloatArray,
    effective_samples: int,
    minimum: float,
) -> float:
    mean = float(np.sum(distances * weights))
    variance = float(np.sum((distances - mean) ** 2 * weights))
    std = np.sqrt(max(variance, 0.0))
    return max(minimum, std * effective_samples ** (-0.2))


def _mixture_pdf(
    axis: FloatArray,
    locations: FloatArray,
    weights: FloatArray,
    bandwidth: float,
) -> FloatArray:
    values = norm.pdf(
        axis[:, None],
        loc=locations[None, :],
        scale=bandwidth,
    ) @ weights
    return np.asarray(values, dtype=float)


class DensityIndexCalculator:
    def __init__(
        self,
        x: FloatArray,
        epsilon: float,
        mean_degree: float,
        confidence_mode: ConfidenceMode = "cell_average",
    ):
        self.x = x
        self.epsilon = epsilon
        self.mean_degree = mean_degree
        self.minimum_bandwidth = 0.01
        # Match the microscopic DistanceCalculator: four minimum-bandwidths
        # beyond each physical endpoint retain the boundary KDE tails.
        error_range = 4 * self.minimum_bandwidth
        self.distance_axis = np.linspace(-error_range, 2 + error_range, 256)
        self.distances = np.arange(x.size, dtype=float) * (x[1] - x[0])
        bins = np.arange(x.size)
        self.distance_bins = np.abs(
            bins[:, None] - bins[None, :]
        ).ravel()

        uniform = np.full(x.size, 1 / x.size)
        random_mass = _distance_mass(
            np.outer(uniform, uniform), self.distance_bins
        )
        random_bw = _bandwidth(
            self.distances,
            random_mass,
            effective_samples=10_000,
            minimum=self.minimum_bandwidth,
        )
        self.random_pdf = _mixture_pdf(
            self.distance_axis,
            self.distances,
            random_mass,
            random_bw,
        )
        self.subjective_worst = norm.pdf(
            self.distance_axis, 0.0, self.minimum_bandwidth
        )
        self.subjective_scale = _js_distance(
            self.subjective_worst,
            self.random_pdf,
            self.distance_axis,
        )
        self.concordant = confidence_geometry(
            x, epsilon, confidence_mode
        ).concordance

    def polarization(self, rho: FloatArray) -> float:
        """Return the objective polarization index for one density state."""

        objective_mass = _distance_mass(
            np.outer(rho, rho), self.distance_bins
        )
        objective_bw = _bandwidth(
            self.distances,
            objective_mass,
            effective_samples=10_000,
            minimum=self.minimum_bandwidth,
        )
        objective_pdf = _mixture_pdf(
            self.distance_axis,
            self.distances,
            objective_mass,
            objective_bw,
        )
        polarized_mass = objective_mass[self.distances >= self.epsilon]
        polarized_distances = self.distances[self.distances >= self.epsilon]
        if polarized_mass.sum() > 1e-15:
            cluster_distance = float(
                np.sum(polarized_distances * polarized_mass)
                / polarized_mass.sum()
            )
        else:
            cluster_distance = self.epsilon
        worst_std = max(
            self.minimum_bandwidth,
            0.5 * cluster_distance * 10_000 ** (-0.2),
        )
        objective_worst = 0.5 * norm.pdf(
            self.distance_axis, 0.0, worst_std
        ) + 0.5 * norm.pdf(
            self.distance_axis, cluster_distance, worst_std
        )
        objective_scale = _js_distance(
            objective_worst,
            self.random_pdf,
            self.distance_axis,
        )
        polarization = 1 - _js_distance(
            objective_pdf, objective_worst, self.distance_axis
        ) / max(objective_scale, 1e-15)
        return float(np.clip(polarization, 0.0, 1.0))

    def subjective(self, edge: FloatArray) -> float:
        """Return the subjective-distance index for one directed-edge state."""

        subjective_mass = _distance_mass(
            edge / self.mean_degree, self.distance_bins
        )
        subjective_bw = _bandwidth(
            self.distances,
            subjective_mass,
            effective_samples=max(int(500 * self.mean_degree), 1),
            minimum=self.minimum_bandwidth,
        )
        subjective_pdf = _mixture_pdf(
            self.distance_axis,
            self.distances,
            subjective_mass,
            subjective_bw,
        )
        subjective = 1 - _js_distance(
            subjective_pdf, self.subjective_worst, self.distance_axis
        ) / max(self.subjective_scale, 1e-15)
        return float(np.clip(subjective, 0.0, 1.0))

    def homophily(self, edge: FloatArray) -> tuple[float, float]:
        """Return normalized and raw bounded-confidence edge mass."""

        homophily_raw = float(
            np.sum(edge * self.concordant) / self.mean_degree
        )
        homophily = normalize_homophily(homophily_raw, self.epsilon)
        return float(np.clip(homophily, 0.0, 1.0)), float(
            np.clip(homophily_raw, 0.0, 1.0)
        )

    def calculate(
        self,
        rho: FloatArray,
        edge: FloatArray,
    ) -> tuple[float, float, float, float]:
        polarization = self.polarization(rho)
        subjective = self.subjective(edge)
        homophily, homophily_raw = self.homophily(edge)

        return (
            polarization,
            subjective,
            homophily,
            homophily_raw,
        )


# Preserve the private name used by older research scripts.
_DensityIndexCalculator = DensityIndexCalculator


def _pathway_index(polarization: FloatArray, homophily: FloatArray) -> float:
    if polarization.size < 2:
        return 0.0
    area = 0.5 * np.sum(
        np.diff(polarization) * (homophily[1:] + homophily[:-1])
    )
    area += (1 - polarization[-1]) * homophily[-1]
    return float(area)


def calculate_index_series(trajectory: KineticTrajectory) -> IndexSeries:
    calculator = DensityIndexCalculator(
        trajectory.x,
        trajectory.parameters.epsilon,
        trajectory.parameters.mean_degree,
        trajectory.parameters.confidence_mode,
    )
    rows = np.asarray(
        [
            calculator.calculate(rho, edge)
            for rho, edge in zip(trajectory.rho, trajectory.edge)
        ]
    )
    polarization = rows[:, 0]
    subjective = rows[:, 1]
    homophily = rows[:, 2]
    return IndexSeries(
        time=trajectory.time.copy(),
        polarization=polarization,
        subjective=subjective,
        homophily=homophily,
        homophily_raw=rows[:, 3],
        pathway=_pathway_index(polarization, homophily),
    )

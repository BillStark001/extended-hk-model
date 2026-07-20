"""Numerical mesoscopic solver for the no-repost/no-history EHK model.

The state uses probability *masses* on a uniform opinion grid:

``rho[i]``
    Fraction of agents in opinion bin ``i``.

``edge[i, j]``
    Directed edges per agent from source bin ``i`` to target bin ``j``.

Consequently ``rho.sum() == 1``, ``edge.sum() == mean_degree``, and
``edge.sum(axis=1) == mean_degree * rho``. The update is a conservative
operator splitting of opinion transport, optional reflecting diffusion, and
one-for-one rewiring.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import sparse


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class KineticParameters:
    """Parameters of the mesoscopic closure.

    ``influence`` and ``rewiring`` are rates per model time unit. With the
    default ``dt=1`` they have the same numerical values as the corresponding
    microscopic per-step parameters.
    """

    epsilon: float = 0.45
    influence: float = 0.05
    rewiring: float = 0.025
    mean_degree: float = 15.0
    recsys_count: int = 10
    recsys: str = "random"
    random_mix: float = 0.1
    opinion_bandwidth: float = 0.1
    noise_diffusion: float = 0.0
    grid_size: int = 81
    dt: float = 1.0
    steps: int = 1200
    record_every: int = 5

    def validate(self) -> None:
        if self.grid_size < 11 or self.grid_size % 2 == 0:
            raise ValueError("grid_size must be an odd integer >= 11")
        if not 0 < self.epsilon <= 2:
            raise ValueError("epsilon must be in (0, 2]")
        if not 0 <= self.influence <= 1:
            raise ValueError("influence must be in [0, 1]")
        if not 0 <= self.rewiring <= 1:
            raise ValueError("rewiring must be in [0, 1]")
        if self.mean_degree <= 0 or self.recsys_count <= 0:
            raise ValueError("mean_degree and recsys_count must be positive")
        if not 0 <= self.random_mix <= 1:
            raise ValueError("random_mix must be in [0, 1]")
        if self.opinion_bandwidth <= 0:
            raise ValueError("opinion_bandwidth must be positive")
        if self.noise_diffusion < 0:
            raise ValueError("noise_diffusion must be non-negative")
        if self.dt <= 0 or self.steps < 1 or self.record_every < 1:
            raise ValueError("dt, steps, and record_every must be positive")


@dataclass
class KineticTrajectory:
    parameters: KineticParameters
    x: FloatArray
    time: FloatArray
    rho: FloatArray
    velocity: FloatArray
    edge: FloatArray
    rewiring_flux: FloatArray

    def metadata(self) -> dict[str, Any]:
        return asdict(self.parameters)


def _row_normalize(values: FloatArray, fallback: FloatArray) -> FloatArray:
    result = np.maximum(values, 0.0)
    row_sum = result.sum(axis=1, keepdims=True)
    valid = row_sum[:, 0] > 1e-15
    result[valid] /= row_sum[valid]
    result[~valid] = fallback[~valid]
    return result


def _conditional_neighbors(
    rho: FloatArray,
    edge: FloatArray,
    mean_degree: float,
) -> FloatArray:
    fallback = np.broadcast_to(rho, edge.shape).copy()
    denominator = mean_degree * rho[:, None]
    neighbors = np.divide(
        edge,
        denominator,
        out=np.zeros_like(edge),
        where=denominator > 1e-15,
    )
    return _row_normalize(neighbors, fallback)


def _recommendation_kernel(
    params: KineticParameters,
    x: FloatArray,
    rho: FloatArray,
    neighbors: FloatArray,
) -> FloatArray:
    """Return one recommendation-slot distribution for every source bin.

    The structure kernel is an outgoing-common-neighbor pair closure. The
    exact simulator uses directed in/out common-neighbor rankings, which
    requires triplet densities and cannot be recovered from ``edge`` alone.
    """

    random_kernel = np.broadcast_to(rho, (x.size, x.size)).copy()
    recsys = params.recsys.casefold()
    if recsys in {"random", "rand"}:
        return random_kernel

    if recsys in {"opinion", "opinionm9", "op"}:
        delta = x[None, :] - x[:, None]
        score = np.exp(-0.5 * (delta / params.opinion_bandwidth) ** 2)
        core = _row_normalize(score * rho[None, :], random_kernel)
    elif recsys in {"structure", "structurem9", "st"}:
        # Expected overlap of the two endpoints' outgoing neighborhoods.
        score = neighbors @ neighbors.T
        core = _row_normalize(score * rho[None, :], random_kernel)
    else:
        raise ValueError(f"unsupported recommendation system: {params.recsys}")

    return params.random_mix * random_kernel + (1 - params.random_mix) * core


def _compute_fields(
    params: KineticParameters,
    x: FloatArray,
    rho: FloatArray,
    edge: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Compute opinion velocity and the rewiring gain/loss field."""

    k = params.mean_degree
    neighbors = _conditional_neighbors(rho, edge, k)
    recommendations = _recommendation_kernel(params, x, rho, neighbors)

    delta = x[None, :] - x[:, None]
    concordant = np.abs(delta) <= params.epsilon
    visible_mass = k * neighbors + params.recsys_count * recommendations
    denominator = np.sum(concordant * visible_mass, axis=1)
    numerator = np.sum(concordant * delta * visible_mass, axis=1)
    velocity = params.influence * np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-15,
    )

    discordant = ~concordant
    discordant_probability = np.sum(discordant * neighbors, axis=1)
    concordant_rec_probability = np.sum(
        concordant * recommendations, axis=1
    )

    loss = np.divide(
        discordant * neighbors,
        discordant_probability[:, None],
        out=np.zeros_like(edge),
        where=discordant_probability[:, None] > 1e-15,
    )
    gain = np.divide(
        concordant * recommendations,
        concordant_rec_probability[:, None],
        out=np.zeros_like(edge),
        where=concordant_rec_probability[:, None] > 1e-15,
    )

    # Probability that finite followee/recommendation samples contain at
    # least one eligible item in each channel.
    eligibility = (
        1 - np.power(1 - discordant_probability, params.mean_degree)
    ) * (
        1 - np.power(
            1 - concordant_rec_probability,
            params.recsys_count,
        )
    )
    event_rate = params.rewiring * rho * eligibility
    rewiring_flux = event_rate[:, None] * (gain - loss)
    return velocity, rewiring_flux


def _reflect(values: FloatArray, lower: float, upper: float) -> FloatArray:
    width = upper - lower
    folded = np.mod(values - lower, 2 * width)
    return lower + np.where(folded <= width, folded, 2 * width - folded)


def _transport_matrix(
    x: FloatArray,
    velocity: FloatArray,
    dt: float,
) -> sparse.csr_matrix:
    """Column-stochastic conservative linear-remapping matrix."""

    dx = x[1] - x[0]
    destination = _reflect(x + dt * velocity, x[0], x[-1])
    coordinate = np.clip((destination - x[0]) / dx, 0, x.size - 1)
    left = np.floor(coordinate).astype(int)
    right = np.minimum(left + 1, x.size - 1)
    weight_right = coordinate - left
    weight_left = 1 - weight_right

    columns = np.arange(x.size)
    rows = np.concatenate((left, right))
    cols = np.concatenate((columns, columns))
    data = np.concatenate((weight_left, weight_right))
    matrix = sparse.coo_matrix(
        (data, (rows, cols)), shape=(x.size, x.size)
    ).tocsr()
    matrix.eliminate_zeros()
    return matrix


def _diffusion_matrix(grid_size: int, ratio: float) -> sparse.csr_matrix:
    """One explicit reflecting-diffusion substep as a Markov matrix."""

    if not 0 <= ratio <= 0.5:
        raise ValueError("diffusion ratio must be in [0, 0.5]")
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for source in range(grid_size):
        if source == 0:
            targets = ((0, 1 - ratio), (1, ratio))
        elif source == grid_size - 1:
            targets = ((grid_size - 2, ratio), (grid_size - 1, 1 - ratio))
        else:
            targets = (
                (source - 1, ratio),
                (source, 1 - 2 * ratio),
                (source + 1, ratio),
            )
        for target, probability in targets:
            rows.append(target)
            cols.append(source)
            data.append(probability)
    return sparse.coo_matrix(
        (data, (rows, cols)), shape=(grid_size, grid_size)
    ).tocsr()


def _apply_markov_transport(
    transition: sparse.csr_matrix,
    rho: FloatArray,
    edge: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    rho_next = np.asarray(transition @ rho).ravel()
    edge_next = np.asarray((transition @ edge) @ transition.T)
    return rho_next, edge_next


def _repair_state(
    rho: FloatArray,
    edge: FloatArray,
    mean_degree: float,
) -> tuple[FloatArray, FloatArray]:
    rho = np.maximum(np.asarray(rho, dtype=float), 0.0)
    rho_sum = rho.sum()
    if rho_sum <= 0:
        raise FloatingPointError("opinion mass vanished")
    rho /= rho_sum

    edge = np.maximum(np.asarray(edge, dtype=float), 0.0)
    desired_rows = mean_degree * rho
    rows = edge.sum(axis=1)
    valid = rows > 1e-15
    edge[valid] *= (desired_rows[valid] / rows[valid])[:, None]
    edge[~valid] = desired_rows[~valid, None] * rho[None, :]
    return rho, edge


def solve(params: KineticParameters) -> KineticTrajectory:
    """Solve and record the mesoscopic density dynamics."""

    params.validate()
    x = np.linspace(-1.0, 1.0, params.grid_size)
    dx = x[1] - x[0]
    rho = np.full(params.grid_size, 1 / params.grid_size, dtype=float)
    edge = params.mean_degree * np.outer(rho, rho)

    record_steps = list(range(0, params.steps + 1, params.record_every))
    if record_steps[-1] != params.steps:
        record_steps.append(params.steps)
    record_set = set(record_steps)

    times: list[float] = []
    rhos: list[FloatArray] = []
    velocities: list[FloatArray] = []
    edges: list[FloatArray] = []
    fluxes: list[FloatArray] = []

    def record(step: int) -> None:
        velocity, flux = _compute_fields(params, x, rho, edge)
        times.append(step * params.dt)
        rhos.append(rho.copy())
        velocities.append(velocity.copy())
        edges.append(edge.copy())
        fluxes.append(flux.copy())

    record(0)

    diffusion_matrices: list[sparse.csr_matrix] = []
    if params.noise_diffusion > 0:
        total_ratio = params.noise_diffusion * params.dt / (dx * dx)
        n_substeps = max(1, int(np.ceil(total_ratio / 0.45)))
        ratio = total_ratio / n_substeps
        diffusion_matrices = [
            _diffusion_matrix(params.grid_size, ratio)
            for _ in range(n_substeps)
        ]

    for step in range(1, params.steps + 1):
        velocity, rewiring_flux = _compute_fields(params, x, rho, edge)

        # Both processes use the old state, matching the synchronous
        # microscopic update to first order in dt.
        edge_with_rewiring = edge + params.dt * rewiring_flux
        transition = _transport_matrix(x, velocity, params.dt)
        rho, edge = _apply_markov_transport(
            transition, rho, edge_with_rewiring
        )
        for diffusion in diffusion_matrices:
            rho, edge = _apply_markov_transport(diffusion, rho, edge)
        rho, edge = _repair_state(rho, edge, params.mean_degree)

        if step in record_set:
            record(step)

    return KineticTrajectory(
        parameters=params,
        x=x,
        time=np.asarray(times),
        rho=np.asarray(rhos),
        velocity=np.asarray(velocities),
        edge=np.asarray(edges),
        rewiring_flux=np.asarray(fluxes),
    )

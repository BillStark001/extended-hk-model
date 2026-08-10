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
BoolArray = NDArray[np.bool_]


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
    # The microscopic ``*M9`` systems reserve this fraction of the finite
    # recommendation slots for Random.  At the standard RecsysCount=10 this
    # gives one random and nine ranked recommendations.
    random_mix: float = 0.1
    opinion_bandwidth: float = 0.1
    noise_diffusion: float = 0.0
    grid_size: int = 81
    dt: float = 1.0
    steps: int = 1200
    record_every: int = 5

    def validate(self) -> None:
        continuous = {
            "epsilon": self.epsilon,
            "influence": self.influence,
            "rewiring": self.rewiring,
            "mean_degree": self.mean_degree,
            "random_mix": self.random_mix,
            "opinion_bandwidth": self.opinion_bandwidth,
            "noise_diffusion": self.noise_diffusion,
            "dt": self.dt,
        }
        nonfinite = [
            name for name, value in continuous.items()
            if not np.isfinite(value)
        ]
        if nonfinite:
            raise ValueError(
                "continuous parameters must be finite: "
                + ", ".join(nonfinite)
            )
        integer_parameters = {
            "grid_size": self.grid_size,
            "recsys_count": self.recsys_count,
            "steps": self.steps,
            "record_every": self.record_every,
        }
        invalid_integers = [
            name for name, value in integer_parameters.items()
            if isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ]
        if invalid_integers:
            raise ValueError(
                "count parameters must be integers: "
                + ", ".join(invalid_integers)
            )
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
        if not float(self.mean_degree).is_integer():
            raise ValueError(
                "mean_degree must be an integer in the fixed-out-degree closure"
            )
        if not 0 <= self.random_mix <= 1:
            raise ValueError("random_mix must be in [0, 1]")
        if self.opinion_bandwidth <= 0:
            raise ValueError("opinion_bandwidth must be positive")
        if self.noise_diffusion < 0:
            raise ValueError("noise_diffusion must be non-negative")
        if self.dt <= 0 or self.steps < 1 or self.record_every < 1:
            raise ValueError("dt, steps, and record_every must be positive")
        if self.dt * self.influence > 1 + 1e-12:
            raise ValueError("dt * influence must not exceed 1")
        if self.dt * self.rewiring > 1 + 1e-12:
            raise ValueError("dt * rewiring must not exceed 1")
        if self.recsys.casefold() not in {
            "random", "rand", "opinion", "op", "structure", "st",
            "opinionm9", "structurem9",
        }:
            raise ValueError(f"unsupported recommendation system: {self.recsys}")


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


@dataclass(frozen=True)
class _GridOperators:
    """Time-independent grid arrays reused by every solver step."""

    delta: FloatArray
    concordant: BoolArray
    opinion_score: FloatArray | None


def _build_grid_operators(
    params: KineticParameters,
    x: FloatArray,
) -> _GridOperators:
    delta = x[None, :] - x[:, None]
    concordant = np.abs(delta) <= params.epsilon
    if params.recsys.casefold() in {"opinion", "opinionm9", "op"}:
        opinion_score = np.exp(
            -0.5 * (delta / params.opinion_bandwidth) ** 2
        )
    else:
        opinion_score = None
    return _GridOperators(delta, concordant, opinion_score)


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
    fallback = np.broadcast_to(rho, edge.shape)
    denominator = mean_degree * rho[:, None]
    neighbors = np.divide(
        edge,
        denominator,
        out=np.zeros_like(edge),
        where=denominator > 1e-15,
    )
    return _row_normalize(neighbors, fallback)


def _recommendation_channels(
    params: KineticParameters,
    x: FloatArray,
    rho: FloatArray,
    neighbors: FloatArray,
    opinion_score: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, int, int]:
    """Return random/core kernels and their finite recommendation-slot counts.

    The structure kernel is an outgoing-common-neighbor pair closure. The
    exact simulator uses directed in/out common-neighbor rankings, which
    requires triplet densities and cannot be recovered from ``edge`` alone.

    ``OpinionM9`` and ``StructureM9`` are not aliases for the pure systems:
    they reserve ``round(RecsysCount * random_mix)`` slots for Random.  This
    reproduces the microscopic default of one random plus nine ranked items
    when ``RecsysCount=10`` and ``random_mix=0.1``.
    """

    random_kernel = np.broadcast_to(rho, (x.size, x.size))
    recsys = params.recsys.casefold()
    if recsys in {"random", "rand"}:
        return random_kernel, random_kernel, params.recsys_count, 0

    if recsys in {"opinion", "opinionm9", "op"}:
        score = opinion_score
        if score is None:
            delta = x[None, :] - x[:, None]
            score = np.exp(-0.5 * (delta / params.opinion_bandwidth) ** 2)
        core = _row_normalize(score * rho[None, :], random_kernel)
    elif recsys in {"structure", "structurem9", "st"}:
        # Expected overlap of the two endpoints' outgoing neighborhoods.
        score = neighbors @ neighbors.T
        core = _row_normalize(score * rho[None, :], random_kernel)
    else:
        raise ValueError(f"unsupported recommendation system: {params.recsys}")

    if recsys in {"opinionm9", "structurem9"}:
        random_slots = int(
            np.floor(params.recsys_count * params.random_mix + 0.5)
        )
        random_slots = min(max(random_slots, 0), params.recsys_count)
    else:
        random_slots = 0
    core_slots = params.recsys_count - random_slots
    return random_kernel, core, random_slots, core_slots


def _recommendation_kernel(
    params: KineticParameters,
    x: FloatArray,
    rho: FloatArray,
    neighbors: FloatArray,
) -> FloatArray:
    """Return the expected source distribution of one recommendation slot."""

    random_kernel, core, random_slots, core_slots = _recommendation_channels(
        params, x, rho, neighbors
    )
    return (
        random_slots * random_kernel + core_slots * core
    ) / params.recsys_count


def _compute_fields(
    params: KineticParameters,
    x: FloatArray,
    rho: FloatArray,
    edge: FloatArray,
    operators: _GridOperators | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Compute opinion velocity and the rewiring gain/loss field."""

    if operators is None:
        operators = _build_grid_operators(params, x)

    k = params.mean_degree
    neighbors = _conditional_neighbors(rho, edge, k)
    random_kernel, core_kernel, random_slots, core_slots = (
        _recommendation_channels(
            params,
            x,
            rho,
            neighbors,
            opinion_score=operators.opinion_score,
        )
    )
    recommendations = (
        random_slots * random_kernel + core_slots * core_kernel
    ) / params.recsys_count

    delta = operators.delta
    concordant = operators.concordant
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
    concordant_random_probability = np.sum(
        concordant * random_kernel, axis=1
    )
    concordant_core_probability = np.sum(
        concordant * core_kernel, axis=1
    )
    concordant_rec_probability = (
        random_slots * concordant_random_probability
        + core_slots * concordant_core_probability
    ) / params.recsys_count

    loss = np.divide(
        discordant * neighbors,
        discordant_probability[:, None],
        out=np.zeros_like(edge),
        where=discordant_probability[:, None] > 1e-15,
    )
    concordant_recommendation_mass = (
        random_slots * random_kernel + core_slots * core_kernel
    )
    gain = np.divide(
        concordant * concordant_recommendation_mass,
        (params.recsys_count * concordant_rec_probability)[:, None],
        out=np.zeros_like(edge),
        where=(params.recsys_count * concordant_rec_probability)[:, None] > 1e-15,
    )

    # Probability that finite followee/recommendation samples contain at
    # least one eligible item in each channel.
    eligibility = (
        1 - np.power(1 - discordant_probability, params.mean_degree)
    ) * (
        1
        - np.power(1 - concordant_random_probability, random_slots)
        * np.power(1 - concordant_core_probability, core_slots)
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


def _validate_state(
    rho: FloatArray,
    edge: FloatArray,
    mean_degree: float,
    tolerance: float = 1e-10,
) -> tuple[FloatArray, FloatArray]:
    """Reject material invariant violations; remove only negative roundoff.

    The transport and one-for-one rewiring operators conserve mass and fixed
    out-degree algebraically. Projecting every step back onto those invariants
    can conceal an unstable time step, so this routine never renormalizes or
    rescales rows.
    """

    rho = np.asarray(rho, dtype=float)
    edge = np.asarray(edge, dtype=float)
    if not np.all(np.isfinite(rho)) or not np.all(np.isfinite(edge)):
        raise FloatingPointError("mesoscopic state contains non-finite values")
    if float(rho.min()) < -tolerance or float(edge.min()) < -tolerance:
        raise FloatingPointError("mesoscopic update produced negative mass")
    if np.any(rho < 0):
        rho = rho.copy()
        rho[rho < 0] = 0.0
    if np.any(edge < 0):
        edge = edge.copy()
        edge[edge < 0] = 0.0

    mass_error = abs(float(rho.sum()) - 1.0)
    row_error = float(
        np.max(np.abs(edge.sum(axis=1) - mean_degree * rho))
    )
    if mass_error > tolerance or row_error > tolerance:
        raise FloatingPointError(
            "mesoscopic update violated mass/out-degree invariants: "
            f"mass_error={mass_error:.3e}, row_error={row_error:.3e}"
        )
    return rho, edge


def solve(params: KineticParameters) -> KineticTrajectory:
    """Solve and record the mesoscopic density dynamics."""

    params.validate()
    x = np.linspace(-1.0, 1.0, params.grid_size)
    dx = x[1] - x[0]
    rho = np.full(params.grid_size, 1 / params.grid_size, dtype=float)
    edge = params.mean_degree * np.outer(rho, rho)
    operators = _build_grid_operators(params, x)

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
        velocity, flux = _compute_fields(
            params, x, rho, edge, operators=operators
        )
        times.append(step * params.dt)
        rhos.append(rho.copy())
        velocities.append(velocity.copy())
        edges.append(edge.copy())
        fluxes.append(flux.copy())

    record(0)

    diffusion_matrix: sparse.csr_matrix | None = None
    diffusion_substeps = 0
    if params.noise_diffusion > 0:
        total_ratio = params.noise_diffusion * params.dt / (dx * dx)
        diffusion_substeps = max(1, int(np.ceil(total_ratio / 0.45)))
        ratio = total_ratio / diffusion_substeps
        diffusion_matrix = _diffusion_matrix(params.grid_size, ratio)

    for step in range(1, params.steps + 1):
        velocity, rewiring_flux = _compute_fields(
            params, x, rho, edge, operators=operators
        )

        # Both processes use the old state, matching the synchronous
        # microscopic update to first order in dt.
        edge_with_rewiring = edge + params.dt * rewiring_flux
        transition = _transport_matrix(x, velocity, params.dt)
        rho, edge = _apply_markov_transport(
            transition, rho, edge_with_rewiring
        )
        if diffusion_matrix is not None:
            for _ in range(diffusion_substeps):
                rho, edge = _apply_markov_transport(
                    diffusion_matrix, rho, edge
                )
        rho, edge = _validate_state(rho, edge, params.mean_degree)

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

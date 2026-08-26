"""Finite-volume PDE solver for the no-repost/no-history EHK model.

The state uses probability *masses* on a uniform opinion grid:

``rho[i]``
    Fraction of agents in opinion bin ``i``.

``edge[i, j]``
    Directed edges per agent from source bin ``i`` to target bin ``j``.

Consequently ``rho.sum() == 1``, ``edge.sum() == mean_degree``, and
``edge.sum(axis=1) == mean_degree * rho``.  The coupled continuity equations
are discretized by a cell-centered finite-volume method with zero numerical
flux at the opinion boundaries.  Backward-Euler face fluxes make each
one-dimensional transport--diffusion sweep conservative and positivity
preserving; the same sweep is applied to the node density and to both edge
endpoints.  One-for-one rewiring is an explicit conservative source step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded

from ..opinion_cells import ConfidenceMode, confidence_geometry
from .directional_wedge import (
    DirectionalWedgeState,
    independent_directional_wedge,
    mix_after_rewiring,
    transport_directional_wedge,
)

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
    # The microscopic ``*M9`` systems reserve this fraction of the finite
    # recommendation slots for Random.  At the standard RecsysCount=10 this
    # gives one random and nine ranked recommendations.
    random_mix: float = 0.1
    # Parameters of the weighted-random recommenders implemented by the
    # microscopic runtime.  OpinionRandom uses
    # [1-|x-y|/opinion_tolerance]_+^recommendation_steepness;
    # StructureRandom uses the same steepness power on its structural score.
    opinion_tolerance: float = 0.4
    recommendation_steepness: float = 1.0
    recommendation_random_ratio: float = 0.0
    # Retained only for the legacy Gaussian ``opinion`` closure.
    opinion_bandwidth: float = 0.1
    noise_diffusion: float = 0.0
    grid_size: int = 81
    dt: float = 1.0
    steps: int = 1200
    record_every: int = 5
    confidence_mode: ConfidenceMode = "cell_average"

    def validate(self) -> None:
        continuous = {
            "epsilon": self.epsilon,
            "influence": self.influence,
            "rewiring": self.rewiring,
            "mean_degree": self.mean_degree,
            "random_mix": self.random_mix,
            "opinion_tolerance": self.opinion_tolerance,
            "recommendation_steepness": self.recommendation_steepness,
            "recommendation_random_ratio": self.recommendation_random_ratio,
            "opinion_bandwidth": self.opinion_bandwidth,
            "noise_diffusion": self.noise_diffusion,
            "dt": self.dt,
        }
        nonfinite = [
            name for name, value in continuous.items() if not np.isfinite(value)
        ]
        if nonfinite:
            raise ValueError(
                "continuous parameters must be finite: " + ", ".join(nonfinite)
            )
        integer_parameters = {
            "grid_size": self.grid_size,
            "recsys_count": self.recsys_count,
            "steps": self.steps,
            "record_every": self.record_every,
        }
        invalid_integers = [
            name
            for name, value in integer_parameters.items()
            if isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
        ]
        if invalid_integers:
            raise ValueError(
                "count parameters must be integers: " + ", ".join(invalid_integers)
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
        if self.opinion_tolerance <= 0:
            raise ValueError("opinion_tolerance must be positive")
        if self.recommendation_steepness <= 0:
            raise ValueError("recommendation_steepness must be positive")
        if not 0 <= self.recommendation_random_ratio <= 1:
            raise ValueError("recommendation_random_ratio must be in [0, 1]")
        if self.opinion_bandwidth <= 0:
            raise ValueError("opinion_bandwidth must be positive")
        if self.noise_diffusion < 0:
            raise ValueError("noise_diffusion must be non-negative")
        if self.dt <= 0 or self.steps < 1 or self.record_every < 1:
            raise ValueError("dt, steps, and record_every must be positive")
        if self.dt * self.rewiring > 1 + 1e-12:
            raise ValueError("dt * rewiring must not exceed 1")
        if self.confidence_mode not in {"cell_average", "center"}:
            raise ValueError(f"unknown confidence mode: {self.confidence_mode}")
        if self.recsys.casefold() not in {
            "random",
            "rand",
            "opinion",
            "op",
            "structure",
            "st",
            "opinionm9",
            "structurem9",
            "opinionrandom",
            "opinion_random",
            "structure_random_l0",
            "structurerandoml0",
            "structure_random_l1",
            "structurerandoml1",
        }:
            raise ValueError(f"unsupported recommendation system: {self.recsys}")
        if self.recsys.casefold() in {
            "structure_random_l1",
            "structurerandoml1",
        } and not np.isclose(self.recommendation_steepness, 1.0):
            raise ValueError(
                "structure_random_l1 closes only the first score moment; "
                "recommendation_steepness must equal 1"
            )


@dataclass
class KineticTrajectory:
    parameters: KineticParameters
    x: FloatArray
    time: FloatArray
    rho: FloatArray
    velocity: FloatArray
    edge: FloatArray
    rewiring_flux: FloatArray
    structural_score: FloatArray | None = None

    def metadata(self) -> dict[str, Any]:
        return asdict(self.parameters)


@dataclass(frozen=True)
class _GridOperators:
    """Time-independent grid arrays reused by every solver step."""

    concordant: FloatArray
    displacement: FloatArray
    opinion_score: FloatArray | None


def _build_grid_operators(
    params: KineticParameters,
    x: FloatArray,
) -> _GridOperators:
    delta = x[None, :] - x[:, None]
    geometry = confidence_geometry(x, params.epsilon, params.confidence_mode)
    concordant = geometry.concordance
    recsys = params.recsys.casefold()
    if recsys in {"opinion", "opinionm9", "op"}:
        opinion_score = np.exp(-0.5 * (delta / params.opinion_bandwidth) ** 2)
    elif recsys in {"opinionrandom", "opinion_random"}:
        opinion_score = (
            np.maximum(
                1.0 - np.abs(delta) / params.opinion_tolerance,
                0.0,
            )
            ** params.recommendation_steepness
        )
    else:
        opinion_score = None
    return _GridOperators(concordant, geometry.displacement, opinion_score)


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
    structural_score: FloatArray | None = None,
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
    elif recsys in {"opinionrandom", "opinion_random"}:
        score = opinion_score
        if score is None:
            delta = x[None, :] - x[:, None]
            score = (
                np.maximum(
                    1.0 - np.abs(delta) / params.opinion_tolerance,
                    0.0,
                )
                ** params.recommendation_steepness
            )
        weighted = _row_normalize(score * rho[None, :], random_kernel)
        beta = params.recommendation_random_ratio
        core = (1.0 - beta) * weighted + beta * random_kernel
    elif recsys in {"structure", "structurem9", "st"}:
        # Expected overlap of the two endpoints' outgoing neighborhoods.
        score = neighbors @ neighbors.T
        core = _row_normalize(score * rho[None, :], random_kernel)
    elif recsys in {"structure_random_l0", "structurerandoml0"}:
        # L0 pair proxy for StructureRandom.  The microscopic score is an
        # integer common-neighbor count; replacing its steepness power by the
        # power of the expected overlap is an explicit moment closure.
        score = np.maximum(neighbors @ neighbors.T, 0.0)
        score = score**params.recommendation_steepness
        weighted = _row_normalize(score * rho[None, :], random_kernel)
        beta = params.recommendation_random_ratio
        core = (1.0 - beta) * weighted + beta * random_kernel
    elif recsys in {"structure_random_l1", "structurerandoml1"}:
        if structural_score is None:
            raise ValueError(
                "structure_random_l1 requires a directional-wedge score mass"
            )
        if structural_score.shape != (x.size, x.size):
            raise ValueError("structural score has incompatible shape")
        # The L1 score mass already contains the target-bin candidate mass;
        # unlike the L0 conditional-overlap proxy it is not multiplied by rho.
        weighted = _row_normalize(structural_score, random_kernel)
        beta = params.recommendation_random_ratio
        core = (1.0 - beta) * weighted + beta * random_kernel
    else:
        raise ValueError(f"unsupported recommendation system: {params.recsys}")

    if recsys in {"opinionm9", "structurem9"}:
        random_slots = int(np.floor(params.recsys_count * params.random_mix + 0.5))
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
    structural_score: FloatArray | None = None,
) -> FloatArray:
    """Return the expected source distribution of one recommendation slot."""

    random_kernel, core, random_slots, core_slots = _recommendation_channels(
        params, x, rho, neighbors, structural_score=structural_score
    )
    return (random_slots * random_kernel + core_slots * core) / params.recsys_count


def _compute_fields(
    params: KineticParameters,
    x: FloatArray,
    rho: FloatArray,
    edge: FloatArray,
    operators: _GridOperators | None = None,
    structural_score: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Compute opinion velocity and the rewiring gain/loss field."""

    if operators is None:
        operators = _build_grid_operators(params, x)

    k = params.mean_degree
    neighbors = _conditional_neighbors(rho, edge, k)
    random_kernel, core_kernel, random_slots, core_slots = _recommendation_channels(
        params,
        x,
        rho,
        neighbors,
        opinion_score=operators.opinion_score,
        structural_score=structural_score,
    )
    recommendations = (
        random_slots * random_kernel + core_slots * core_kernel
    ) / params.recsys_count

    concordant = operators.concordant
    visible_mass = k * neighbors + params.recsys_count * recommendations
    denominator = np.sum(concordant * visible_mass, axis=1)
    numerator = np.sum(operators.displacement * visible_mass, axis=1)
    velocity = params.influence * np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-15,
    )

    discordant = 1.0 - concordant
    discordant_probability = np.sum(discordant * neighbors, axis=1)
    concordant_random_probability = np.sum(concordant * random_kernel, axis=1)
    concordant_core_probability = np.sum(concordant * core_kernel, axis=1)
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
    eligibility = (1 - np.power(1 - discordant_probability, params.mean_degree)) * (
        1
        - np.power(1 - concordant_random_probability, random_slots)
        * np.power(1 - concordant_core_probability, core_slots)
    )
    event_rate = params.rewiring * rho * eligibility
    rewiring_flux = event_rate[:, None] * (gain - loss)
    return velocity, rewiring_flux


def _finite_volume_system(
    velocity: FloatArray,
    diffusion: float,
    dx: float,
    dt: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return the tridiagonal backward-Euler finite-volume system.

    At an interior face, first-order upwinding supplies the advective flux and
    a centered difference supplies the diffusive flux.  Boundary-face fluxes
    are identically zero, which is the finite-volume no-flux condition.  The
    resulting semi-discrete generator is conservative and Metzler, so
    ``I - dt * L`` is an M-matrix for every positive ``dt``.
    """

    face_velocity = 0.5 * (velocity[:-1] + velocity[1:])
    diffusion_rate = diffusion / (dx * dx)
    left_to_right = np.maximum(face_velocity, 0.0) / dx + diffusion_rate
    right_to_left = np.maximum(-face_velocity, 0.0) / dx + diffusion_rate

    size = velocity.size
    lower = -dt * left_to_right.copy()
    upper = -dt * right_to_left.copy()
    diagonal = np.ones(size, dtype=float)
    diagonal[:-1] += dt * left_to_right
    diagonal[1:] += dt * right_to_left
    return lower, diagonal, upper


def _solve_tridiagonal(
    lower: FloatArray,
    diagonal: FloatArray,
    upper: FloatArray,
    right_hand_side: FloatArray,
) -> FloatArray:
    """Solve a tridiagonal system for one or many right-hand sides."""

    rhs = np.asarray(right_hand_side, dtype=float)
    vector_input = rhs.ndim == 1
    if vector_input:
        rhs = rhs[:, None]
    if rhs.ndim != 2 or rhs.shape[0] != diagonal.size:
        raise ValueError("right-hand side has incompatible shape")

    bands = np.zeros((3, diagonal.size), dtype=float)
    bands[0, 1:] = upper
    bands[1] = diagonal
    bands[2, :-1] = lower
    solution = solve_banded((1, 1), bands, rhs, overwrite_ab=True, check_finite=False)
    return solution[:, 0] if vector_input else solution


def _advance_transport_diffusion(
    rho: FloatArray,
    edge: FloatArray,
    velocity: FloatArray,
    diffusion: float,
    dx: float,
    dt: float,
) -> tuple[FloatArray, FloatArray]:
    """Advance the coupled node/edge transport--diffusion PDE by one step.

    The x-endpoint sweep advances ``rho`` and the rows of ``edge`` with the
    same finite-volume operator.  The y-endpoint sweep then advances every
    edge row along its target coordinate.  Because the second sweep preserves
    each row sum, ``edge.sum(axis=1) == mean_degree * rho`` is inherited
    algebraically from the input state.
    """

    lower, diagonal, upper = _finite_volume_system(velocity, diffusion, dx, dt)
    return _advance_transport_diffusion_with_system(rho, edge, lower, diagonal, upper)


def _transport_array_axis(
    values: FloatArray,
    axis: int,
    lower: FloatArray,
    diagonal: FloatArray,
    upper: FloatArray,
) -> FloatArray:
    """Apply one finite-volume system along an arbitrary array axis."""

    moved = np.moveaxis(np.asarray(values, dtype=float), axis, 0)
    original_shape = moved.shape
    right_hand_side = moved.reshape(original_shape[0], -1)
    solution = _solve_tridiagonal(lower, diagonal, upper, right_hand_side).reshape(
        original_shape
    )
    return np.moveaxis(solution, 0, axis)


def _advance_transport_diffusion_with_system(
    rho: FloatArray,
    edge: FloatArray,
    lower: FloatArray,
    diagonal: FloatArray,
    upper: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Advance node and edge states with one precomputed FV system."""

    rho_next = _solve_tridiagonal(lower, diagonal, upper, rho)
    edge_after_x = _solve_tridiagonal(lower, diagonal, upper, edge)
    edge_next = _solve_tridiagonal(lower, diagonal, upper, edge_after_x.T).T
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
    row_error = float(np.max(np.abs(edge.sum(axis=1) - mean_degree * rho)))
    if mass_error > tolerance or row_error > tolerance:
        raise FloatingPointError(
            "mesoscopic update violated mass/out-degree invariants: "
            f"mass_error={mass_error:.3e}, row_error={row_error:.3e}"
        )
    return rho, edge


def solve(params: KineticParameters) -> KineticTrajectory:
    """Solve and record the mesoscopic density dynamics."""

    params.validate()
    dx = 2.0 / params.grid_size
    x = -1.0 + (np.arange(params.grid_size, dtype=float) + 0.5) * dx
    rho = np.full(params.grid_size, 1 / params.grid_size, dtype=float)
    edge = params.mean_degree * np.outer(rho, rho)
    operators = _build_grid_operators(params, x)
    uses_l1 = params.recsys.casefold() in {"structure_random_l1", "structurerandoml1"}
    wedge: DirectionalWedgeState | None = (
        independent_directional_wedge(rho, edge) if uses_l1 else None
    )

    record_steps = list(range(0, params.steps + 1, params.record_every))
    if record_steps[-1] != params.steps:
        record_steps.append(params.steps)
    record_set = set(record_steps)

    times: list[float] = []
    rhos: list[FloatArray] = []
    velocities: list[FloatArray] = []
    edges: list[FloatArray] = []
    fluxes: list[FloatArray] = []
    structural_scores: list[FloatArray] = []

    def record(step: int) -> None:
        structural_score = wedge.union_score_mass() if wedge is not None else None
        velocity, flux = _compute_fields(
            params,
            x,
            rho,
            edge,
            operators=operators,
            structural_score=structural_score,
        )
        times.append(step * params.dt)
        rhos.append(rho.copy())
        velocities.append(velocity.copy())
        edges.append(edge.copy())
        fluxes.append(flux.copy())
        if structural_score is not None:
            structural_scores.append(structural_score.copy())

    record(0)

    for step in range(1, params.steps + 1):
        structural_score = wedge.union_score_mass() if wedge is not None else None
        velocity, rewiring_flux = _compute_fields(
            params,
            x,
            rho,
            edge,
            operators=operators,
            structural_score=structural_score,
        )

        # Lie splitting of the PDE: a conservative rewiring source step,
        # followed by conservative no-flux transport--diffusion sweeps in the
        # source and target opinion coordinates.  All coefficients are frozen
        # at the old state, so the nonlinear method is first order in time.
        edge_with_rewiring = edge + params.dt * rewiring_flux
        wedge_after_rewiring = (
            mix_after_rewiring(wedge, rho, edge, edge_with_rewiring)
            if wedge is not None
            else None
        )
        lower, diagonal, upper = _finite_volume_system(
            velocity, params.noise_diffusion, dx, params.dt
        )
        rho, edge = _advance_transport_diffusion_with_system(
            rho, edge_with_rewiring, lower, diagonal, upper
        )
        if wedge_after_rewiring is not None:
            wedge = transport_directional_wedge(
                wedge_after_rewiring,
                lambda values, axis, lo=lower, diag=diagonal, up=upper: (
                    _transport_array_axis(values, axis, lo, diag, up)
                ),
            )
            wedge.validate(params.grid_size)
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
        structural_score=(np.asarray(structural_scores) if structural_scores else None),
    )

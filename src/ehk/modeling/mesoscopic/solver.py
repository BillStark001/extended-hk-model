"""Mesoscopic kinetic solvers for the no-repost/no-history EHK model.

The state uses probability *masses* on a uniform opinion grid:

``rho[i]``
    Fraction of agents in opinion bin ``i``.

``edge[i, j]``
    Directed edges per agent from source bin ``i`` to target bin ``j``.

Consequently ``rho.sum() == 1``, ``edge.sum() == mean_degree``, and
``edge.sum(axis=1) == mean_degree * rho``. Opinion updating can use either an
explicit nonlocal transition kernel or a sparse one-/two-moment closure of
that kernel. The latter is the discrete-time Fokker--Planck approximation: on
the current grid it matches every source row's destination mean and variance
exactly, instead of mixing a synchronous jump with an unrelated implicit PDE
time step. Both operators are conservative and positivity preserving and are
applied consistently to node, edge, and optional directional-wedge states.
One-for-one rewiring is an explicit conservative source step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import solve_banded
from scipy.sparse import csr_matrix, spmatrix

from ..opinion_cells import (
    CompromiseJumpGeometry,
    ConfidenceMode,
    compromise_jump_geometry,
    confidence_geometry,
)
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

    ``dynamics`` selects deterministic HK averaging or Deffuant random-neighbor
    compromise. ``opinion_method`` selects the full nonlocal push-forward or a
    sparse discrete Fokker--Planck closure that exactly matches the full
    kernel's first two destination moments on the chosen opinion grid.
    ``dt * influence`` is the compromise fraction per numerical step and
    ``dt * rewiring`` is the rewiring probability per step.
    """

    epsilon: float = 0.45
    influence: float = 0.05
    rewiring: float = 0.025
    dynamics: str = "hk"
    opinion_method: str = "fokker_planck"
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
        if self.dynamics.casefold() not in {"hk", "deffuant"}:
            raise ValueError(f"unsupported opinion dynamics: {self.dynamics}")
        if self.opinion_method.casefold() not in {
            "fokker_planck",
            "nonlocal_jump",
        }:
            raise ValueError(f"unsupported opinion method: {self.opinion_method}")
        if self.dt * self.influence > 1 + 1e-12:
            raise ValueError("dt * influence must not exceed 1")
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
            "structure_random_l1_mean_power",
            "structurerandoml1meanpower",
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
    displacement_variance: FloatArray
    displacement_second_moment: FloatArray
    endogenous_diffusion: FloatArray
    structural_score: FloatArray | None = None

    def metadata(self) -> dict[str, Any]:
        return asdict(self.parameters)


@dataclass(frozen=True)
class _GridOperators:
    """Time-independent grid arrays reused by every solver step."""

    concordant: FloatArray
    displacement: FloatArray
    displacement_second: FloatArray
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
    return _GridOperators(
        concordant,
        geometry.displacement,
        geometry.displacement_second,
        opinion_score,
    )


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
    elif recsys in {
        "structure_random_l1",
        "structurerandoml1",
        "structure_random_l1_mean_power",
        "structurerandoml1meanpower",
    }:
        if structural_score is None:
            raise ValueError(
                "structure_random_l1 requires a directional-wedge score mass"
            )
        if structural_score.shape != (x.size, x.size):
            raise ValueError("structural score has incompatible shape")
        # The L1 score mass already contains the target-bin candidate mass.
        # The strict L1 rule therefore uses it directly and is defined only
        # for zeta=1.  The explicitly named mean-power variant closes higher
        # powers as C E[S]^zeta from the retained M1=C E[S].  It does not claim
        # to recover the unavailable higher score moments E[S^zeta].
        if recsys in {
            "structure_random_l1_mean_power",
            "structurerandoml1meanpower",
        }:
            candidate_mass = np.broadcast_to(rho, structural_score.shape)
            conditional_mean = np.divide(
                structural_score,
                candidate_mass,
                out=np.zeros_like(structural_score),
                where=candidate_mass > 1e-15,
            )
            score_mass = candidate_mass * np.power(
                conditional_mean, params.recommendation_steepness
            )
        else:
            score_mass = structural_score
        weighted = _row_normalize(score_mass, random_kernel)
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


def _compute_fields_with_selection(
    params: KineticParameters,
    x: FloatArray,
    rho: FloatArray,
    edge: FloatArray,
    operators: _GridOperators | None = None,
    structural_score: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Compute drift, rewiring flux, and the concordant selection kernel."""

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
    selection = np.divide(
        concordant * visible_mass,
        denominator[:, None],
        out=np.zeros_like(visible_mass),
        where=denominator[:, None] > 1e-15,
    )
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
    return velocity, rewiring_flux, selection


def _compute_fields(
    params: KineticParameters,
    x: FloatArray,
    rho: FloatArray,
    edge: FloatArray,
    operators: _GridOperators | None = None,
    structural_score: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Compute opinion velocity and the rewiring gain/loss field."""

    velocity, rewiring_flux, _ = _compute_fields_with_selection(
        params,
        x,
        rho,
        edge,
        operators=operators,
        structural_score=structural_score,
    )
    return velocity, rewiring_flux


def _deposition_coordinates(
    x: FloatArray,
    destination: FloatArray,
) -> tuple[NDArray[np.int64], NDArray[np.int64], FloatArray]:
    """Return conservative linear-deposition indices and upper weights."""

    size = x.size
    dx = float(x[1] - x[0])
    coordinate = (destination - x[0]) / dx
    lower = np.clip(np.floor(coordinate).astype(np.int64), 0, size - 1)
    upper = np.minimum(lower + 1, size - 1)
    fraction = np.clip(coordinate - lower, 0.0, 1.0)
    return lower, upper, fraction


def _validate_selection_grid(
    x: FloatArray, selection: FloatArray, influence: float
) -> tuple[FloatArray, FloatArray]:
    x_values = np.asarray(x, dtype=float)
    probabilities = np.asarray(selection, dtype=float)
    if x_values.ndim != 1 or probabilities.shape != (x_values.size,) * 2:
        raise ValueError("selection must have shape (x.size, x.size)")
    if not 0 <= influence <= 1:
        raise ValueError("influence must be in [0, 1]")
    if x_values.size < 2 or not np.allclose(
        np.diff(x_values), x_values[1] - x_values[0]
    ):
        raise ValueError("x must be a uniform one-dimensional grid")
    if (
        not np.all(np.isfinite(x_values))
        or not np.all(np.diff(x_values) > 0)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0)
    ):
        raise ValueError(
            "x must increase and selection must be finite and non-negative"
        )
    return x_values, probabilities


def _deffuant_transition_from_selection(
    x: FloatArray,
    influence: float,
    selection: FloatArray,
    jump_geometry: CompromiseJumpGeometry | None = None,
) -> FloatArray:
    """Return a row-stochastic source-to-destination compromise kernel.

    ``selection[i, j]`` is the probability that an agent in cell ``i`` picks
    a concordant visible opinion in cell ``j``. The continuous destination
    ``x_i + influence * (x_j - x_i)`` is deposited linearly onto the two
    neighboring cell centers, preserving its first moment exactly.
    """

    x_values, probabilities = _validate_selection_grid(x, selection, influence)

    size = x_values.size
    if jump_geometry is not None:
        if jump_geometry.destination_index.shape[:2] != (size, size):
            raise ValueError("jump geometry must match the selection grid")
        source = np.broadcast_to(np.arange(size)[:, None], (size, size))
        transition = np.zeros((size, size), dtype=float)
        for slot in range(jump_geometry.weight.shape[2]):
            destination = jump_geometry.destination_index[:, :, slot]
            contribution = probabilities * jump_geometry.weight[:, :, slot]
            valid = destination >= 0
            np.add.at(
                transition,
                (source[valid], destination[valid]),
                contribution[valid],
            )
        row_sum = transition.sum(axis=1)
        invalid = np.flatnonzero(row_sum <= 1e-15)
        transition[invalid, invalid] = 1.0
        transition /= transition.sum(axis=1, keepdims=True)
        return transition

    row_sum = probabilities.sum(axis=1)
    valid = row_sum > 1e-15
    normalized = np.divide(
        probabilities,
        row_sum[:, None],
        out=np.zeros_like(probabilities),
        where=valid[:, None],
    )
    destination = x_values[:, None] + influence * (
        x_values[None, :] - x_values[:, None]
    )
    lower, upper, fraction = _deposition_coordinates(x_values, destination)
    source = np.broadcast_to(np.arange(size)[:, None], lower.shape)
    transition = np.zeros((size, size), dtype=float)
    np.add.at(transition, (source, lower), normalized * (1.0 - fraction))
    np.add.at(transition, (source, upper), normalized * fraction)
    invalid = np.flatnonzero(~valid)
    transition[invalid, invalid] = 1.0
    transition /= transition.sum(axis=1, keepdims=True)
    return transition


def _hk_transition_from_selection(
    x: FloatArray,
    influence: float,
    selection: FloatArray,
    displacement_mean: FloatArray | None = None,
) -> FloatArray:
    """Return the deterministic HK conditional-mean push-forward kernel."""

    x_values, probabilities = _validate_selection_grid(x, selection, influence)
    size = x_values.size
    row_sum = probabilities.sum(axis=1)
    valid = row_sum > 1e-15
    if displacement_mean is None:
        mean_target = np.divide(
            probabilities @ x_values,
            row_sum,
            out=x_values.copy(),
            where=valid,
        )
        displacement = mean_target - x_values
    else:
        displacement = np.asarray(displacement_mean, dtype=float)
        if displacement.shape != x_values.shape:
            raise ValueError("displacement mean must match x")
    destination = x_values + influence * displacement
    lower, upper, fraction = _deposition_coordinates(x_values, destination)
    source = np.arange(size)
    transition = np.zeros((size, size), dtype=float)
    transition[source, lower] += 1.0 - fraction
    transition[source, upper] += fraction
    transition /= transition.sum(axis=1, keepdims=True)
    return transition


def _moment_matched_transition(
    x: FloatArray,
    reference: FloatArray,
    *,
    tolerance: float = 5e-13,
) -> csr_matrix:
    """Compress a transition to a sparse positive two-moment closure.

    Each reference row defines a destination mean and variance on the current
    grid.  The closure mixes (i) the minimum-variance two-node distribution
    bracketing that mean and (ii) the narrowest balanced outer two-node
    distribution needed to bracket the requested variance.  It therefore has
    at most four nonzeros per row and exactly preserves zeroth, first, and
    second destination moments up to roundoff.

    Matching the *discrete* reference moments is important. Linear deposition
    itself contributes grid-scale variance, so matching only the continuous
    pre-deposition moments makes two nominally identical operators disagree at
    finite resolution.
    """

    grid = np.asarray(x, dtype=float)
    kernel = np.asarray(reference, dtype=float)
    size = grid.size
    if grid.ndim != 1 or kernel.shape != (size, size):
        raise ValueError("reference transition must be square and match x")
    if np.any(kernel < -tolerance) or not np.allclose(
        kernel.sum(axis=1), 1.0, atol=tolerance
    ):
        raise ValueError("reference transition must be non-negative and stochastic")

    destination_mean = kernel @ grid
    destination_second = kernel @ (grid * grid)
    destination_variance = np.maximum(
        destination_second - destination_mean * destination_mean,
        0.0,
    )
    dx = float(grid[1] - grid[0])
    coordinate = (destination_mean - grid[0]) / dx
    inner_left = np.clip(np.floor(coordinate).astype(np.int64), 0, size - 1)
    inner_right = np.minimum(inner_left + 1, size - 1)
    on_node = np.abs(destination_mean - grid[inner_left]) <= tolerance
    inner_right[on_node] = inner_left[on_node]
    inner_span = grid[inner_right] - grid[inner_left]
    inner_right_weight = np.divide(
        destination_mean - grid[inner_left],
        inner_span,
        out=np.zeros(size, dtype=float),
        where=inner_span > 0,
    )
    inner_variance = (1.0 - inner_right_weight) * (
        grid[inner_left] - destination_mean
    ) ** 2 + inner_right_weight * (grid[inner_right] - destination_mean) ** 2
    variance_scale = np.maximum.reduce(
        (np.ones(size), destination_variance, inner_variance)
    )
    if np.any(destination_variance < inner_variance - tolerance * variance_scale):
        raise FloatingPointError(
            "reference variance is below the grid's realizable minimum"
        )

    needs_outer = destination_variance > (inner_variance + tolerance * variance_scale)
    left_capacity = destination_mean - grid[0]
    right_capacity = grid[-1] - destination_mean
    if np.any(
        needs_outer & ((left_capacity <= tolerance) | (right_capacity <= tolerance))
    ):
        raise FloatingPointError(
            "positive variance is not realizable at an opinion boundary"
        )
    radius = np.sqrt(destination_variance)
    left_distance = np.minimum(radius, left_capacity)
    right_distance = np.divide(
        destination_variance,
        left_distance,
        out=np.zeros(size, dtype=float),
        where=left_distance > 0,
    )
    right_limited = right_distance > right_capacity
    right_distance[right_limited] = right_capacity[right_limited]
    left_distance[right_limited] = np.divide(
        destination_variance[right_limited],
        right_distance[right_limited],
        out=np.zeros(np.count_nonzero(right_limited), dtype=float),
        where=right_distance[right_limited] > 0,
    )
    outer_left = np.clip(
        np.floor((destination_mean - left_distance - grid[0]) / dx).astype(np.int64),
        0,
        size - 1,
    )
    outer_right = np.clip(
        np.ceil((destination_mean + right_distance - grid[0]) / dx).astype(np.int64),
        0,
        size - 1,
    )
    if np.any(needs_outer & (outer_left >= outer_right)):
        raise FloatingPointError("failed to bracket a positive variance")
    outer_span = grid[outer_right] - grid[outer_left]
    outer_right_weight = np.divide(
        destination_mean - grid[outer_left],
        outer_span,
        out=np.zeros(size, dtype=float),
        where=outer_span > 0,
    )
    outer_variance = (1.0 - outer_right_weight) * (
        grid[outer_left] - destination_mean
    ) ** 2 + outer_right_weight * (grid[outer_right] - destination_mean) ** 2
    if np.any(
        needs_outer
        & (outer_variance < destination_variance - tolerance * variance_scale)
    ):
        raise FloatingPointError("failed to bracket the requested variance")
    mixture = np.divide(
        destination_variance - inner_variance,
        outer_variance - inner_variance,
        out=np.zeros(size, dtype=float),
        where=needs_outer,
    )
    mixture = np.clip(mixture, 0.0, 1.0)

    source = np.arange(size, dtype=np.int64)
    rows = np.tile(source, 4)
    columns = np.concatenate((inner_left, inner_right, outer_left, outer_right))
    values = np.concatenate(
        (
            (1.0 - mixture) * (1.0 - inner_right_weight),
            (1.0 - mixture) * inner_right_weight,
            mixture * (1.0 - outer_right_weight),
            mixture * outer_right_weight,
        )
    )
    positive = values > 0.0
    closure = csr_matrix(
        (values[positive], (rows[positive], columns[positive])),
        shape=kernel.shape,
    )
    closure.sum_duplicates()
    closure.eliminate_zeros()
    row_sum = np.asarray(closure.sum(axis=1)).ravel()
    closure.data *= np.repeat(1.0 / row_sum, np.diff(closure.indptr))
    row_sum = np.asarray(closure.sum(axis=1)).ravel()
    if not np.allclose(row_sum, 1.0, atol=5e-12):
        raise FloatingPointError("moment closure is not row stochastic")
    closure_mean = np.asarray(closure @ grid).ravel()
    closure_second = np.asarray(closure @ (grid * grid)).ravel()
    if not np.allclose(closure_mean, destination_mean, atol=5e-12) or not np.allclose(
        closure_second, destination_second, atol=5e-12
    ):
        raise FloatingPointError("moment closure failed to preserve moments")
    return closure


def _selection_displacement_moments(
    selection: FloatArray,
    operators: _GridOperators,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return mean, raw second moment, and variance of ``Y-X`` given ``X``."""

    pair_mean = np.divide(
        operators.displacement,
        operators.concordant,
        out=np.zeros_like(operators.displacement),
        where=operators.concordant > 1e-15,
    )
    pair_second = np.divide(
        operators.displacement_second,
        operators.concordant,
        out=np.zeros_like(operators.displacement_second),
        where=operators.concordant > 1e-15,
    )
    mean = np.sum(selection * pair_mean, axis=1)
    second = np.sum(selection * pair_second, axis=1)
    variance = np.maximum(second - mean * mean, 0.0)
    return mean, second, variance


def _endogenous_diffusion(
    params: KineticParameters,
    displacement_mean: FloatArray,
    displacement_second_moment: FloatArray,
) -> FloatArray:
    """Return the physical one-step diffusion from conditional jump variance.

    The synchronous one-step closure transports the conditional mean exactly;
    only variance around that mean is diffusive. HK therefore has no physical
    endogenous diffusion, whereas Deffuant retains the conditional-neighbor
    variance. Grid-deposition variance is matched separately by the discrete
    transition and is intentionally excluded from this diagnostic.
    """

    if params.dynamics.casefold() == "hk":
        return np.zeros_like(displacement_mean)
    variance = np.maximum(
        displacement_second_moment - displacement_mean * displacement_mean,
        0.0,
    )
    return 0.5 * params.dt * params.influence * params.influence * variance


def _jump_transition_from_selection(
    params: KineticParameters,
    x: FloatArray,
    selection: FloatArray,
    displacement_mean: FloatArray,
    jump_geometry: CompromiseJumpGeometry | None = None,
) -> FloatArray:
    """Build the full frozen jump transition for either opinion dynamics."""

    influence = params.dt * params.influence
    if params.dynamics.casefold() == "deffuant":
        geometry = jump_geometry or compromise_jump_geometry(
            x,
            params.epsilon,
            influence,
            params.confidence_mode,
        )
        return _deffuant_transition_from_selection(
            x,
            influence,
            selection,
            geometry,
        )
    return _hk_transition_from_selection(
        x,
        influence,
        selection,
        displacement_mean,
    )


def opinion_transition_matrix(
    params: KineticParameters,
    x: FloatArray,
    rho: FloatArray,
    edge: FloatArray,
    *,
    structural_score: FloatArray | None = None,
) -> FloatArray:
    """Build the configured frozen opinion transition for one state."""

    operators = _build_grid_operators(params, np.asarray(x, dtype=float))
    _, _, selection = _compute_fields_with_selection(
        params,
        np.asarray(x, dtype=float),
        np.asarray(rho, dtype=float),
        np.asarray(edge, dtype=float),
        operators=operators,
        structural_score=structural_score,
    )
    displacement_mean, _, _ = _selection_displacement_moments(selection, operators)
    reference = _jump_transition_from_selection(
        params,
        np.asarray(x, dtype=float),
        selection,
        displacement_mean,
    )
    if (
        params.opinion_method.casefold() == "nonlocal_jump"
        or params.dynamics.casefold() == "hk"
    ):
        return reference
    return _moment_matched_transition(x, reference).toarray()


def deffuant_transition_matrix(
    params: KineticParameters,
    x: FloatArray,
    rho: FloatArray,
    edge: FloatArray,
    *,
    structural_score: FloatArray | None = None,
) -> FloatArray:
    """Build a frozen Deffuant kernel; retained as an explicit public alias."""

    if params.dynamics.casefold() != "deffuant":
        raise ValueError("deffuant transition requires dynamics='deffuant'")
    return opinion_transition_matrix(
        params, x, rho, edge, structural_score=structural_score
    )


def _apply_transition_axis(
    values: FloatArray,
    axis: int,
    transition: FloatArray | spmatrix,
) -> FloatArray:
    """Apply a row-stochastic source-to-destination kernel along one axis."""

    moved = np.moveaxis(np.asarray(values, dtype=float), axis, 0)
    shape = moved.shape
    result = transition.T @ moved.reshape(shape[0], -1)
    return np.moveaxis(result.reshape(shape), 0, axis)


def advance_jump_density(
    rho: FloatArray,
    transition: FloatArray,
) -> FloatArray:
    """Apply one frozen row-stochastic opinion jump to a node density."""

    rho_values = np.asarray(rho, dtype=float)
    kernel = np.asarray(transition, dtype=float)
    if rho_values.ndim != 1 or kernel.shape != (rho_values.size,) * 2:
        raise ValueError("transition must be square and match rho")
    if np.any(kernel < -1e-12) or not np.allclose(kernel.sum(axis=1), 1.0, atol=1e-12):
        raise ValueError("transition must be non-negative and row stochastic")
    return np.asarray(rho_values @ kernel, dtype=float)


def advance_deffuant_density(
    rho: FloatArray,
    transition: FloatArray,
) -> FloatArray:
    """Apply a Deffuant kernel; retained as an explicit public alias."""

    return advance_jump_density(rho, transition)


def _finite_volume_system(
    velocity: FloatArray,
    diffusion: float | FloatArray,
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
    size = velocity.size
    diffusion_values = np.asarray(diffusion, dtype=float)
    if diffusion_values.ndim == 0:
        diffusion_values = np.full(size, float(diffusion_values))
    if diffusion_values.shape != (size,) or np.any(diffusion_values < 0):
        raise ValueError("diffusion must be non-negative and scalar or velocity-shaped")
    left_to_right = np.maximum(face_velocity, 0.0) / dx + diffusion_values[:-1] / (
        dx * dx
    )
    right_to_left = np.maximum(-face_velocity, 0.0) / dx + diffusion_values[1:] / (
        dx * dx
    )

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


def advance_opinion_density(
    rho: FloatArray,
    velocity: FloatArray,
    *,
    dx: float,
    dt: float,
    diffusion: float = 0.0,
) -> FloatArray:
    """Apply one frozen-field opinion-only step to a node density.

    This is the node component of the solver's backward-Euler finite-volume
    transport operator.  It is public so channel counterfactuals can measure
    the macro-level response to opinion updating without applying rewiring.
    """

    rho_values = np.asarray(rho, dtype=float)
    velocity_values = np.asarray(velocity, dtype=float)
    if rho_values.ndim != 1 or velocity_values.shape != rho_values.shape:
        raise ValueError("rho and velocity must be equal one-dimensional arrays")
    if dx <= 0 or dt <= 0 or diffusion < 0:
        raise ValueError("dx and dt must be positive and diffusion non-negative")
    lower, diagonal, upper = _finite_volume_system(velocity_values, diffusion, dx, dt)
    result = _solve_tridiagonal(lower, diagonal, upper, rho_values)
    if not np.all(np.isfinite(result)) or float(result.min()) < -1e-10:
        raise FloatingPointError("opinion counterfactual produced invalid density")
    result = np.maximum(result, 0.0)
    mass = float(result.sum())
    if abs(mass - float(rho_values.sum())) > 1e-10:
        raise FloatingPointError("opinion counterfactual violated node mass")
    return result


def advance_frozen_opinion_density(
    params: KineticParameters,
    x: FloatArray,
    rho: FloatArray,
    edge: FloatArray,
    *,
    dt: float | None = None,
    structural_score: FloatArray | None = None,
) -> FloatArray:
    """Advance only the configured frozen opinion operator for one state.

    Rewiring and exogenous regularizing diffusion are excluded. The moment
    closure is built from the same full jump kernel as the nonlocal update.
    """

    probe = replace(
        params,
        dt=params.dt if dt is None else dt,
        rewiring=0.0,
        noise_diffusion=0.0,
    )
    probe.validate()
    x_values = np.asarray(x, dtype=float)
    rho_values = np.asarray(rho, dtype=float)
    edge_values = np.asarray(edge, dtype=float)
    operators = _build_grid_operators(probe, x_values)
    _, _, selection = _compute_fields_with_selection(
        probe,
        x_values,
        rho_values,
        edge_values,
        operators=operators,
        structural_score=structural_score,
    )
    mean, _, _ = _selection_displacement_moments(selection, operators)
    reference = _jump_transition_from_selection(
        probe,
        x_values,
        selection,
        mean,
    )
    transition: FloatArray | spmatrix = reference
    if (
        probe.opinion_method.casefold() == "fokker_planck"
        and probe.dynamics.casefold() == "deffuant"
    ):
        transition = _moment_matched_transition(x_values, reference)
    return np.asarray(transition.T @ rho_values, dtype=float)


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


def solve(
    params: KineticParameters,
    *,
    record_steps: Sequence[int] | None = None,
) -> KineticTrajectory:
    """Solve and record the mesoscopic dynamics at regular or custom steps."""

    params.validate()
    dx = 2.0 / params.grid_size
    x = -1.0 + (np.arange(params.grid_size, dtype=float) + 0.5) * dx
    rho = np.full(params.grid_size, 1 / params.grid_size, dtype=float)
    edge = params.mean_degree * np.outer(rho, rho)
    operators = _build_grid_operators(params, x)
    jump_geometry = (
        compromise_jump_geometry(
            x,
            params.epsilon,
            params.dt * params.influence,
            params.confidence_mode,
        )
        if params.dynamics.casefold() == "deffuant"
        else None
    )
    uses_l1 = params.recsys.casefold() in {
        "structure_random_l1",
        "structurerandoml1",
        "structure_random_l1_mean_power",
        "structurerandoml1meanpower",
    }
    wedge: DirectionalWedgeState | None = (
        independent_directional_wedge(rho, edge) if uses_l1 else None
    )

    if record_steps is None:
        selected_steps = list(range(0, params.steps + 1, params.record_every))
    else:
        invalid = [
            step
            for step in record_steps
            if isinstance(step, (bool, np.bool_))
            or not isinstance(step, (int, np.integer))
            or not 0 <= int(step) <= params.steps
        ]
        if invalid:
            raise ValueError("record_steps must contain integers in [0, steps]")
        selected_steps = sorted({int(step) for step in record_steps})
    selected_steps = sorted({0, params.steps, *selected_steps})
    record_set = set(selected_steps)

    times: list[float] = []
    rhos: list[FloatArray] = []
    velocities: list[FloatArray] = []
    edges: list[FloatArray] = []
    fluxes: list[FloatArray] = []
    displacement_variances: list[FloatArray] = []
    displacement_second_moments: list[FloatArray] = []
    endogenous_diffusions: list[FloatArray] = []
    structural_scores: list[FloatArray] = []

    def record(step: int) -> None:
        structural_score = wedge.union_score_mass() if wedge is not None else None
        velocity, flux, selection = _compute_fields_with_selection(
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
        displacement_mean, displacement_second_moment, displacement_variance = (
            _selection_displacement_moments(selection, operators)
        )
        displacement_variances.append(displacement_variance)
        displacement_second_moments.append(displacement_second_moment)
        endogenous_diffusions.append(
            _endogenous_diffusion(params, displacement_mean, displacement_second_moment)
        )
        if structural_score is not None:
            structural_scores.append(structural_score.copy())

    record(0)

    for step in range(1, params.steps + 1):
        structural_score = wedge.union_score_mass() if wedge is not None else None
        velocity, rewiring_flux, selection = _compute_fields_with_selection(
            params,
            x,
            rho,
            edge,
            operators=operators,
            structural_score=structural_score,
        )

        # Lie splitting: a conservative rewiring source step followed by the
        # configured conservative opinion operator. All coefficients are
        # frozen at the old state, so the nonlinear method is first order in
        # time.
        edge_with_rewiring = edge + params.dt * rewiring_flux
        wedge_after_rewiring = (
            mix_after_rewiring(wedge, rho, edge, edge_with_rewiring)
            if wedge is not None
            else None
        )
        displacement_mean, _, _ = _selection_displacement_moments(selection, operators)
        reference_transition = _jump_transition_from_selection(
            params,
            x,
            selection,
            displacement_mean,
            jump_geometry,
        )
        sparse_transition: spmatrix = csr_matrix(reference_transition)
        if (
            params.opinion_method.casefold() == "fokker_planck"
            and params.dynamics.casefold() == "deffuant"
        ):
            sparse_transition = _moment_matched_transition(
                x,
                reference_transition,
            )
        rho = np.asarray(sparse_transition.T @ rho, dtype=float)
        edge = _apply_transition_axis(edge_with_rewiring, 0, sparse_transition)
        edge = _apply_transition_axis(edge, 1, sparse_transition)
        if wedge_after_rewiring is not None:
            wedge = transport_directional_wedge(
                wedge_after_rewiring,
                lambda values, axis, kernel=sparse_transition: _apply_transition_axis(
                    values, axis, kernel
                ),
            )
        if params.noise_diffusion > 0:
            zero_velocity = np.zeros_like(velocity)
            lower, diagonal, upper = _finite_volume_system(
                zero_velocity, params.noise_diffusion, dx, params.dt
            )
            rho, edge = _advance_transport_diffusion_with_system(
                rho, edge, lower, diagonal, upper
            )
            if wedge is not None:
                wedge = transport_directional_wedge(
                    wedge,
                    lambda values, axis, lo=lower, diag=diagonal, up=upper: (
                        _transport_array_axis(values, axis, lo, diag, up)
                    ),
                )
        if wedge is not None:
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
        displacement_variance=np.asarray(displacement_variances),
        displacement_second_moment=np.asarray(displacement_second_moments),
        endogenous_diffusion=np.asarray(endogenous_diffusions),
        structural_score=(np.asarray(structural_scores) if structural_scores else None),
    )

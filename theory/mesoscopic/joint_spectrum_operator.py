"""Joint opinion--edge linear operator for the periodic EHK pair closure.

The scalar spectrum used in the paper perturbs only the opinion density on the
``q=0`` product-edge manifold.  This module retains the complete conditional
edge perturbation ``h_m(s)`` for each source Fourier mode ``m`` and permits
both influence ``alpha`` and rewiring ``q``.

For ``q>0`` the uniform product-edge state is not stationary.  The correct
translation-invariant base is therefore a rewiring orbit parameterized by its
remaining discordant-neighbor mass.  The module exposes both frozen-time
Jacobians along that orbit and time-ordered finite-horizon singular spectra.
It contains no finite-system committor or terminal-outcome code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.linalg import expm


ComplexArray = NDArray[np.complex128]
FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class JointSpectrumParameters:
    """Parameters of the periodic pair-closure tangent operator."""

    length: float = 2.0
    epsilon: float = 0.45
    mean_degree: int = 15
    recsys_count: int = 10
    random_mix: float = 0.1
    opinion_bandwidth: float = 0.1
    opinion_tolerance: float = 0.4
    diffusion: float = 1e-5
    grid_size: int = 32
    difference_step: float = 1e-6

    def validate(self) -> None:
        if self.length <= 0:
            raise ValueError("length must be positive")
        if not 0 < self.epsilon < self.length / 2:
            raise ValueError("epsilon must lie in (0, length / 2)")
        if self.mean_degree < 1 or self.recsys_count < 1:
            raise ValueError("mean_degree and recsys_count must be positive")
        if not 0 <= self.random_mix <= 1:
            raise ValueError("random_mix must lie in [0, 1]")
        if self.opinion_bandwidth <= 0 or self.opinion_tolerance <= 0:
            raise ValueError("opinion kernel scales must be positive")
        if self.diffusion < 0:
            raise ValueError("diffusion must be non-negative")
        if self.grid_size < 16 or self.grid_size % 2:
            raise ValueError("grid_size must be an even integer >= 16")
        if not 0 < self.difference_step < 1e-2:
            raise ValueError("difference_step must lie in (0, 1e-2)")


@dataclass(frozen=True)
class RecommendationSlots:
    """Finite random/core recommendation-slot allocation and core kernel."""

    name: str
    random: int
    opinion: int
    kernel: str = "gaussian_opinion"
    steepness: float = 1.0

    @property
    def total(self) -> int:
        return self.random + self.opinion


@dataclass(frozen=True)
class JointGrid:
    """Time-independent periodic-grid objects for one recommendation rule."""

    x: FloatArray
    dx: float
    delta: FloatArray
    concordant: BoolArray
    random: FloatArray
    opinion: FloatArray
    gain: FloatArray
    rho0: float
    initial_discordant: float
    recommendation_eligibility: float


@dataclass(frozen=True)
class EigenSpectrum:
    eigenvalues: ComplexArray
    eigenvectors: ComplexArray
    rho_energy_fraction: FloatArray
    order: NDArray[np.int64]


@dataclass(frozen=True)
class FiniteTimeSpectrum:
    log_singular_gains: FloatArray
    finite_time_rates: FloatArray
    rho_seed_log_gain: float
    rho_seed_rate: float
    dominant_output_rho_fraction: float
    final_discordant_fraction_ratio: float


def recommendation_slots(
    name: str,
    *,
    recsys_count: int = 10,
    random_mix: float = 0.1,
    steepness: float = 1.0,
) -> RecommendationSlots:
    """Resolve a recommendation rule used by the periodic pair closure."""

    if steepness <= 0:
        raise ValueError("recommendation steepness must be positive")

    normalized = name.casefold()
    if normalized in {"random", "rand"}:
        return RecommendationSlots("Random", recsys_count, 0)
    if normalized in {"opinion", "op"}:
        return RecommendationSlots("Opinion", 0, recsys_count)
    if normalized == "opinionm9":
        random_count = int(np.floor(recsys_count * random_mix + 0.5))
        random_count = min(max(random_count, 0), recsys_count)
        return RecommendationSlots(
            "OpinionM9", random_count, recsys_count - random_count
        )
    if normalized in {"opinionrandom", "opinion_random"}:
        return RecommendationSlots(
            f"OpinionRandom-zeta{steepness:g}",
            0,
            recsys_count,
            "opinion_random",
            steepness,
        )
    if normalized in {
        "structurerandoml0", "structure_random_l0", "structurerandom"
    }:
        return RecommendationSlots(
            f"L0-StructureRandom-zeta{steepness:g}",
            0,
            recsys_count,
            "structure_random_l0",
            steepness,
        )
    raise ValueError(
        "unsupported joint pair-spectrum recommendation rule; "
        f"got {name!r}"
    )


def _normalize_target_kernel(
    raw: ComplexArray,
    *,
    dx: float,
) -> ComplexArray:
    denominator = raw.sum(axis=1, keepdims=True) * dx
    if np.any(np.abs(denominator) < 1e-14):
        raise FloatingPointError("recommendation kernel has zero row mass")
    return raw / denominator


def _core_recommendation_kernel(
    rho: ComplexArray,
    neighbors: ComplexArray,
    *,
    slots: RecommendationSlots,
    parameters: JointSpectrumParameters,
    delta: FloatArray,
    dx: float,
) -> ComplexArray:
    """Return the normalized core kernel used by one recommendation slot."""

    if slots.kernel == "gaussian_opinion":
        score = np.exp(-0.5 * (delta / parameters.opinion_bandwidth) ** 2)
    elif slots.kernel == "opinion_random":
        score = np.maximum(
            1.0 - np.abs(delta) / parameters.opinion_tolerance,
            0.0,
        ) ** slots.steepness
    elif slots.kernel == "structure_random_l0":
        # Pair-level L0 closure: expected outgoing-neighborhood overlap.
        # The exact microscopic score is integer-valued and motif-dependent.
        score = (neighbors @ neighbors.T) * dx
        score = score ** slots.steepness
    else:
        raise ValueError(f"unsupported recommendation kernel: {slots.kernel}")
    return _normalize_target_kernel(score * rho[None, :], dx=dx)


def periodic_derivative(
    values: ComplexArray,
    *,
    length: float,
    axis: int,
    order: int,
) -> ComplexArray:
    """Spectral derivative on a periodic axis."""

    frequency = 2 * np.pi * np.fft.fftfreq(
        values.shape[axis], d=length / values.shape[axis]
    )
    shape = [1] * values.ndim
    shape[axis] = frequency.size
    multiplier = (1j * frequency.reshape(shape)) ** order
    return np.fft.ifft(
        multiplier * np.fft.fft(values, axis=axis), axis=axis
    ).astype(complex)


def build_joint_grid(
    slots: RecommendationSlots,
    parameters: JointSpectrumParameters,
) -> JointGrid:
    """Build normalized target kernels and the assortative rewiring gain."""

    parameters.validate()
    if slots.total != parameters.recsys_count:
        raise ValueError("recommendation slots do not match recsys_count")
    size = parameters.grid_size
    x = np.arange(size, dtype=float) * parameters.length / size
    dx = parameters.length / size
    delta = x[None, :] - x[:, None]
    delta = (delta + parameters.length / 2) % parameters.length
    delta -= parameters.length / 2
    concordant = np.abs(delta) <= parameters.epsilon
    rho0 = 1 / parameters.length
    random = np.full_like(delta, rho0)
    rho = np.full(size, rho0, dtype=complex)
    neighbors = np.full((size, size), rho0, dtype=complex)
    opinion = _core_recommendation_kernel(
        rho,
        neighbors,
        slots=slots,
        parameters=parameters,
        delta=delta,
        dx=dx,
    ).real
    recommendation_mass = slots.random * random + slots.opinion * opinion
    gain_raw = concordant * recommendation_mass
    gain = gain_raw / (gain_raw.sum(axis=1, keepdims=True) * dx)
    concordant_random = float((concordant[0] * random[0]).sum() * dx)
    concordant_opinion = float((concordant[0] * opinion[0]).sum() * dx)
    recommendation_eligibility = 1 - (
        (1 - concordant_random) ** slots.random
        * (1 - concordant_opinion) ** slots.opinion
    )
    return JointGrid(
        x=x,
        dx=dx,
        delta=delta,
        concordant=concordant,
        random=random,
        opinion=opinion,
        gain=gain,
        rho0=rho0,
        initial_discordant=1 - concordant_random,
        recommendation_eligibility=float(recommendation_eligibility),
    )


def base_state(
    grid: JointGrid,
    parameters: JointSpectrumParameters,
    *,
    discordant_fraction: float,
) -> tuple[ComplexArray, ComplexArray, FloatArray]:
    """Return the uniform-opinion state on the analytic rewiring base orbit."""

    d0 = grid.initial_discordant
    if not 0 < discordant_fraction <= d0 * (1 + 1e-12):
        raise ValueError("discordant_fraction must lie in (0, initial_discordant]")
    discordant_fraction = min(discordant_fraction, d0)
    neighbor = (
        grid.concordant * grid.random
        + (d0 - discordant_fraction) * grid.gain
        + (discordant_fraction / d0) * (~grid.concordant) * grid.random
    )
    rho = np.full(parameters.grid_size, grid.rho0, dtype=complex)
    edge = (
        parameters.mean_degree * rho[:, None] * neighbor
    ).astype(complex)
    return rho, edge, neighbor[0].copy()


def pair_rhs(
    rho: ComplexArray,
    edge: ComplexArray,
    *,
    slots: RecommendationSlots,
    alpha: float,
    rewiring: float,
    parameters: JointSpectrumParameters,
    grid: JointGrid,
) -> tuple[ComplexArray, ComplexArray]:
    """Continuous periodic pair-closure right-hand side."""

    neighbors = edge / (parameters.mean_degree * rho[:, None])
    random = np.broadcast_to(rho[None, :], edge.shape)
    opinion = _core_recommendation_kernel(
        rho,
        neighbors,
        slots=slots,
        parameters=parameters,
        delta=grid.delta,
        dx=grid.dx,
    )
    recommendation_mass = slots.random * random + slots.opinion * opinion
    visible = parameters.mean_degree * neighbors + recommendation_mass
    denominator = (grid.concordant * visible).sum(axis=1) * grid.dx
    numerator = (
        grid.concordant * grid.delta * visible
    ).sum(axis=1) * grid.dx
    velocity = alpha * numerator / denominator

    discordant = ~grid.concordant
    discordant_probability = (
        discordant * neighbors
    ).sum(axis=1) * grid.dx
    concordant_random = (grid.concordant * random).sum(axis=1) * grid.dx
    concordant_opinion = (grid.concordant * opinion).sum(axis=1) * grid.dx
    loss = discordant * neighbors / discordant_probability[:, None]
    gain_raw = grid.concordant * recommendation_mass
    gain = gain_raw / (gain_raw.sum(axis=1, keepdims=True) * grid.dx)
    eligibility = (
        1 - (1 - discordant_probability) ** parameters.mean_degree
    ) * (
        1
        - (1 - concordant_random) ** slots.random
        * (1 - concordant_opinion) ** slots.opinion
    )
    rewiring_flux = (
        rewiring
        * rho[:, None]
        * eligibility[:, None]
        * (gain - loss)
    )

    rho_rhs = -periodic_derivative(
        rho * velocity, length=parameters.length, axis=0, order=1
    )
    rho_rhs += parameters.diffusion * periodic_derivative(
        rho, length=parameters.length, axis=0, order=2
    )
    edge_rhs = -periodic_derivative(
        edge * velocity[:, None],
        length=parameters.length,
        axis=0,
        order=1,
    )
    edge_rhs -= periodic_derivative(
        edge * velocity[None, :],
        length=parameters.length,
        axis=1,
        order=1,
    )
    edge_rhs += parameters.diffusion * (
        periodic_derivative(
            edge, length=parameters.length, axis=0, order=2
        )
        + periodic_derivative(
            edge, length=parameters.length, axis=1, order=2
        )
    )
    edge_rhs += rewiring_flux
    return rho_rhs, edge_rhs


def perturbation(
    *,
    r_amplitude: complex,
    h: ComplexArray,
    mode: int,
    base_neighbor: FloatArray,
    parameters: JointSpectrumParameters,
    grid: JointGrid,
) -> tuple[ComplexArray, ComplexArray]:
    """Map ``(r_m, h_m(s))`` coordinates into ``(delta rho, delta E)``."""

    phase = np.exp(2j * np.pi * mode * grid.x / parameters.length)
    delta_rho = grid.rho0 * r_amplitude * phase
    delta_edge = np.empty(
        (parameters.grid_size, parameters.grid_size), dtype=complex
    )
    target_indices = np.arange(parameters.grid_size)
    for source in range(parameters.grid_size):
        offsets = (target_indices - source) % parameters.grid_size
        delta_edge[source] = (
            parameters.mean_degree
            * grid.rho0
            * phase[source]
            * (
                base_neighbor[offsets] * r_amplitude
                + h[offsets]
            )
        )
    return delta_rho, delta_edge


def project_tangent(
    delta_rho_rhs: ComplexArray,
    delta_edge_rhs: ComplexArray,
    *,
    mode: int,
    base_neighbor: FloatArray,
    parameters: JointSpectrumParameters,
    grid: JointGrid,
) -> tuple[complex, ComplexArray]:
    """Project a tangent RHS back onto ``(r_m, h_m(s))`` coordinates."""

    phase_conjugate = np.exp(
        -2j * np.pi * mode * grid.x / parameters.length
    )
    r_dot = complex(
        np.mean(delta_rho_rhs * phase_conjugate) / grid.rho0
    )
    conditional_dot = np.empty(parameters.grid_size, dtype=complex)
    sources = np.arange(parameters.grid_size)
    for offset in range(parameters.grid_size):
        targets = (sources + offset) % parameters.grid_size
        conditional_dot[offset] = np.mean(
            delta_edge_rhs[sources, targets] * phase_conjugate
        ) / (parameters.mean_degree * grid.rho0)
    h_dot = conditional_dot - base_neighbor * r_dot
    h_dot -= h_dot.mean()
    return r_dot, h_dot


def joint_jacobian(
    *,
    slots: RecommendationSlots,
    alpha: float,
    rewiring: float,
    mode: int,
    discordant_fraction: float,
    parameters: JointSpectrumParameters,
    grid: JointGrid,
) -> ComplexArray:
    """Centered-Fréchet Jacobian in the complete mode-``m`` pair sector."""

    rho, edge, base_neighbor = base_state(
        grid, parameters, discordant_fraction=discordant_fraction
    )
    size = parameters.grid_size
    matrix = np.empty((size, size), dtype=complex)
    step = parameters.difference_step
    for column in range(size):
        r_amplitude = 1.0 if column == 0 else 0.0
        h = np.zeros(size, dtype=complex)
        if column > 0:
            h[column - 1] = 1.0
            h[-1] = -1.0
        delta_rho, delta_edge = perturbation(
            r_amplitude=r_amplitude,
            h=h,
            mode=mode,
            base_neighbor=base_neighbor,
            parameters=parameters,
            grid=grid,
        )
        plus = pair_rhs(
            rho + step * delta_rho,
            edge + step * delta_edge,
            slots=slots,
            alpha=alpha,
            rewiring=rewiring,
            parameters=parameters,
            grid=grid,
        )
        minus = pair_rhs(
            rho - step * delta_rho,
            edge - step * delta_edge,
            slots=slots,
            alpha=alpha,
            rewiring=rewiring,
            parameters=parameters,
            grid=grid,
        )
        r_dot, h_dot = project_tangent(
            (plus[0] - minus[0]) / (2 * step),
            (plus[1] - minus[1]) / (2 * step),
            mode=mode,
            base_neighbor=base_neighbor,
            parameters=parameters,
            grid=grid,
        )
        matrix[0, column] = r_dot
        matrix[1:, column] = h_dot[:-1]
    return matrix


def affine_jacobian_bases(
    *,
    slots: RecommendationSlots,
    mode: int,
    discordant_fraction: float,
    parameters: JointSpectrumParameters,
    grid: JointGrid,
) -> tuple[ComplexArray, ComplexArray, ComplexArray]:
    """Return ``A0, A_alpha, A_q`` with ``A=A0+alpha*A_alpha+q*A_q``."""

    common = {
        "slots": slots,
        "mode": mode,
        "discordant_fraction": discordant_fraction,
        "parameters": parameters,
        "grid": grid,
    }
    base = joint_jacobian(alpha=0.0, rewiring=0.0, **common)
    alpha_one = joint_jacobian(alpha=1.0, rewiring=0.0, **common)
    q_one = joint_jacobian(alpha=0.0, rewiring=1.0, **common)
    return base, alpha_one - base, q_one - base


def physical_transform(
    parameters: JointSpectrumParameters,
    grid: JointGrid,
) -> tuple[FloatArray, FloatArray]:
    """Transform coordinates to the physical ``|r|^2 + integral |h|^2`` norm."""

    size = parameters.grid_size
    metric = np.zeros((size, size))
    metric[0, 0] = 1.0
    metric[1:, 1:] = grid.dx * (
        np.eye(size - 1) + np.ones((size - 1, size - 1))
    )
    transform = np.linalg.cholesky(metric).T
    return transform, np.linalg.inv(transform)


def eigenspectrum(
    matrix: ComplexArray,
    *,
    parameters: JointSpectrumParameters,
    grid: JointGrid,
) -> EigenSpectrum:
    """Return all eigenvalues and physical opinion-energy fractions."""

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    transform, _inverse = physical_transform(parameters, grid)
    physical = transform @ eigenvectors
    energy = np.sum(np.abs(physical) ** 2, axis=0)
    rho_fraction = np.abs(physical[0]) ** 2 / energy
    order = np.argsort(eigenvalues.real)[::-1]
    return EigenSpectrum(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        rho_energy_fraction=np.asarray(rho_fraction, dtype=float),
        order=order,
    )


def discordant_derivative(
    discordant_fraction: float,
    *,
    rewiring: float,
    parameters: JointSpectrumParameters,
    grid: JointGrid,
) -> float:
    """Analytic translation-invariant rewiring-orbit ODE."""

    followee_eligibility = 1 - (
        1 - discordant_fraction
    ) ** parameters.mean_degree
    return float(
        -rewiring
        * grid.recommendation_eligibility
        * followee_eligibility
        / parameters.mean_degree
    )


def discordant_orbit(
    times: FloatArray,
    *,
    rewiring: float,
    parameters: JointSpectrumParameters,
    grid: JointGrid,
) -> FloatArray:
    """Evaluate remaining discordant mass on the analytic base orbit."""

    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or times.size == 0 or np.any(times < 0):
        raise ValueError("times must be a nonempty one-dimensional nonnegative array")
    if np.any(np.diff(times) < 0):
        raise ValueError("times must be sorted")
    if rewiring == 0 or times[-1] == 0:
        return np.full(times.size, grid.initial_discordant)

    def derivative(_time: float, state: FloatArray) -> FloatArray:
        return np.asarray(
            [
                discordant_derivative(
                    max(float(state[0]), 0.0),
                    rewiring=rewiring,
                    parameters=parameters,
                    grid=grid,
                )
            ]
        )

    solution = solve_ivp(
        derivative,
        (0.0, float(times[-1])),
        np.asarray([grid.initial_discordant]),
        t_eval=times,
        rtol=1e-10,
        atol=1e-12,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return np.maximum(solution.y[0], 0.0)


def interpolate_matrix(
    fractions: FloatArray,
    table: ComplexArray,
    fraction: float,
) -> ComplexArray:
    """Linearly interpolate a matrix table in discordant-mass ratio."""

    upper = int(np.searchsorted(fractions, fraction, side="right"))
    if upper <= 0:
        return table[0].copy()
    if upper >= fractions.size:
        return table[-1].copy()
    lower = upper - 1
    weight = (fraction - fractions[lower]) / (
        fractions[upper] - fractions[lower]
    )
    return (1 - weight) * table[lower] + weight * table[upper]


def time_ordered_spectrum(
    *,
    fractions: FloatArray,
    base_table: ComplexArray,
    alpha_table: ComplexArray,
    q_table: ComplexArray,
    alpha: float,
    rewiring: float,
    horizon: float,
    time_steps: int,
    parameters: JointSpectrumParameters,
    grid: JointGrid,
) -> FiniteTimeSpectrum:
    """Compute the complete finite-horizon singular spectrum of the propagator."""

    if horizon <= 0 or time_steps < 1:
        raise ValueError("horizon and time_steps must be positive")
    if rewiring == 0:
        midpoint_times = np.asarray([horizon / 2])
        interval = horizon
    else:
        midpoint_times = (
            np.arange(time_steps, dtype=float) + 0.5
        ) * horizon / time_steps
        interval = horizon / time_steps
    discordant = discordant_orbit(
        midpoint_times,
        rewiring=rewiring,
        parameters=parameters,
        grid=grid,
    )
    endpoint = discordant_orbit(
        np.asarray([horizon]),
        rewiring=rewiring,
        parameters=parameters,
        grid=grid,
    )[0]
    transform, inverse = physical_transform(parameters, grid)
    propagator = np.eye(parameters.grid_size, dtype=complex)
    rho_seed = np.zeros(parameters.grid_size, dtype=complex)
    rho_seed[0] = 1.0
    propagator_scale = 0.0
    rho_scale = 0.0
    for value in discordant:
        fraction = float(value / grid.initial_discordant)
        matrix = interpolate_matrix(fractions, base_table, fraction)
        matrix += alpha * interpolate_matrix(fractions, alpha_table, fraction)
        matrix += rewiring * interpolate_matrix(fractions, q_table, fraction)
        physical_matrix = transform @ matrix @ inverse
        step = expm(physical_matrix * interval)
        propagator = step @ propagator
        rho_seed = step @ rho_seed
        propagator_norm = float(np.linalg.norm(propagator))
        rho_norm = float(np.linalg.norm(rho_seed))
        if propagator_norm == 0 or rho_norm == 0:
            raise FloatingPointError("time-ordered propagator underflowed")
        propagator /= propagator_norm
        rho_seed /= rho_norm
        propagator_scale += float(np.log(propagator_norm))
        rho_scale += float(np.log(rho_norm))
    left, singular_values, _right = np.linalg.svd(
        propagator, full_matrices=False
    )
    log_gains = propagator_scale + np.log(singular_values)
    return FiniteTimeSpectrum(
        log_singular_gains=np.asarray(log_gains, dtype=float),
        finite_time_rates=np.asarray(log_gains / horizon, dtype=float),
        rho_seed_log_gain=rho_scale,
        rho_seed_rate=rho_scale / horizon,
        dominant_output_rho_fraction=float(abs(left[0, 0]) ** 2),
        final_discordant_fraction_ratio=float(
            endpoint / grid.initial_discordant
        ),
    )

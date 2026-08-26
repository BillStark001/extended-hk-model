"""Factorial stochastic mesoscopic generator for terminal outcomes.

The two modeling choices are independent parameters:

``state_level``
    ``pair`` retains ``(rho, edge)``; ``score_moments`` additionally retains
    ``(W, S2, S4)``.

``timescale``
    ``unsplit`` performs one synchronous opinion/rewiring step;
    ``fast_slow`` conditionally absorbs the rewiring channel before a slow
    opinion step when ``q / alpha`` exceeds a declared threshold.

This module contains reusable generator mechanics only.  Experiment grids,
parallel execution, scoring against microscopic data, and plotting belong in
``experiments/theory_guided``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.typing import NDArray
from scipy.signal import find_peaks
from scipy.stats import binom, norm

FloatArray = NDArray[np.float64]
StateLevel = Literal["pair", "score_moments"]
Timescale = Literal["unsplit", "fast_slow"]
_GH_NODES, _GH_WEIGHTS = hermgauss(9)
_GH_WEIGHTS = _GH_WEIGHTS / math.sqrt(math.pi)


@dataclass(frozen=True)
class TerminalGeneratorParameters:
    population: int = 500
    grid_size: int = 15
    mean_degree: int = 15
    recsys_count: int = 10
    epsilon: float = 0.45
    influence: float = 0.05
    rewiring: float = 0.05
    recsys: str = "random"
    recommendation_steepness: float = 1.0
    opinion_tolerance: float = 0.4
    state_level: StateLevel = "score_moments"
    timescale: Timescale = "unsplit"
    max_steps: int = 20_000
    seed: int = 1
    fast_slow_ratio_threshold: float = 20.0
    fast_max_steps: int = 20_000
    fast_zero_checks: int = 20

    def validate(self) -> None:
        if self.population < 2 or self.grid_size < 3:
            raise ValueError("population and grid_size are too small")
        if not 0 < self.mean_degree < self.population:
            raise ValueError("mean_degree must lie in [1, population)")
        if self.recsys_count < 1 or self.max_steps < 1:
            raise ValueError("recsys_count and max_steps must be positive")
        if not 0 < self.epsilon <= 2:
            raise ValueError("epsilon must lie in (0, 2]")
        if not 0 < self.influence <= 1 or not 0 <= self.rewiring <= 1:
            raise ValueError("influence must be positive and rates must not exceed one")
        if self.recsys not in {"random", "opinion_random", "structure_random"}:
            raise ValueError(f"unsupported recommender: {self.recsys}")
        if self.recommendation_steepness not in {1.0, 4.0}:
            raise ValueError("the generator supports score powers 1 and 4")
        if self.state_level not in {"pair", "score_moments"}:
            raise ValueError(f"unknown state level: {self.state_level}")
        if self.timescale not in {"unsplit", "fast_slow"}:
            raise ValueError(f"unknown timescale: {self.timescale}")
        if self.fast_slow_ratio_threshold <= 0 or self.fast_max_steps < 1:
            raise ValueError("invalid fast-slow controls")


@dataclass
class TerminalState:
    rho: FloatArray
    edge: FloatArray
    wedge: FloatArray | None = None
    score2: FloatArray | None = None
    score4: FloatArray | None = None

    @property
    def lifted(self) -> bool:
        return self.wedge is not None


@dataclass(frozen=True)
class TerminalGeneratorResult:
    parameters: TerminalGeneratorParameters
    category: str
    raw_category: str
    converged: bool
    stopped_step: int
    rewiring_events: int
    mean_cap_fraction: float
    fast_slow_applied: bool
    fast_substeps: int
    fast_max_steps_hits: int
    final_fast_residual: float
    state: TerminalState


def opinion_grid(grid_size: int) -> FloatArray:
    dx = 2.0 / grid_size
    return -1.0 + (np.arange(grid_size, dtype=float) + 0.5) * dx


def _row_normalize(values: FloatArray, fallback: FloatArray) -> FloatArray:
    output = np.maximum(values, 0.0).copy()
    total = output.sum(axis=1, keepdims=True)
    valid = total[:, 0] > 1e-15
    output[valid] /= total[valid]
    output[~valid] = fallback[~valid]
    return output


def _neighbors(state: TerminalState, degree: int) -> FloatArray:
    fallback = np.broadcast_to(state.rho, state.edge.shape)
    values = np.divide(
        state.edge,
        degree * state.rho[:, None],
        out=np.zeros_like(state.edge),
        where=state.rho[:, None] > 1e-15,
    )
    return _row_normalize(values, fallback)


def _score_moment_target(
    rho: FloatArray,
    edge: FloatArray,
    population: int,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Independent-edge target for ``W``, ``S2``, and ``S4``."""

    undirected = np.maximum(edge + edge.T, 0.0)
    per_center = np.divide(
        undirected,
        rho[None, :],
        out=np.zeros_like(undirected),
        where=rho[None, :] > 1e-15,
    )
    wedge = np.einsum("ir,jr,r->irj", per_center, per_center, rho, optimize=True)
    score1 = wedge.sum(axis=1)

    pair_count = population * np.outer(rho, rho)
    adjacency_probability = np.divide(
        undirected,
        population * np.outer(rho, rho),
        out=np.zeros_like(undirected),
        where=np.outer(rho, rho) > 1e-15,
    )
    adjacency_probability = np.clip(adjacency_probability, 0.0, 1.0)
    common_probability = (
        adjacency_probability[:, None, :] * adjacency_probability[None, :, :]
    )
    counts = population * rho
    power_sums = np.stack(
        [
            np.einsum("ijr,r->ij", common_probability**power, counts, optimize=True)
            for power in range(1, 5)
        ]
    )
    p1, p2, p3, p4 = power_sums
    factorial2 = p1 * p1 - p2
    factorial3 = p1**3 - 3.0 * p1 * p2 + 2.0 * p3
    factorial4 = p1**4 - 6.0 * p1 * p1 * p2 + 3.0 * p2 * p2 + 8.0 * p1 * p3 - 6.0 * p4
    score2 = pair_count * np.maximum(factorial2 + p1, 0.0)
    score4 = pair_count * np.maximum(
        factorial4 + 6.0 * factorial3 + 7.0 * factorial2 + p1, 0.0
    )
    return wedge, np.maximum(score2, score1), np.maximum(score4, score2)


def initialize_terminal_state(
    parameters: TerminalGeneratorParameters,
) -> TerminalState:
    parameters.validate()
    rng = np.random.default_rng(parameters.seed)
    counts = rng.multinomial(
        parameters.population,
        np.full(parameters.grid_size, 1.0 / parameters.grid_size),
    )
    rho = counts / parameters.population
    edge_counts = np.zeros((parameters.grid_size, parameters.grid_size), dtype=int)
    for source, count in enumerate(counts):
        if count:
            edge_counts[source] = rng.multinomial(
                parameters.mean_degree * int(count), rho
            )
    edge = edge_counts / parameters.population
    if parameters.state_level == "pair":
        return TerminalState(rho, edge)
    wedge, score2, score4 = _score_moment_target(rho, edge, parameters.population)
    return TerminalState(rho, edge, wedge, score2, score4)


def _candidate_counts(
    parameters: TerminalGeneratorParameters,
    state: TerminalState,
    neighbors: FloatArray,
) -> FloatArray:
    candidates = parameters.population * state.rho[None, :] - (
        parameters.mean_degree * neighbors
    )
    candidates -= np.eye(state.rho.size)
    return np.maximum(candidates, 0.0)


def recommendation_kernel(
    parameters: TerminalGeneratorParameters,
    x: FloatArray,
    state: TerminalState,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return recommendation, neighbor, and available-candidate kernels."""

    neighbors = _neighbors(state, parameters.mean_degree)
    candidates = _candidate_counts(parameters, state, neighbors)
    fallback = _row_normalize(candidates, np.broadcast_to(state.rho, state.edge.shape))
    if parameters.recsys == "random":
        raw = candidates
    elif parameters.recsys == "opinion_random":
        score = np.maximum(
            1.0 - np.abs(x[:, None] - x[None, :]) / parameters.opinion_tolerance,
            0.0,
        )
        raw = candidates * score**parameters.recommendation_steepness
    elif state.lifted:
        assert state.wedge is not None and state.score4 is not None
        score_mass = (
            state.wedge.sum(axis=1)
            if parameters.recommendation_steepness == 1.0
            else state.score4
        )
        available_fraction = np.divide(
            candidates,
            parameters.population * state.rho[None, :],
            out=np.zeros_like(candidates),
            where=state.rho[None, :] > 1e-15,
        )
        raw = score_mass * available_fraction
    else:
        # L0 closes the random score by the corresponding expected overlap.
        overlap = np.maximum(neighbors @ neighbors.T, 0.0)
        raw = candidates * overlap**parameters.recommendation_steepness
    return _row_normalize(raw, fallback), neighbors, candidates


def _deposit_hat(
    probability: FloatArray,
    x: FloatArray,
    value: float,
    weight: float,
) -> None:
    value = float(np.clip(value, -1.0, 1.0))
    if value <= x[0]:
        probability[0] += weight
    elif value >= x[-1]:
        probability[-1] += weight
    else:
        right = int(np.searchsorted(x, value, side="right"))
        left = right - 1
        fraction = (value - x[left]) / (x[right] - x[left])
        probability[left] += weight * (1.0 - fraction)
        probability[right] += weight * fraction


def _conditional_moments(
    probability: FloatArray, x: FloatArray
) -> tuple[float, float, float]:
    total = float(probability.sum())
    if total <= 1e-15:
        return 0.0, 0.0, 0.0
    conditional = probability / total
    mean = float(conditional @ x)
    variance = float(conditional @ ((x - mean) ** 2))
    return min(total, 1.0), mean, max(variance, 0.0)


def _opinion_transition(
    parameters: TerminalGeneratorParameters,
    x: FloatArray,
    state: TerminalState,
    neighbors: FloatArray,
    recommendations: FloatArray,
    candidates: FloatArray,
) -> FloatArray:
    size = x.size
    transition = np.zeros((size, size), dtype=float)
    for source in range(size):
        concordant = np.abs(x - x[source]) <= parameters.epsilon
        p_n, mean_n, var_n = _conditional_moments(neighbors[source] * concordant, x)
        p_r, mean_r, var_r = _conditional_moments(
            recommendations[source] * concordant, x
        )
        pmf_n = binom.pmf(
            np.arange(parameters.mean_degree + 1), parameters.mean_degree, p_n
        )
        pmf_r = binom.pmf(
            np.arange(parameters.recsys_count + 1), parameters.recsys_count, p_r
        )
        available = float(candidates[source, concordant].sum())
        output = transition[source]
        for count_n, probability_n in enumerate(pmf_n):
            if probability_n < 1e-14:
                continue
            for count_r, probability_r in enumerate(pmf_r):
                weight = float(probability_n * probability_r)
                if weight < 1e-14:
                    continue
                count = count_n + count_r
                if count == 0:
                    output[source] += weight
                    continue
                target_mean = (count_n * mean_n + count_r * mean_r) / count
                finite = 1.0
                if count_r > 0 and available > 1:
                    finite = max((available - count_r) / (available - 1.0), 0.0)
                variance = (count_n * var_n + count_r * var_r * finite) / count**2
                mean = (1.0 - parameters.influence) * x[
                    source
                ] + parameters.influence * target_mean
                standard_deviation = parameters.influence * math.sqrt(variance)
                if standard_deviation <= 1e-8:
                    _deposit_hat(output, x, mean, weight)
                else:
                    for node, quadrature_weight in zip(
                        _GH_NODES, _GH_WEIGHTS, strict=True
                    ):
                        _deposit_hat(
                            output,
                            x,
                            mean + math.sqrt(2.0) * standard_deviation * float(node),
                            weight * float(quadrature_weight),
                        )
        output[:] = np.maximum(output, 0.0)
        output[:] /= output.sum()
    return transition


def _sample_transition(
    parameters: TerminalGeneratorParameters,
    state: TerminalState,
    transition: FloatArray,
    rng: np.random.Generator,
) -> FloatArray:
    counts = np.rint(parameters.population * state.rho).astype(int)
    counts[int(np.argmax(state.rho))] += parameters.population - int(counts.sum())
    sampled = np.zeros_like(transition)
    for source, count in enumerate(counts):
        if count > 0:
            sampled[source] = rng.multinomial(count, transition[source]) / count
        else:
            sampled[source, source] = 1.0
    return sampled


def _sample_edges(
    parameters: TerminalGeneratorParameters,
    rho: FloatArray,
    expected: FloatArray,
    rng: np.random.Generator,
) -> FloatArray:
    counts = np.rint(parameters.population * rho).astype(int)
    counts[int(np.argmax(rho))] += parameters.population - int(counts.sum())
    output = np.zeros_like(expected)
    for source, count in enumerate(counts):
        edge_count = parameters.mean_degree * int(count)
        if edge_count:
            weights = np.maximum(expected[source], 0.0)
            probability = weights / weights.sum() if weights.sum() else rho
            output[source] = rng.multinomial(edge_count, probability)
    return output / parameters.population


def _tau_leap_rewiring(
    parameters: TerminalGeneratorParameters,
    x: FloatArray,
    state: TerminalState,
    neighbors: FloatArray,
    recommendations: FloatArray,
    rng: np.random.Generator,
) -> tuple[FloatArray, int, float, float]:
    concordant = np.abs(x[:, None] - x[None, :]) <= parameters.epsilon
    discordant = ~concordant
    p_discordant = np.sum(discordant * neighbors, axis=1)
    p_rec_concordant = np.sum(concordant * recommendations, axis=1)
    eligibility = (1.0 - (1.0 - p_discordant) ** parameters.mean_degree) * (
        1.0 - (1.0 - p_rec_concordant) ** parameters.recsys_count
    )
    counts = np.rint(parameters.population * state.rho).astype(int)
    counts[int(np.argmax(state.rho))] += parameters.population - int(counts.sum())
    output = state.edge.copy()
    total_events = attempted = capped = 0
    for source, count in enumerate(counts):
        probability = min(parameters.rewiring * eligibility[source], 1.0)
        requested = int(rng.binomial(int(count), probability))
        if requested <= 0:
            continue
        attempted += requested
        available = np.floor(
            parameters.population * np.maximum(output[source] * discordant[source], 0.0)
            + 1e-10
        ).astype(int)
        actual = min(requested, int(available.sum()))
        capped += requested - actual
        if actual <= 0:
            continue
        loss = rng.multivariate_hypergeometric(available, actual)
        gain_weights = recommendations[source] * concordant[source]
        if gain_weights.sum() <= 1e-15:
            continue
        gain = rng.multinomial(actual, gain_weights / gain_weights.sum())
        output[source] += (gain - loss) / parameters.population
        total_events += actual
    residual = float(
        parameters.population * parameters.rewiring * np.sum(state.rho * eligibility)
    )
    return output, total_events, capped / attempted if attempted else 0.0, residual


def _update_moments_after_rewiring(
    parameters: TerminalGeneratorParameters,
    state: TerminalState,
    edge_after: FloatArray,
) -> tuple[FloatArray | None, FloatArray | None, FloatArray | None]:
    if not state.lifted:
        return None, None, None
    assert state.wedge is not None
    assert state.score2 is not None and state.score4 is not None
    target_wedge, target2, target4 = _score_moment_target(
        state.rho, edge_after, parameters.population
    )
    before = state.edge + state.edge.T
    change = np.abs(edge_after + edge_after.T - before)
    center_change = np.divide(
        change.sum(axis=0),
        before.sum(axis=0),
        out=np.zeros(state.rho.size),
        where=before.sum(axis=0) > 1e-15,
    )
    center_change = np.clip(center_change, 0.0, 1.0)
    persistence = (1.0 - center_change) ** 2
    wedge = (
        persistence[None, :, None] * state.wedge
        + (1.0 - persistence[None, :, None]) * target_wedge
    )
    global_persistence = (1.0 - float(state.rho @ center_change)) ** 2
    score2 = global_persistence * state.score2 + (1.0 - global_persistence) * target2
    score4 = global_persistence * state.score4 + (1.0 - global_persistence) * target4
    return wedge, score2, score4


def _transport_state(
    parameters: TerminalGeneratorParameters,
    state: TerminalState,
    transition: FloatArray,
    rng: np.random.Generator,
) -> TerminalState:
    rho = state.rho @ transition
    expected_edge = transition.T @ state.edge @ transition
    edge = _sample_edges(parameters, rho, expected_edge, rng)
    if not state.lifted:
        return TerminalState(rho, edge)
    assert state.wedge is not None
    assert state.score2 is not None and state.score4 is not None
    wedge = np.einsum(
        "ia,rb,jc,irj->abc",
        transition,
        transition,
        transition,
        state.wedge,
        optimize=True,
    )
    score2 = transition.T @ state.score2 @ transition
    score4 = transition.T @ state.score4 @ transition
    return TerminalState(rho, edge, wedge, score2, score4)


def validate_terminal_state(
    parameters: TerminalGeneratorParameters, state: TerminalState
) -> None:
    arrays = [state.rho, state.edge]
    if state.lifted:
        assert state.wedge is not None
        assert state.score2 is not None and state.score4 is not None
        arrays.extend([state.wedge, state.score2, state.score4])
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise FloatingPointError("terminal generator state is non-finite")
    if min(float(value.min()) for value in arrays) < -1e-9:
        raise FloatingPointError("terminal generator state is negative")
    if abs(float(state.rho.sum()) - 1.0) > 1e-9:
        raise FloatingPointError("node mass is not conserved")
    error = np.max(np.abs(state.edge.sum(axis=1) - parameters.mean_degree * state.rho))
    if error > 1e-8:
        raise FloatingPointError(f"fixed out-degree invariant failed: {error}")


def absorbing_peak_category(
    rho: FloatArray,
    x: FloatArray,
    population: int,
    epsilon: float,
) -> str | None:
    occupied = np.flatnonzero(rho >= 0.5 / population)
    if occupied.size == 0:
        return None
    components: list[list[int]] = [[int(occupied[0])]]
    for index in occupied[1:]:
        if x[int(index)] - x[components[-1][-1]] > epsilon:
            components.append([int(index)])
        else:
            components[-1].append(int(index))
    if any(x[item[-1]] - x[item[0]] > epsilon for item in components):
        return None
    count = len(components)
    return f"k{count}" if count <= 3 else "k4plus"


def terminal_peak_category(
    rho: FloatArray,
    x: FloatArray,
    population: int,
    epsilon: float,
) -> str:
    axis = np.linspace(-1.0, 1.0, 1001)
    mean = float(rho @ x)
    variance = float(rho @ ((x - mean) ** 2))
    bandwidth = max(0.1, math.sqrt(max(variance, 1e-16)) * population**-0.2)
    density = np.sum(
        rho[:, None] * norm.pdf(axis[None, :], x[:, None], bandwidth), axis=0
    )
    peaks, _ = find_peaks(
        density,
        height=float(density.max()) * 0.1,
        distance=math.floor(epsilon / (axis[1] - axis[0])) + 1,
    )
    count = max(int(peaks.size), 1)
    return f"k{count}" if count <= 3 else "k4plus"


def _fast_absorb(
    parameters: TerminalGeneratorParameters,
    x: FloatArray,
    state: TerminalState,
    rng: np.random.Generator,
) -> tuple[TerminalState, int, int, float, bool, float]:
    events = 0
    cap_sum = 0.0
    zero_checks = 0
    residual = math.inf
    for substep in range(1, parameters.fast_max_steps + 1):
        recommendations, neighbors, _ = recommendation_kernel(parameters, x, state)
        edge, count, cap, residual = _tau_leap_rewiring(
            parameters, x, state, neighbors, recommendations, rng
        )
        wedge, score2, score4 = _update_moments_after_rewiring(parameters, state, edge)
        state = TerminalState(state.rho, edge, wedge, score2, score4)
        validate_terminal_state(parameters, state)
        events += count
        cap_sum += cap
        zero_checks = zero_checks + 1 if count == 0 else 0
        if residual <= 1e-12 or (
            zero_checks >= parameters.fast_zero_checks and residual < 0.25
        ):
            return state, substep, events, cap_sum, False, residual
    return state, parameters.fast_max_steps, events, cap_sum, True, residual


def run_terminal_generator(
    parameters: TerminalGeneratorParameters,
) -> TerminalGeneratorResult:
    """Sample one terminal-generator path for one factorial cell."""

    parameters.validate()
    x = opinion_grid(parameters.grid_size)
    state = initialize_terminal_state(parameters)
    rng = np.random.default_rng(parameters.seed + 3_000_000_000)
    total_events = total_substeps = max_hits = 0
    cap_sum = 0.0
    cap_denominator = 0
    residual = math.nan
    ratio = parameters.rewiring / parameters.influence
    fast_applied = (
        parameters.timescale == "fast_slow"
        and ratio >= parameters.fast_slow_ratio_threshold
    )
    raw_category = terminal_peak_category(
        state.rho, x, parameters.population, parameters.epsilon
    )

    for step in range(1, parameters.max_steps + 1):
        if fast_applied:
            state, substeps, events, fast_cap, hit, residual = _fast_absorb(
                parameters, x, state, rng
            )
            total_substeps += substeps
            total_events += events
            cap_sum += fast_cap
            cap_denominator += substeps
            max_hits += int(hit)

        recommendations, neighbors, candidates = recommendation_kernel(
            parameters, x, state
        )
        # In the unsplit kernel, opinion and rewiring channels are sampled
        # from the same old state.  In the fast-slow kernel this old state is
        # the conditionally absorbed fast state.
        transition = _opinion_transition(
            parameters, x, state, neighbors, recommendations, candidates
        )
        if not fast_applied:
            edge, events, cap, residual = _tau_leap_rewiring(
                parameters, x, state, neighbors, recommendations, rng
            )
            wedge, score2, score4 = _update_moments_after_rewiring(
                parameters, state, edge
            )
            state = TerminalState(state.rho, edge, wedge, score2, score4)
            total_events += events
            cap_sum += cap
            cap_denominator += 1
        sampled = _sample_transition(parameters, state, transition, rng)
        state = _transport_state(parameters, state, sampled, rng)
        validate_terminal_state(parameters, state)
        absorbing = absorbing_peak_category(
            state.rho, x, parameters.population, parameters.epsilon
        )
        raw_category = terminal_peak_category(
            state.rho, x, parameters.population, parameters.epsilon
        )
        if absorbing is not None:
            return TerminalGeneratorResult(
                parameters,
                absorbing,
                raw_category,
                True,
                step,
                total_events,
                cap_sum / max(cap_denominator, 1),
                fast_applied,
                total_substeps,
                max_hits,
                residual,
                state,
            )

    return TerminalGeneratorResult(
        parameters,
        "censored",
        raw_category,
        False,
        parameters.max_steps,
        total_events,
        cap_sum / max(cap_denominator, 1),
        fast_applied,
        total_substeps,
        max_hits,
        residual,
        state,
    )


__all__ = [
    "StateLevel",
    "TerminalGeneratorParameters",
    "TerminalGeneratorResult",
    "TerminalState",
    "Timescale",
    "absorbing_peak_category",
    "initialize_terminal_state",
    "opinion_grid",
    "recommendation_kernel",
    "run_terminal_generator",
    "terminal_peak_category",
    "validate_terminal_state",
]

"""Connect periodic HK spectra to representative nonlinear states.

The experiment separately excites modes m=1, 2, and 3 for the Random and
Opinion spectra at the paper parameters. A dealiased pseudospectral RK4 solver
evolves the same periodic continuum PDE used in the linearization inside each
mode's exact symmetry sector. With q=0 the product manifold is invariant, so
the edge density is reconstructed exactly as E=kbar*rho(x)*rho(y).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from ehk.common.plotting import setup_paper_params
from theory.mesoscopic.cli_utils import write_run_metadata
from theory.mesoscopic.spectrum_check import (
    _measured_growth_rate,
    _opinion_growth_rate,
    _random_growth_rate,
)
from theory.paths import MESOSCOPIC_OUTPUT


@dataclass(frozen=True)
class PeriodicParameters:
    length: float = 2.0
    epsilon: float = 0.45
    influence: float = 0.05
    mean_degree: float = 15.0
    recsys_count: int = 10
    bandwidth: float = 0.1
    diffusion: float = 1e-5
    grid_size: int = 1024
    mode: int = 2
    amplitude: float = 0.08
    dt: float = 0.01
    max_time: float = 1000.0
    check_every: int = 200
    minimum_time: float = 100.0
    steady_tolerance: float = 2e-8

    def validate(self) -> None:
        values = vars(self)
        if any(not np.isfinite(value) for value in values.values()):
            raise ValueError("all steady-state parameters must be finite")
        if self.grid_size < 64 or self.grid_size % 2:
            raise ValueError("grid_size must be an even integer >= 64")
        if self.mode < 1 or self.mode >= self.grid_size // 3:
            raise ValueError("mode must lie in the dealiased Fourier band")
        if not 0 < self.amplitude < 1:
            raise ValueError("amplitude must lie in (0, 1)")
        if self.dt <= 0 or self.max_time <= 0:
            raise ValueError("dt and max_time must be positive")


@dataclass(frozen=True)
class PeriodicOperators:
    x: np.ndarray
    dx: float
    wave_numbers: np.ndarray
    dealias: np.ndarray
    symmetry: np.ndarray
    concordance_fft: np.ndarray
    displacement_fft: np.ndarray
    opinion_normalizer_fft: np.ndarray
    opinion_concordance_fft: np.ndarray
    opinion_displacement_fft: np.ndarray


@dataclass(frozen=True)
class SteadyState:
    recommender: str
    time: float
    density: np.ndarray
    residual_l1: float
    selected_growth_analytic: float
    selected_growth_measured: float
    selected_growth_solver: float
    converged: bool


def _build_operators(params: PeriodicParameters) -> PeriodicOperators:
    dx = params.length / params.grid_size
    x = -params.length / 2 + np.arange(params.grid_size) * dx
    displacement = np.arange(params.grid_size, dtype=float) * dx
    displacement[displacement > params.length / 2] -= params.length
    concordant = np.abs(displacement) <= params.epsilon
    opinion_score = np.exp(-0.5 * (displacement / params.bandwidth) ** 2)
    wave_numbers = 2 * np.pi * np.fft.fftfreq(params.grid_size, d=dx)
    mode_numbers = np.fft.fftfreq(params.grid_size) * params.grid_size
    dealias = np.abs(mode_numbers) <= params.grid_size // 3
    integer_modes = np.rint(mode_numbers).astype(int)
    symmetry = integer_modes % params.mode == 0

    def correlation_fft(kernel: np.ndarray) -> np.ndarray:
        return np.conj(np.fft.fft(kernel))

    return PeriodicOperators(
        x=x,
        dx=dx,
        wave_numbers=wave_numbers,
        dealias=dealias,
        symmetry=symmetry,
        concordance_fft=correlation_fft(concordant.astype(float)),
        displacement_fft=correlation_fft(concordant * displacement),
        opinion_normalizer_fft=correlation_fft(opinion_score),
        opinion_concordance_fft=correlation_fft(concordant * opinion_score),
        opinion_displacement_fft=correlation_fft(
            concordant * displacement * opinion_score
        ),
    )


def _correlate(
    kernel_fft: np.ndarray,
    density_fft: np.ndarray,
    dx: float,
) -> np.ndarray:
    return np.fft.ifft(kernel_fft * density_fft).real * dx


def _velocity(
    density: np.ndarray,
    recommender: str,
    params: PeriodicParameters,
    operators: PeriodicOperators,
) -> np.ndarray:
    density_fft = np.fft.fft(density)
    neighbor_mass = _correlate(operators.concordance_fft, density_fft, operators.dx)
    neighbor_displacement = _correlate(
        operators.displacement_fft, density_fft, operators.dx
    )
    if recommender == "random":
        numerator = neighbor_displacement
        denominator = neighbor_mass
    elif recommender == "opinion":
        opinion_normalizer = _correlate(
            operators.opinion_normalizer_fft, density_fft, operators.dx
        )
        opinion_mass = np.divide(
            _correlate(
                operators.opinion_concordance_fft,
                density_fft,
                operators.dx,
            ),
            opinion_normalizer,
        )
        opinion_displacement = np.divide(
            _correlate(
                operators.opinion_displacement_fft,
                density_fft,
                operators.dx,
            ),
            opinion_normalizer,
        )
        numerator = (
            params.mean_degree * neighbor_displacement
            + params.recsys_count * opinion_displacement
        )
        denominator = (
            params.mean_degree * neighbor_mass + params.recsys_count * opinion_mass
        )
    else:
        raise ValueError(f"unsupported recommender: {recommender}")
    return params.influence * np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-14,
    )


def _rhs(
    density: np.ndarray,
    recommender: str,
    params: PeriodicParameters,
    operators: PeriodicOperators,
) -> np.ndarray:
    velocity = _velocity(density, recommender, params, operators)
    flux_fft = np.fft.fft(density * velocity)
    flux_fft[~operators.dealias] = 0.0
    density_fft = np.fft.fft(density)
    rhs_fft = (
        -1j * operators.wave_numbers * flux_fft
        - params.diffusion * operators.wave_numbers**2 * density_fft
    )
    # The m-fold periodic subspace is an exact invariant of the continuum
    # equation. Enforcing it removes exponentially amplified roundoff in
    # lower modes and isolates the nonlinear branch selected by mode m.
    rhs_fft[~operators.symmetry] = 0.0
    return np.fft.ifft(rhs_fft).real


def _initial_density(
    params: PeriodicParameters,
    operators: PeriodicOperators,
    *,
    amplitude: float | None = None,
) -> np.ndarray:
    relative_amplitude = params.amplitude if amplitude is None else amplitude
    kappa = 2 * np.pi * params.mode / params.length
    phase_shift = params.length / (2 * params.mode)
    return (1 / params.length) * (
        1 + relative_amplitude * np.cos(kappa * (operators.x + phase_shift))
    )


def _rk4_step(
    density: np.ndarray,
    recommender: str,
    params: PeriodicParameters,
    operators: PeriodicOperators,
) -> np.ndarray:
    k1 = _rhs(density, recommender, params, operators)
    k2 = _rhs(
        density + 0.5 * params.dt * k1,
        recommender,
        params,
        operators,
    )
    k3 = _rhs(
        density + 0.5 * params.dt * k2,
        recommender,
        params,
        operators,
    )
    k4 = _rhs(
        density + params.dt * k3,
        recommender,
        params,
        operators,
    )
    return density + params.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def _selected_growth(
    recommender: str,
    params: PeriodicParameters,
) -> tuple[float, float]:
    kappa = 2 * np.pi * params.mode / params.length
    common = {
        "epsilon": params.epsilon,
        "influence": params.influence,
        "diffusion": params.diffusion,
    }
    if recommender == "random":
        analytic = _random_growth_rate(kappa, **common)
    else:
        analytic = _opinion_growth_rate(
            kappa,
            length=params.length,
            mean_degree=params.mean_degree,
            recsys_count=params.recsys_count,
            bandwidth=params.bandwidth,
            **common,
        )
    measured = _measured_growth_rate(
        kappa,
        recommender=recommender,
        length=params.length,
        mean_degree=params.mean_degree,
        recsys_count=params.recsys_count,
        bandwidth=params.bandwidth,
        grid_size=1024,
        amplitude=1e-6,
        **common,
    )
    return analytic, measured


def _solver_growth(
    recommender: str,
    params: PeriodicParameters,
    operators: PeriodicOperators,
) -> float:
    """Project this solver's infinitesimal RHS onto the selected mode."""
    relative_amplitude = 1e-6
    rho0 = 1 / params.length
    density = _initial_density(
        params,
        operators,
        amplitude=relative_amplitude,
    )
    perturbation = (density / rho0 - 1) / relative_amplitude
    projected_rhs = (
        2
        * operators.dx
        / params.length
        * np.sum(_rhs(density, recommender, params, operators) * perturbation)
    )
    return float(projected_rhs / (rho0 * relative_amplitude))


def _solve(
    recommender: str,
    params: PeriodicParameters,
    operators: PeriodicOperators,
) -> SteadyState:
    density = _initial_density(params, operators)
    steps = int(np.ceil(params.max_time / params.dt))
    residual_l1 = float("inf")
    time = 0.0
    for step in range(1, steps + 1):
        density = _rk4_step(density, recommender, params, operators)
        time = step * params.dt
        if step % params.check_every:
            continue
        if not np.all(np.isfinite(density)) or density.min() < -1e-8:
            raise FloatingPointError(
                f"{recommender} periodic solver lost nonnegativity at "
                f"t={time:g} (minimum={density.min():.3e})"
            )
        mass_error = abs(float(density.sum() * operators.dx) - 1.0)
        if mass_error > 2e-10:
            raise FloatingPointError(f"{recommender} mass error {mass_error:.3e}")
        residual_l1 = float(
            np.sum(np.abs(_rhs(density, recommender, params, operators))) * operators.dx
        )
        if time >= params.minimum_time and residual_l1 < params.steady_tolerance:
            break
    analytic, measured = _selected_growth(recommender, params)
    solver_growth = _solver_growth(
        recommender,
        params,
        operators,
    )
    if abs(solver_growth - analytic) > 5e-4:
        raise FloatingPointError(
            f"{recommender} solver growth rate differs from the analytic "
            f"spectrum by {abs(solver_growth - analytic):.3e}"
        )
    return SteadyState(
        recommender=recommender,
        time=time,
        density=density,
        residual_l1=residual_l1,
        selected_growth_analytic=analytic,
        selected_growth_measured=measured,
        selected_growth_solver=solver_growth,
        converged=residual_l1 < params.steady_tolerance,
    )


def _solve_mode(
    params: PeriodicParameters,
) -> tuple[int, PeriodicOperators, dict[str, SteadyState]]:
    """Solve both kernels for one mode; safe for process-level parallelism."""
    operators = _build_operators(params)
    states = {
        recommender: _solve(recommender, params, operators)
        for recommender in ("random", "opinion")
    }
    return params.mode, operators, states


def _plot(
    states: dict[tuple[int, str], SteadyState],
    parameters: dict[int, PeriodicParameters],
    operators: dict[int, PeriodicOperators],
    modes: tuple[int, ...],
    output_path: Path,
) -> None:
    setup_paper_params()
    figure = plt.figure(
        figsize=(7.1, 2.0 + 1.55 * len(modes)),
        constrained_layout=True,
    )
    grid = figure.add_gridspec(
        1 + len(modes),
        4,
        width_ratios=(1.25, 1.0, 1.25, 1.0),
        height_ratios=(1.0, *([0.92] * len(modes))),
    )
    spectrum_axes = {
        "random": figure.add_subplot(grid[0, :2]),
        "opinion": figure.add_subplot(grid[0, 2:]),
    }
    density_axes: dict[tuple[int, str], plt.Axes] = {}
    edge_axes: dict[tuple[int, str], plt.Axes] = {}
    for row, mode in enumerate(modes, start=1):
        density_axes[mode, "random"] = figure.add_subplot(grid[row, 0])
        edge_axes[mode, "random"] = figure.add_subplot(grid[row, 1])
        density_axes[mode, "opinion"] = figure.add_subplot(grid[row, 2])
        edge_axes[mode, "opinion"] = figure.add_subplot(grid[row, 3])

    names = {"random": "Random spectrum", "opinion": "Opinion spectrum"}
    colors = {"random": "#0072B2", "opinion": "#D55E00"}
    mode_markers = {1: "o", 2: "s", 3: "^"}
    reference = parameters[modes[0]]
    mode_numbers = np.arange(1, 9)
    kappas = 2 * np.pi * mode_numbers / reference.length
    curve_kappa = np.linspace(0, kappas[-1] * 1.04, 350)
    common = {
        "epsilon": reference.epsilon,
        "influence": reference.influence,
        "diffusion": reference.diffusion,
    }

    edge_relative = {
        key: np.maximum(
            parameters[key[0]].length ** 2 * np.outer(state.density, state.density),
            1e-4,
        )
        for key, state in states.items()
    }
    edge_max = max(float(values.max()) for values in edge_relative.values())
    edge_norm = LogNorm(vmin=1e-4, vmax=edge_max)

    for recommender in ("random", "opinion"):
        if recommender == "random":
            curve = np.asarray(
                [_random_growth_rate(value, **common) for value in curve_kappa]
            )
            sampled = np.asarray(
                [
                    _measured_growth_rate(
                        value,
                        recommender=recommender,
                        length=reference.length,
                        mean_degree=reference.mean_degree,
                        recsys_count=reference.recsys_count,
                        bandwidth=reference.bandwidth,
                        grid_size=1024,
                        amplitude=1e-6,
                        **common,
                    )
                    for value in kappas
                ]
            )
        else:
            curve = np.asarray(
                [
                    _opinion_growth_rate(
                        value,
                        length=reference.length,
                        mean_degree=reference.mean_degree,
                        recsys_count=reference.recsys_count,
                        bandwidth=reference.bandwidth,
                        **common,
                    )
                    for value in curve_kappa
                ]
            )
            sampled = np.asarray(
                [
                    _measured_growth_rate(
                        value,
                        recommender=recommender,
                        length=reference.length,
                        mean_degree=reference.mean_degree,
                        recsys_count=reference.recsys_count,
                        bandwidth=reference.bandwidth,
                        grid_size=1024,
                        amplitude=1e-6,
                        **common,
                    )
                    for value in kappas
                ]
            )
        spectrum_axis = spectrum_axes[recommender]
        spectrum_axis.axhline(0, color="0.45", lw=0.7)
        spectrum_axis.plot(
            curve_kappa * reference.epsilon,
            curve,
            color=colors[recommender],
        )
        spectrum_axis.scatter(
            kappas * reference.epsilon,
            sampled,
            facecolors="white",
            edgecolors=colors[recommender],
            s=20,
            zorder=3,
        )
        for mode in modes:
            selected = mode - 1
            selected_x = kappas[selected] * reference.epsilon
            selected_y = sampled[selected]
            spectrum_axis.scatter(
                [selected_x],
                [selected_y],
                marker=mode_markers.get(mode, "D"),
                facecolors=colors[recommender],
                edgecolors="black",
                linewidths=0.5,
                s=34,
                zorder=4,
            )
            spectrum_axis.annotate(
                rf"$m={mode}$",
                (selected_x, selected_y),
                xytext=(0, -10 if mode == 2 else 7),
                textcoords="offset points",
                ha="center",
                va="top" if mode == 2 else "bottom",
                fontsize=6.5,
            )
        spectrum_axis.set_title(names[recommender])
        spectrum_axis.set_xlabel(r"wave number $z=\kappa\epsilon$")
        if recommender == "random":
            spectrum_axis.set_ylabel(r"growth rate $\lambda$")
        spectrum_axis.grid(alpha=0.18, linewidth=0.5)

    images = []
    for row, mode in enumerate(modes):
        params = parameters[mode]
        mode_operators = operators[mode]
        initial_density = _initial_density(params, mode_operators)
        density_max = max(
            float(states[mode, recommender].density.max())
            for recommender in ("random", "opinion")
        )
        for recommender in ("random", "opinion"):
            state = states[mode, recommender]
            density_axis = density_axes[mode, recommender]
            density_axis.plot(
                mode_operators.x,
                initial_density,
                color="0.5",
                ls="--",
                lw=0.8,
            )
            density_axis.plot(
                mode_operators.x,
                state.density,
                color=colors[recommender],
                lw=1.1,
            )
            status = "steady" if state.converged else "late"
            density_axis.text(
                0.5,
                0.9,
                rf"{status}, $t={state.time:g}$",
                transform=density_axis.transAxes,
                ha="center",
                va="top",
                fontsize=6.2,
            )
            density_axis.set_xlim(-1, 1)
            density_axis.set_ylim(0, 1.05 * density_max)
            density_axis.set_xticks((-1, 0, 1))
            if row == len(modes) - 1:
                density_axis.set_xlabel(r"opinion $x$")
            if recommender == "random":
                density_axis.set_ylabel(rf"$m={mode}$" + "\n" + r"$\rho(x)$")
            density_axis.grid(alpha=0.18, linewidth=0.5)

            edge_axis = edge_axes[mode, recommender]
            images.append(
                edge_axis.imshow(
                    edge_relative[mode, recommender],
                    origin="lower",
                    extent=(-1, 1, -1, 1),
                    aspect="equal",
                    cmap="cividis",
                    norm=edge_norm,
                    interpolation="nearest",
                )
            )
            edge_axis.set_xticks((-1, 0, 1))
            edge_axis.set_yticks((-1, 0, 1))
            if row == len(modes) - 1:
                edge_axis.set_xlabel(r"source $x$")
            if recommender == "random":
                edge_axis.set_ylabel(r"target $y$")

    state_title_size = 9.5
    density_axes[modes[0], "random"].set_title(
        r"Random $\rho$", fontsize=state_title_size
    )
    edge_axes[modes[0], "random"].set_title(
        r"Random $L^2E/\bar{k}$", fontsize=state_title_size
    )
    density_axes[modes[0], "opinion"].set_title(
        r"Opinion $\rho$", fontsize=state_title_size
    )
    edge_axes[modes[0], "opinion"].set_title(
        r"Opinion $L^2E/\bar{k}$", fontsize=state_title_size
    )
    figure.colorbar(
        images[-1],
        ax=list(edge_axes.values()),
        shrink=0.9,
        label="relative edge density",
    )
    for suffix in (".pdf", ".png"):
        figure.savefig(output_path.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=1024)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--mode", type=int, help="legacy single-mode run")
    mode_group.add_argument("--modes", type=int, nargs="+")
    parser.add_argument("--amplitude", type=float, default=0.08)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--max-time", type=float, default=1000.0)
    parser.add_argument("--steady-tolerance", type=float, default=2e-8)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MESOSCOPIC_OUTPUT.resolve() / "spectrum_steady_states",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_modes = args.modes or (
        [args.mode] if args.mode is not None else [1, 2, 3]
    )
    modes = tuple(dict.fromkeys(requested_modes))
    if args.jobs < 1:
        raise ValueError("jobs must be a positive integer")
    parameters = {
        mode: PeriodicParameters(
            grid_size=args.grid_size,
            mode=mode,
            amplitude=args.amplitude,
            dt=args.dt,
            max_time=args.max_time,
            steady_tolerance=args.steady_tolerance,
        )
        for mode in modes
    }
    for params in parameters.values():
        params.validate()
    parameter_sequence = [parameters[mode] for mode in modes]
    if args.jobs == 1:
        mode_results = [_solve_mode(params) for params in parameter_sequence]
    else:
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(modes))) as executor:
            mode_results = list(executor.map(_solve_mode, parameter_sequence))
    operators = {
        mode: mode_operators for mode, mode_operators, _mode_states in mode_results
    }
    states = {
        (mode, recommender): state
        for mode, _mode_operators, mode_states in mode_results
        for recommender, state in mode_states.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "steady_states.npz",
        modes=np.asarray(modes),
        x=operators[modes[0]].x,
        **{
            f"mode_{mode}_initial_density": _initial_density(
                parameters[mode], operators[mode]
            )
            for mode in modes
        },
        **{
            f"mode_{mode}_{recommender}_density": state.density
            for (mode, recommender), state in states.items()
        },
    )
    fieldnames = (
        "mode",
        "recommender",
        "status",
        "time",
        "residual_l1",
        "mass",
        "minimum_density",
        "maximum_density",
        "selected_growth_analytic",
        "selected_growth_measured",
        "selected_growth_solver",
    )
    with (args.output_dir / "steady_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for (mode, recommender), state in states.items():
            params = parameters[mode]
            mode_operators = operators[mode]
            writer.writerow(
                {
                    "mode": mode,
                    "recommender": recommender,
                    "status": "steady" if state.converged else "late_time",
                    "time": state.time,
                    "residual_l1": state.residual_l1,
                    "mass": float(state.density.sum() * mode_operators.dx),
                    "minimum_density": float(state.density.min()),
                    "maximum_density": float(state.density.max()),
                    "selected_growth_analytic": state.selected_growth_analytic,
                    "selected_growth_measured": state.selected_growth_measured,
                    "selected_growth_solver": state.selected_growth_solver,
                }
            )
    _plot(
        states,
        parameters,
        operators,
        modes,
        args.output_dir / "f_spectrum_steady_states",
    )
    reference_parameters = vars(parameters[modes[0]]).copy()
    reference_parameters.pop("mode")
    reference_parameters["modes"] = list(modes)
    write_run_metadata(
        args.output_dir / "run_metadata.json",
        analysis="periodic HK spectrum-to-nonlinear-state comparison",
        command=shlex.join(
            [
                sys.executable,
                "-m",
                "theory.mesoscopic.spectrum_steady_states",
                *sys.argv[1:],
            ]
        ),
        parameters=reference_parameters,
        configuration={
            "output_dir": str(args.output_dir.resolve()),
            "jobs": args.jobs,
        },
    )
    for (mode, recommender), state in states.items():
        print(
            f"m={mode} {recommender}: t={state.time:g}, "
            f"residual={state.residual_l1:.3e}, "
            f"rho_max={state.density.max():.3f}, "
            f"lambda_m={state.selected_growth_analytic:.6f}, "
            f"lambda_solver={state.selected_growth_solver:.6f}"
        )


if __name__ == "__main__":
    main()

"""Validate the HK Fourier growth rates against the nonlinear periodic PDE.

The check perturbs the uniform density by one cosine mode, evaluates the full
nonlinear velocity and continuity-equation right-hand side, and projects that
right-hand side back onto the injected mode.  Agreement as the perturbation
amplitude tends to zero tests the linearization without using the derived
growth-rate expression in the numerical measurement.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad

from ehk.common.plotting import setup_paper_params
from theory.mesoscopic.cli_utils import write_run_metadata
from theory.paths import MESOSCOPIC_OUTPUT


def _periodic_derivative(values: np.ndarray, length: float, order: int) -> np.ndarray:
    frequencies = 2 * np.pi * np.fft.fftfreq(
        values.size, d=length / values.size
    )
    multiplier = (1j * frequencies) ** order
    return np.fft.ifft(multiplier * np.fft.fft(values)).real


def _measured_growth_rate(
    kappa: float,
    *,
    recommender: str,
    length: float,
    epsilon: float,
    influence: float,
    mean_degree: float,
    recsys_count: int,
    bandwidth: float,
    diffusion: float,
    grid_size: int,
    amplitude: float,
) -> float:
    dx = length / grid_size
    x = np.arange(grid_size, dtype=float) * dx
    rho0 = 1 / length
    perturbation = np.cos(kappa * x)
    rho = rho0 * (1 + amplitude * perturbation)

    displacement = x[None, :] - x[:, None]
    displacement = (displacement + length / 2) % length - length / 2
    concordant = np.abs(displacement) <= epsilon
    if recommender == "random":
        recommendation = np.broadcast_to(rho, displacement.shape)
    elif recommender == "opinion":
        score = np.exp(-0.5 * (displacement / bandwidth) ** 2)
        unnormalized = score * rho[None, :]
        recommendation = unnormalized / (
            unnormalized.sum(axis=1, keepdims=True) * dx
        )
    else:
        raise ValueError(f"unsupported recommender: {recommender}")

    visible = mean_degree * rho[None, :] + recsys_count * recommendation
    denominator = (concordant * visible).sum(axis=1) * dx
    numerator = (concordant * displacement * visible).sum(axis=1) * dx
    velocity = influence * numerator / denominator
    rhs = -_periodic_derivative(rho * velocity, length, 1)
    rhs += diffusion * _periodic_derivative(rho, length, 2)
    projected_rhs = 2 * dx / length * np.sum(rhs * perturbation)
    return float(projected_rhs / (rho0 * amplitude))


def _random_growth_rate(
    kappa: float,
    *,
    epsilon: float,
    influence: float,
    diffusion: float,
) -> float:
    z = kappa * epsilon
    if abs(z) < 1e-10:
        attraction = influence * z * z / 3
    else:
        attraction = influence * (np.sin(z) - z * np.cos(z)) / z
    return float(attraction - diffusion * kappa * kappa)


def _opinion_growth_rate(
    kappa: float,
    *,
    length: float,
    epsilon: float,
    influence: float,
    mean_degree: float,
    recsys_count: int,
    bandwidth: float,
    diffusion: float,
) -> float:
    rho0 = 1 / length

    def kernel(value: float) -> float:
        return float(np.exp(-0.5 * (value / bandwidth) ** 2))

    kernel_mass = quad(kernel, -length / 2, length / 2, epsabs=1e-12)[0]
    concordant_fraction = (
        quad(kernel, -epsilon, epsilon, epsabs=1e-12)[0] / kernel_mass
    )
    random_moment = 2 * quad(
        lambda value: value * np.sin(kappa * value),
        0,
        epsilon,
        epsabs=1e-12,
    )[0]
    kernel_moment = 2 * quad(
        lambda value: value * kernel(value) * np.sin(kappa * value),
        0,
        epsilon,
        epsabs=1e-12,
    )[0]
    numerator = (
        mean_degree * rho0 * random_moment
        + recsys_count * kernel_moment / kernel_mass
    )
    denominator = (
        2 * epsilon * mean_degree * rho0
        + recsys_count * concordant_fraction
    )
    return float(
        influence * kappa * numerator / denominator
        - diffusion * kappa * kappa
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=float, default=2.0)
    parser.add_argument("--epsilon", type=float, default=0.45)
    parser.add_argument("--influence", type=float, default=0.05)
    parser.add_argument("--mean-degree", type=float, default=15.0)
    parser.add_argument("--recsys-count", type=int, default=10)
    parser.add_argument("--bandwidth", type=float, default=0.1)
    parser.add_argument("--diffusion", type=float, default=1e-5)
    parser.add_argument("--grid-size", type=int, default=1024)
    parser.add_argument("--amplitude", type=float, default=1e-6)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MESOSCOPIC_OUTPUT.resolve() / "spectrum_check",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mode_numbers = np.arange(1, args.modes + 1)
    kappas = 2 * np.pi * mode_numbers / args.length
    rows = []
    for mode, kappa in zip(mode_numbers, kappas):
        common = {
            "epsilon": args.epsilon,
            "influence": args.influence,
            "diffusion": args.diffusion,
        }
        random_analytic = _random_growth_rate(kappa, **common)
        opinion_analytic = _opinion_growth_rate(
            kappa,
            length=args.length,
            mean_degree=args.mean_degree,
            recsys_count=args.recsys_count,
            bandwidth=args.bandwidth,
            **common,
        )
        random_measured = _measured_growth_rate(
            kappa,
            recommender="random",
            length=args.length,
            mean_degree=args.mean_degree,
            recsys_count=args.recsys_count,
            bandwidth=args.bandwidth,
            grid_size=args.grid_size,
            amplitude=args.amplitude,
            **common,
        )
        opinion_measured = _measured_growth_rate(
            kappa,
            recommender="opinion",
            length=args.length,
            mean_degree=args.mean_degree,
            recsys_count=args.recsys_count,
            bandwidth=args.bandwidth,
            grid_size=args.grid_size,
            amplitude=args.amplitude,
            **common,
        )
        rows.append(
            {
                "mode": int(mode),
                "kappa": float(kappa),
                "z": float(kappa * args.epsilon),
                "lambda_random_analytic": random_analytic,
                "lambda_random_measured": random_measured,
                "lambda_opinion_analytic": opinion_analytic,
                "lambda_opinion_measured": opinion_measured,
            }
        )

    with (args.output_dir / "spectrum_check.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)

    curve_kappa = np.linspace(0, kappas[-1] * 1.04, 300)
    random_curve = np.asarray(
        [
            _random_growth_rate(
                value,
                epsilon=args.epsilon,
                influence=args.influence,
                diffusion=args.diffusion,
            )
            for value in curve_kappa
        ]
    )
    opinion_curve = np.asarray(
        [
            _opinion_growth_rate(
                value,
                length=args.length,
                epsilon=args.epsilon,
                influence=args.influence,
                mean_degree=args.mean_degree,
                recsys_count=args.recsys_count,
                bandwidth=args.bandwidth,
                diffusion=args.diffusion,
            )
            for value in curve_kappa
        ]
    )

    setup_paper_params()
    fig, axis = plt.subplots(figsize=(4.25, 3.0), constrained_layout=True)
    axis.axhline(0, color="0.45", lw=0.7)
    axis.plot(
        curve_kappa * args.epsilon,
        random_curve,
        color="#0072B2",
        label="Random, analytic",
    )
    axis.plot(
        curve_kappa * args.epsilon,
        opinion_curve,
        color="#D55E00",
        label="Opinion, analytic",
    )
    axis.scatter(
        kappas * args.epsilon,
        [row["lambda_random_measured"] for row in rows],
        facecolors="white",
        edgecolors="#0072B2",
        s=24,
        zorder=3,
        label="Random, measured",
    )
    axis.scatter(
        kappas * args.epsilon,
        [row["lambda_opinion_measured"] for row in rows],
        marker="s",
        facecolors="white",
        edgecolors="#D55E00",
        s=22,
        zorder=3,
        label="Opinion, measured",
    )
    axis.set_xlabel(r"dimensionless wave number $z=\kappa\epsilon$")
    axis.set_ylabel(r"growth rate $\lambda$")
    axis.legend(frameon=False, fontsize=7, ncol=2)
    axis.grid(alpha=0.18, linewidth=0.5)
    figure_path = args.output_dir / "f_spectrum_validation"
    fig.savefig(figure_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(figure_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    parameters = {
        key: value
        for key, value in vars(args).items()
        if key != "output_dir"
    }
    write_run_metadata(
        args.output_dir / "run_metadata.json",
        analysis="periodic HK spectrum validation",
        command=shlex.join(
            [sys.executable, "-m", "theory.mesoscopic.spectrum_check", *sys.argv[1:]]
        ),
        parameters=parameters,
        configuration={"output_dir": str(args.output_dir.resolve())},
    )
    max_error = max(
        abs(row["lambda_random_measured"] - row["lambda_random_analytic"])
        for row in rows
    )
    max_error = max(
        max_error,
        max(
            abs(row["lambda_opinion_measured"] - row["lambda_opinion_analytic"])
            for row in rows
        ),
    )
    print(f"spectrum check written to {args.output_dir}; max abs error={max_error:.3e}")


if __name__ == "__main__":
    main()

"""Run and plot the mesoscopic no-repost/no-history pathway solver.

Run from the repository root with, for example:

    python -m works.solve_kinetic
    python -m works.solve_kinetic --noise 1e-5 --tag noise_1e-5
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.plot import setup_paper_params
from works.kinetic import (
    IndexSeries,
    KineticParameters,
    KineticTrajectory,
    calculate_index_series,
    solve,
)


@dataclass
class ScenarioResult:
    key: str
    title: str
    trajectory: KineticTrajectory
    indices: IndexSeries


def _first_crossing(time: np.ndarray, values: np.ndarray, threshold: float):
    found = np.flatnonzero(values >= threshold)
    return float(time[found[0]]) if found.size else None


def _path_label(indices: IndexSeries, threshold: float = 0.5) -> str:
    t_p = _first_crossing(indices.time, indices.polarization, threshold)
    t_h = _first_crossing(indices.time, indices.homophily, threshold)
    if t_p is None and t_h is None:
        return "unresolved"
    if t_h is None or (t_p is not None and t_p < t_h):
        return "PbS"
    if t_p is None or t_h < t_p:
        return "SbP"
    return "simultaneous"


def _save_result(result: ScenarioResult, output_dir: Path) -> None:
    trajectory = result.trajectory
    indices = result.indices
    np.savez_compressed(
        output_dir / f"{result.key}.npz",
        parameters=json.dumps(asdict(trajectory.parameters), sort_keys=True),
        x=trajectory.x,
        time=trajectory.time,
        rho=trajectory.rho,
        velocity=trajectory.velocity,
        edge=trajectory.edge,
        rewiring_flux=trajectory.rewiring_flux,
        I_p=indices.polarization,
        I_s=indices.subjective,
        I_h=indices.homophily,
        rho_epsilon=indices.homophily_raw,
        I_w=np.asarray(indices.pathway),
    )


def _plot_fields(results: list[ScenarioResult], figure_path: Path) -> None:
    setup_paper_params()
    fig, axes = plt.subplots(
        len(results),
        3,
        figsize=(12, 3.6 * len(results)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, result in enumerate(results):
        trajectory = result.trajectory
        x = trajectory.x
        dx = x[1] - x[0]
        extent = (
            trajectory.time[0],
            trajectory.time[-1],
            x[0],
            x[-1],
        )

        density = trajectory.rho.T / dx
        image_density = axes[row, 0].imshow(
            density,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="magma",
        )
        fig.colorbar(image_density, ax=axes[row, 0], label=r"$\rho(x,t)$")
        axes[row, 0].set_title(f"{result.title}: opinion density", loc="left")
        axes[row, 0].set_xlabel("time")
        axes[row, 0].set_ylabel(r"opinion $x$")

        vmax = max(float(np.max(np.abs(trajectory.velocity))), 1e-12)
        image_velocity = axes[row, 1].imshow(
            trajectory.velocity.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="coolwarm",
            vmin=-vmax,
            vmax=vmax,
        )
        fig.colorbar(image_velocity, ax=axes[row, 1], label=r"$v(x,t)$")
        axes[row, 1].set_title("opinion velocity", loc="left")
        axes[row, 1].set_xlabel("time")
        axes[row, 1].set_ylabel(r"opinion $x$")

        edge_density = trajectory.edge[-1] / (
            trajectory.parameters.mean_degree * dx * dx
        )
        image_edge = axes[row, 2].imshow(
            np.log10(edge_density.T + 1e-8),
            origin="lower",
            extent=(x[0], x[-1], x[0], x[-1]),
            aspect="equal",
            cmap="viridis",
        )
        fig.colorbar(
            image_edge,
            ax=axes[row, 2],
            label=r"$\log_{10} e(x,y,t_f)$",
        )
        axes[row, 2].plot(x, x, color="white", lw=0.7, alpha=0.7)
        axes[row, 2].set_title("final directed-edge density", loc="left")
        axes[row, 2].set_xlabel(r"source opinion $x$")
        axes[row, 2].set_ylabel(r"target opinion $y$")

    for suffix in (".pdf", ".png"):
        fig.savefig(figure_path.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_indices(results: list[ScenarioResult], figure_path: Path) -> None:
    setup_paper_params()
    fig, axes = plt.subplots(
        1,
        len(results) + 1,
        figsize=(4.2 * (len(results) + 1), 4),
        constrained_layout=True,
    )
    colors = ("tab:blue", "tab:red", "tab:green")
    for ax, result in zip(axes[:-1], results):
        indices = result.indices
        ax.plot(indices.time, indices.polarization, color=colors[0], label=r"$I_p$")
        ax.plot(indices.time, indices.homophily, color=colors[1], label=r"$I_h$")
        ax.plot(indices.time, indices.subjective, color=colors[2], label=r"$I_s$")
        ax.set_xlim(indices.time[0], indices.time[-1])
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
        ax.set_xlabel("time")
        ax.set_ylabel("index")
        ax.set_title(
            f"{result.title}: {_path_label(indices)}\n"
            rf"$I_w={indices.pathway:.3f}$",
            loc="left",
        )
        ax.legend(frameon=False)

    path_ax = axes[-1]
    for result, color in zip(results, colors):
        indices = result.indices
        path_ax.plot(
            indices.polarization,
            indices.homophily,
            color=color,
            label=(
                f"{result.title} ({_path_label(indices)}, "
                rf"$I_w={indices.pathway:.3f}$)"
            ),
        )
        path_ax.scatter(
            indices.polarization[0],
            indices.homophily[0],
            color=color,
            marker="o",
            s=24,
        )
        path_ax.scatter(
            indices.polarization[-1],
            indices.homophily[-1],
            color=color,
            marker="x",
            s=36,
        )
    path_ax.set_xlim(-0.03, 1.03)
    path_ax.set_ylim(-0.03, 1.03)
    path_ax.set_aspect("equal")
    path_ax.grid(alpha=0.25)
    path_ax.set_xlabel(r"polarization $I_p$")
    path_ax.set_ylabel(r"structural homophily $I_h$")
    path_ax.set_title(r"pathway plane ($\circ$: start, $\times$: end)", loc="left")
    path_ax.legend(frameon=False, fontsize=9)

    for suffix in (".pdf", ".png"):
        fig.savefig(figure_path.with_suffix(suffix), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_summary(results: list[ScenarioResult], output_dir: Path) -> None:
    fieldnames = [
        "scenario",
        "influence",
        "rewiring",
        "noise_diffusion",
        "recsys",
        "path",
        "I_w",
        "t_Ip_0.5",
        "t_Ih_0.5",
        "I_p_final",
        "I_h_final",
        "I_s_final",
        "rho_epsilon_final",
    ]
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            params = result.trajectory.parameters
            indices = result.indices
            writer.writerow(
                {
                    "scenario": result.key,
                    "influence": params.influence,
                    "rewiring": params.rewiring,
                    "noise_diffusion": params.noise_diffusion,
                    "recsys": params.recsys,
                    "path": _path_label(indices),
                    "I_w": indices.pathway,
                    "t_Ip_0.5": _first_crossing(
                        indices.time, indices.polarization, 0.5
                    ),
                    "t_Ih_0.5": _first_crossing(
                        indices.time, indices.homophily, 0.5
                    ),
                    "I_p_final": indices.polarization[-1],
                    "I_h_final": indices.homophily[-1],
                    "I_s_final": indices.subjective[-1],
                    "rho_epsilon_final": indices.homophily_raw[-1],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--record-every", type=int, default=5)
    parser.add_argument("--grid-size", type=int, default=81)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.45)
    parser.add_argument("--noise", type=float, default=0.0, help="reflecting diffusion D0")
    parser.add_argument("--recsys", default="random", choices=("random", "opinion", "structure"))
    parser.add_argument("--output-dir", type=Path, default=Path("kinetic_output"))
    parser.add_argument("--figure-dir", type=Path, default=Path("fig"))
    parser.add_argument("--tag", default="noiseless")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.figure_dir.mkdir(parents=True, exist_ok=True)

    base = KineticParameters(
        epsilon=args.epsilon,
        recsys=args.recsys,
        noise_diffusion=args.noise,
        grid_size=args.grid_size,
        dt=args.dt,
        steps=args.steps,
        record_every=args.record_every,
    )
    definitions = (
        ("pbs", "PbS candidate", replace(base, influence=0.05, rewiring=0.025)),
        ("sbp", "SbP candidate", replace(base, influence=0.005, rewiring=0.025)),
    )
    results: list[ScenarioResult] = []
    for key, title, params in definitions:
        print(
            f"solving {key}: alpha={params.influence:g}, "
            f"q={params.rewiring:g}, D0={params.noise_diffusion:g}"
        )
        trajectory = solve(params)
        indices = calculate_index_series(trajectory)
        result = ScenarioResult(key, title, trajectory, indices)
        results.append(result)
        _save_result(result, args.output_dir)
        print(
            f"  path={_path_label(indices)}, I_w={indices.pathway:.4f}, "
            f"final=(I_p={indices.polarization[-1]:.4f}, "
            f"I_h={indices.homophily[-1]:.4f}, "
            f"I_s={indices.subjective[-1]:.4f})"
        )

    _write_summary(results, args.output_dir)
    _plot_fields(results, args.figure_dir / f"f_kinetic_fields_{args.tag}")
    _plot_indices(results, args.figure_dir / f"f_kinetic_indices_{args.tag}")
    print(f"data written to {args.output_dir}")


if __name__ == "__main__":
    main()

"""Compare L0 and directional-wedge L1 landscapes and pathway indices.

This task deliberately keeps the numerical method, initial condition, time
grid, and index definitions fixed.  The only factor is the StructureRandom
state closure: pair-only L0 versus the four-channel directional-wedge L1.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy

from ehk.common.plotting import setup_paper_params
from ehk.common.plotting.landscape import (
    CMAP_NAME,
    plot_normalized_time_colorbar,
    potential_from_force,
)
from ehk.metrics import IndexSeries, calculate_index_series
from ehk.modeling.mesoscopic import KineticParameters, KineticTrajectory, solve


@dataclass(frozen=True)
class ComparisonCase:
    key: str
    title: str
    influence: float
    rewiring: float


CASES = (
    ComparisonCase("no_rewiring", "no rewiring", 0.05, 0.0),
    ComparisonCase("balanced", "balanced", 0.05, 0.05),
    ComparisonCase("influence_dominant", "influence-dominant", 0.3, 0.005),
    ComparisonCase("rewiring_dominant", "rewiring-dominant", 0.005, 0.3),
)
CLOSURES = (
    ("l0", "structure_random_l0", "L0 pair closure"),
    ("l1", "structure_random_l1", "L1 directional wedges"),
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ComparisonResult:
    case: ComparisonCase
    level: str
    label: str
    trajectory: KineticTrajectory
    indices: IndexSeries
    potential: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_comparison(base: KineticParameters) -> list[ComparisonResult]:
    results: list[ComparisonResult] = []
    for case in CASES:
        for level, recsys, label in CLOSURES:
            parameters = replace(
                base,
                influence=case.influence,
                rewiring=case.rewiring,
                recsys=recsys,
                recommendation_steepness=1.0,
            )
            print(
                f"solving {case.key}/{level}: "
                f"alpha={case.influence:g}, q={case.rewiring:g}"
            )
            trajectory = solve(parameters)
            indices = calculate_index_series(trajectory)
            # The displayed landscape removes the update-rate coefficient,
            # matching the repository's paper-comparable social-force scale.
            social_force = trajectory.velocity / case.influence
            potential = potential_from_force(trajectory.x, social_force)
            results.append(
                ComparisonResult(case, level, label, trajectory, indices, potential)
            )
    return results


def _save_arrays(result: ComparisonResult, output_dir: Path) -> None:
    trajectory = result.trajectory
    indices = result.indices
    arrays = {
        "parameters": json.dumps(asdict(trajectory.parameters), sort_keys=True),
        "x": trajectory.x,
        "time": trajectory.time,
        "rho": trajectory.rho,
        "velocity": trajectory.velocity,
        "edge": trajectory.edge,
        "rewiring_flux": trajectory.rewiring_flux,
        "potential": result.potential,
        "I_p": indices.polarization,
        "I_s": indices.subjective,
        "I_h": indices.homophily,
        "rho_epsilon": indices.homophily_raw,
        "I_w": np.asarray(indices.pathway),
    }
    if trajectory.structural_score is not None:
        arrays["structural_score"] = trajectory.structural_score
    np.savez_compressed(output_dir / f"{result.case.key}_{result.level}.npz", **arrays)


def _write_summary(results: list[ComparisonResult], output_dir: Path) -> None:
    rows = []
    for result in results:
        indices = result.indices
        rows.append(
            {
                "case": result.case.key,
                "level": result.level,
                "closure": result.label,
                "influence": result.case.influence,
                "rewiring": result.case.rewiring,
                "q_over_alpha": result.case.rewiring / result.case.influence,
                "I_w": indices.pathway,
                "I_p_final": indices.polarization[-1],
                "I_h_final": indices.homophily[-1],
                "I_s_final": indices.subjective[-1],
            }
        )
    by_cell = {(row["case"], row["level"]): row for row in rows}
    for case in CASES:
        delta = by_cell[(case.key, "l1")]["I_w"] - by_cell[(case.key, "l0")]["I_w"]
        by_cell[(case.key, "l0")]["delta_I_w_l1_minus_l0"] = delta
        by_cell[(case.key, "l1")]["delta_I_w_l1_minus_l0"] = delta

    fields = tuple(rows[0])
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot_potentials(results: list[ComparisonResult], output_dir: Path) -> None:
    setup_paper_params()
    lookup = {(item.case.key, item.level): item for item in results}
    fig, axes = plt.subplots(
        2, len(CASES), figsize=(14, 6), sharex=True, constrained_layout=True
    )
    for column, case in enumerate(CASES):
        for row, (level, _, label) in enumerate(CLOSURES):
            result = lookup[(case.key, level)]
            indices = np.linspace(0, result.trajectory.time.size - 1, 7, dtype=int)
            colors = plt.get_cmap(CMAP_NAME)(np.linspace(0.05, 0.95, indices.size))
            for index, color in zip(indices, colors):
                axes[row, column].plot(
                    result.trajectory.x,
                    result.potential[index],
                    color=color,
                    linewidth=1.0,
                    alpha=0.9,
                )
            axes[row, column].grid(alpha=0.2)
            axes[row, column].set_xlim(-1, 1)
            axes[row, column].set_title(
                f"{case.title}\n{label}, $I_w={result.indices.pathway:.3f}$",
                fontsize=9,
            )
            axes[row, column].set_xlabel("opinion $x$")
            if column == 0:
                axes[row, column].set_ylabel("coefficient-free $V(x,t)$")
    plot_normalized_time_colorbar(fig, axes, pad=0.01)
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"potential_landscape_l0_l1.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def _plot_pathways(results: list[ComparisonResult], output_dir: Path) -> None:
    setup_paper_params()
    lookup = {(item.case.key, item.level): item for item in results}
    positions = np.arange(len(CASES), dtype=float)
    width = 0.36
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), constrained_layout=True)
    for offset, (level, _, label) in zip((-width / 2, width / 2), CLOSURES):
        values = [lookup[(case.key, level)].indices.pathway for case in CASES]
        axes[0].bar(positions + offset, values, width=width, label=label)
    deltas = [
        lookup[(case.key, "l1")].indices.pathway
        - lookup[(case.key, "l0")].indices.pathway
        for case in CASES
    ]
    axes[1].bar(positions, deltas, width=0.6, color="tab:purple")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    labels = [case.key.replace("_", "\n") for case in CASES]
    for axis in axes:
        axis.set_xticks(positions, labels)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("pathway index $I_w$")
    axes[0].legend(frameon=False)
    axes[0].set_title("same solver, different closure", loc="left")
    axes[1].set_ylabel(r"$I_w^{L1}-I_w^{L0}$")
    axes[1].set_title("L1 correction", loc="left")
    for suffix in ("pdf", "png"):
        fig.savefig(
            output_dir / f"pathway_index_l0_l1.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/theory/mesoscopic/l1_comparison"),
    )
    parser.add_argument("--grid-size", type=int, default=21)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--record-every", type=int, default=20)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--noise", type=float, default=1e-5)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = KineticParameters(
        epsilon=0.45,
        mean_degree=15,
        recsys_count=10,
        recommendation_random_ratio=0.0,
        recommendation_steepness=1.0,
        grid_size=args.grid_size,
        steps=args.steps,
        record_every=args.record_every,
        dt=args.dt,
        noise_diffusion=args.noise,
    )
    results = run_comparison(base)
    for result in results:
        _save_arrays(result, output_dir)
    _write_summary(results, output_dir)
    _plot_potentials(results, output_dir)
    _plot_pathways(results, output_dir)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": " ".join([sys.executable, *sys.argv]),
        "parameters": asdict(base),
        "cases": [asdict(case) for case in CASES],
        "closure_factor": [item[:2] for item in CLOSURES],
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "potential_scale": "velocity divided by influence",
        "source_sha256": {
            "comparison": _sha256(Path(__file__).resolve()),
            "solver": _sha256(
                REPOSITORY_ROOT / "src/ehk/modeling/mesoscopic/solver.py"
            ),
            "directional_wedge": _sha256(
                REPOSITORY_ROOT / "src/ehk/modeling/mesoscopic/directional_wedge.py"
            ),
        },
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote L0/L1 comparison to {output_dir}")


if __name__ == "__main__":
    main()

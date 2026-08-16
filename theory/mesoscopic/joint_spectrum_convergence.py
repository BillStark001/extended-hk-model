"""Compare two completed joint-spectrum discretizations.

This is a Section 3.6 numerical audit.  It compares operator spectra only and
does not read microscopic trajectories or finite-system outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from pathlib import Path

from theory.mesoscopic.cli_utils import write_run_metadata
from theory.paths import MESOSCOPIC_OUTPUT


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("the two spectrum outputs have no matching rows")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _grid_size(directory: Path) -> int:
    metadata = json.loads((directory / "run_metadata.json").read_text())
    return int(metadata["parameters"]["grid_size"])


def _finite_key(row: dict[str, str]) -> tuple[str, float, float, int, float]:
    return (
        row["recommender"],
        float(row["alpha"]),
        float(row["q"]),
        int(row["mode"]),
        float(row["horizon"]),
    )


def _instantaneous_key(
    row: dict[str, str],
) -> tuple[str, float, float, float, int]:
    return (
        row["recommender"],
        float(row["alpha"]),
        float(row["q"]),
        float(row["base_discordant_fraction_ratio"]),
        int(row["mode"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison-dir",
        type=Path,
        default=MESOSCOPIC_OUTPUT.resolve() / "joint_spectrum",
    )
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MESOSCOPIC_OUTPUT.resolve() / "joint_spectrum",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_grid = _grid_size(args.comparison_dir)
    reference_grid = _grid_size(args.reference_dir)

    comparison_finite = {
        _finite_key(row): row
        for row in _read(args.comparison_dir / "joint_finite_time_summary.csv")
    }
    reference_finite = {
        _finite_key(row): row
        for row in _read(args.reference_dir / "joint_finite_time_summary.csv")
    }
    finite_rows = []
    for key in sorted(comparison_finite.keys() & reference_finite.keys()):
        comparison = comparison_finite[key]
        reference = reference_finite[key]
        comparison_rate = float(comparison["leading_singular_rate"])
        reference_rate = float(reference["leading_singular_rate"])
        comparison_rho = float(comparison["rho_seed_rate"])
        reference_rho = float(reference["rho_seed_rate"])
        finite_rows.append(
            {
                "recommender": key[0],
                "alpha": key[1],
                "q": key[2],
                "mode": key[3],
                "horizon": key[4],
                "comparison_grid": comparison_grid,
                "reference_grid": reference_grid,
                "comparison_joint_rate": comparison_rate,
                "reference_joint_rate": reference_rate,
                "joint_rate_difference": comparison_rate - reference_rate,
                "comparison_rho_seed_rate": comparison_rho,
                "reference_rho_seed_rate": reference_rho,
                "rho_seed_rate_difference": comparison_rho - reference_rho,
            }
        )
    _write(args.output_dir / "joint_grid_convergence_finite_time.csv", finite_rows)

    comparison_instantaneous = {
        _instantaneous_key(row): row
        for row in _read(args.comparison_dir / "joint_instantaneous_summary.csv")
    }
    reference_instantaneous = {
        _instantaneous_key(row): row
        for row in _read(args.reference_dir / "joint_instantaneous_summary.csv")
    }
    instantaneous_rows = []
    for key in sorted(
        comparison_instantaneous.keys() & reference_instantaneous.keys()
    ):
        comparison = comparison_instantaneous[key]
        reference = reference_instantaneous[key]
        comparison_rate = float(comparison["spectral_abscissa"])
        reference_rate = float(reference["spectral_abscissa"])
        instantaneous_rows.append(
            {
                "recommender": key[0],
                "alpha": key[1],
                "q": key[2],
                "base_discordant_fraction_ratio": key[3],
                "mode": key[4],
                "comparison_grid": comparison_grid,
                "reference_grid": reference_grid,
                "comparison_spectral_abscissa": comparison_rate,
                "reference_spectral_abscissa": reference_rate,
                "spectral_abscissa_difference": comparison_rate - reference_rate,
            }
        )
    _write(
        args.output_dir / "joint_grid_convergence_instantaneous.csv",
        instantaneous_rows,
    )
    write_run_metadata(
        args.output_dir / "joint_grid_convergence_metadata.json",
        analysis="joint spectrum grid convergence",
        command=shlex.join(
            [
                sys.executable,
                "-m",
                "theory.mesoscopic.joint_spectrum_convergence",
                *sys.argv[1:],
            ]
        ),
        parameters={
            "comparison_grid": comparison_grid,
            "reference_grid": reference_grid,
        },
        configuration={
            "comparison_dir": str(args.comparison_dir.resolve()),
            "reference_dir": str(args.reference_dir.resolve()),
            "output_dir": str(args.output_dir.resolve()),
        },
    )
    print(
        f"joint spectrum convergence written to {args.output_dir} "
        f"({comparison_grid} vs {reference_grid})"
    )


if __name__ == "__main__":
    main()

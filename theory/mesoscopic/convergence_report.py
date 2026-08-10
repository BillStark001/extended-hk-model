"""Summarize grid sensitivity of completed five-recommender scans."""

from __future__ import annotations

import argparse
import csv
import shlex
import sys
from pathlib import Path

import numpy as np

from theory.mesoscopic.cli_utils import write_run_metadata
from theory.mesoscopic.recommender_scan import RECOMMENDERS
from theory.paths import MESOSCOPIC_OUTPUT


DEFAULT_GRIDS = (61, 81, 101)


def _default_scan_dir(grid_size: int) -> Path:
    suffix = "" if grid_size == 81 else f"_grid{grid_size}"
    return MESOSCOPIC_OUTPUT.resolve() / f"recsys_noise_1e-5{suffix}"


def _path_counts(summary_path: Path, recommender: str) -> dict[str, int]:
    with summary_path.open(encoding="utf-8") as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["recsys"] == recommender
        ]
    return {
        label: sum(row["path"] == label for row in rows)
        for label in ("PbS", "SbP", "simultaneous", "unresolved")
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scan-dir",
        action="append",
        type=Path,
        help="GRID=PATH; repeat for each grid (defaults: 61, 81, 101)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=MESOSCOPIC_OUTPUT.resolve() / "recsys_grid_convergence",
    )
    return parser.parse_args()


def _parse_scan_dirs(values: list[Path] | None) -> dict[int, Path]:
    if not values:
        return {grid: _default_scan_dir(grid) for grid in DEFAULT_GRIDS}
    result = {}
    for value in values:
        grid_text, separator, path_text = str(value).partition("=")
        if not separator:
            raise ValueError("--scan-dir must have the form GRID=PATH")
        result[int(grid_text)] = Path(path_text)
    return result


def main() -> None:
    args = parse_args()
    scan_dirs = _parse_scan_dirs(args.scan_dir)
    arrays = {
        grid: np.load(path / "recsys_indices.npz")
        for grid, path in scan_dirs.items()
    }
    if 81 not in arrays:
        raise ValueError("an 81-bin reference scan is required")
    reference = arrays[81]
    rows = []
    for grid in sorted(arrays):
        data = arrays[grid]
        for recommender in RECOMMENDERS:
            pathway = data[f"{recommender}_I_w"]
            final_polarization = data[f"{recommender}_I_p"][:, :, -1]
            delta_pathway = np.abs(
                pathway - reference[f"{recommender}_I_w"]
            )
            delta_polarization = np.abs(
                final_polarization
                - reference[f"{recommender}_I_p"][:, :, -1]
            )
            counts = _path_counts(
                scan_dirs[grid] / "recsys_summary.csv", recommender
            )
            rows.append(
                {
                    "grid_size": grid,
                    "recsys": recommender,
                    "mean_I_w": float(pathway.mean()),
                    "mean_final_I_p": float(final_polarization.mean()),
                    **counts,
                    "mae_I_w_vs_81": float(delta_pathway.mean()),
                    "p95_I_w_vs_81": float(np.quantile(delta_pathway, 0.95)),
                    "max_I_w_vs_81": float(delta_pathway.max()),
                    "mae_final_I_p_vs_81": float(delta_polarization.mean()),
                    "p95_final_I_p_vs_81": float(
                        np.quantile(delta_polarization, 0.95)
                    ),
                    "max_final_I_p_vs_81": float(delta_polarization.max()),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "grid_convergence.csv"
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=rows[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    write_run_metadata(
        args.output_dir / "run_metadata.json",
        analysis="five-recommender grid-convergence summary",
        command=shlex.join(
            [
                sys.executable,
                "-m",
                "theory.mesoscopic.convergence_report",
                *sys.argv[1:],
            ]
        ),
        parameters={"reference_grid": 81},
        configuration={
            "scan_dirs": {
                str(grid): str(path.resolve())
                for grid, path in scan_dirs.items()
            },
            "output_dir": str(args.output_dir.resolve()),
        },
    )
    print(f"grid-convergence summary written to {output_path}")


if __name__ == "__main__":
    main()

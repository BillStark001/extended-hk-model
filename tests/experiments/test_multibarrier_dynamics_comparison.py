import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.theory_guided.macroscopic_timescale_ratio.scenarios import (
    select_scenarios,
)
from experiments.theory_guided.multibarrier_dynamics_comparison.analyze import (
    paired_differences,
)
from experiments.theory_guided.multibarrier_dynamics_comparison.protocol import (
    DEFAULT_CONFIGURATIONS,
    DEFAULT_DYNAMICS,
    DEFAULT_METHODS,
    PAPER_RATES,
    build_cases,
    parse_epsilon_grids,
    select_grid_cells,
)
from experiments.theory_guided.multibarrier_dynamics_comparison.run import main
from experiments.theory_guided.multibarrier_dynamics_comparison.select_comparison_points import (
    select_points,
)


class MultibarrierDynamicsComparisonTests(unittest.TestCase):
    def test_default_band_has_28_cells_and_1680_cases(self):
        epsilon_grids = parse_epsilon_grids(None)
        cells = select_grid_cells(
            PAPER_RATES,
            selection="anti_diagonal_band",
            band_offsets=(-1, 0, 1),
        )
        cases = build_cases(
            epsilon_grids,
            DEFAULT_DYNAMICS,
            DEFAULT_METHODS,
            select_scenarios(list(DEFAULT_CONFIGURATIONS)),
            cells,
        )

        self.assertEqual(len(cells), 28)
        self.assertEqual({cell.diagonal_offset for cell in cells}, {-1, 0, 1})
        self.assertEqual(len(cases), 1680)
        self.assertEqual(epsilon_grids[0].grid_size, 121)

    def test_four_way_smoke_run_is_resumable_and_pairable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            arguments = [
                "--epsilon-grid",
                "0.2:11",
                "--grid-selection",
                "explicit",
                "--explicit-cell",
                "0:0",
                "--configurations",
                "random",
                "--steps",
                "2",
                "--early-until",
                "1",
                "--record-every",
                "2",
                "--persistence",
                "1",
                "--jobs",
                "1",
                "--skip-analysis",
                "--output-dir",
                str(output),
            ]
            main(arguments)
            checkpoints = sorted((output / "cells").rglob("*.npz"))
            timestamps = {path: path.stat().st_mtime_ns for path in checkpoints}
            main(arguments)

            self.assertEqual(len(checkpoints), 4)
            self.assertEqual(
                timestamps,
                {path: path.stat().st_mtime_ns for path in checkpoints},
            )
            with np.load(checkpoints[0], allow_pickle=False) as arrays:
                self.assertIn("multi_dominant_barrier_height", arrays.files)
                self.assertIn("displacement_second_moment", arrays.files)
                self.assertEqual(arrays["rho"].shape[-1], 11)

            import csv

            with (output / "summary.csv").open(newline="", encoding="utf-8") as stream:
                paired = paired_differences(list(csv.DictReader(stream)))
            selected = select_points(paired, points_per_epsilon=1)
            self.assertEqual(len(paired), 1)
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected[0]["selection_reason"], "method_disagreement")


if __name__ == "__main__":
    unittest.main()

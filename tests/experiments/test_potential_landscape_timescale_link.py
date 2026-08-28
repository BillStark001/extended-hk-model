import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.theory_guided.macroscopic_timescale_ratio.scenarios import (
    SCENARIOS,
)
from experiments.theory_guided.potential_landscape_timescale_link.run import (
    build_cases,
    main,
)


def _write_offsets(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("predictor", "configuration", "transition")
        )
        writer.writeheader()
        for index, scenario in enumerate(SCENARIOS):
            writer.writerow(
                {
                    "predictor": "rate_ratio",
                    "configuration": scenario.key,
                    "transition": 0.5 + 0.1 * index,
                }
            )


class PotentialLandscapeTimescaleLinkTests(unittest.TestCase):
    def test_build_cases_uses_fitted_transition_center(self):
        with tempfile.TemporaryDirectory() as directory:
            offsets = Path(directory) / "offsets.csv"
            _write_offsets(offsets)
            cases = build_cases(
                offsets,
                designs=("common_rates", "transition_center"),
                configurations=("random",),
                common_alpha=0.1,
                common_q=0.1,
                transition_alpha=0.2,
            )
        self.assertEqual(len(cases), 2)
        self.assertAlmostEqual(cases[0].q, 0.1)
        self.assertAlmostEqual(cases[1].q, 0.1)

    def test_smoke_run_writes_resumable_landscape_analysis(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            offsets = root / "offsets.csv"
            output = root / "output"
            _write_offsets(offsets)
            arguments = [
                "--transition-offsets",
                str(offsets),
                "--output-dir",
                str(output),
                "--designs",
                "common_rates",
                "--configurations",
                "random",
                "--grid-size",
                "11",
                "--steps",
                "4",
                "--early-until",
                "2",
                "--record-every",
                "2",
                "--jobs",
                "1",
            ]
            main(arguments)
            checkpoint = output / "cases" / "common_rates__random.npz"
            timestamp = checkpoint.stat().st_mtime_ns
            main(arguments)

            self.assertEqual(checkpoint.stat().st_mtime_ns, timestamp)
            expected = {
                "RESULTS.md",
                "analysis_metrics.json",
                "f_barrier_evolution.pdf",
                "f_barrier_evolution.png",
                "f_landscape_timing.pdf",
                "f_landscape_timing.png",
                "f_potential_snapshots.pdf",
                "f_potential_snapshots.png",
                "protocol.json",
                "run_metadata.json",
                "summary.csv",
            }
            self.assertEqual(
                {path.name for path in output.iterdir() if path.is_file()},
                expected,
            )
            with np.load(checkpoint, allow_pickle=False) as arrays:
                self.assertEqual(arrays["rho"].shape[-1], 11)
                self.assertEqual(arrays["potential"].shape[-1], 11)
                self.assertIn("barrier_height", arrays.files)


if __name__ == "__main__":
    unittest.main()

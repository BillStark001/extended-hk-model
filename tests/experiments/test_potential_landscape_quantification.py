import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.theory_guided.potential_landscape_quantification.run import (
    main,
    record_schedule,
)


class PotentialLandscapeWorkflowTests(unittest.TestCase):
    def test_record_schedule_is_dense_early_and_includes_final_step(self):
        self.assertEqual(
            record_schedule(12, early_until=5, early_every=1, late_every=4),
            (0, 1, 2, 3, 4, 5, 9, 12),
        )

    def test_smoke_run_writes_quantitative_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            main(
                [
                    "--output-dir",
                    str(output),
                    "--grid-size",
                    "11",
                    "--steps",
                    "3",
                    "--early-until",
                    "2",
                    "--cases",
                    "balanced",
                    "--closures",
                    "l0",
                ]
            )
            expected = {
                "balanced_l0.npz",
                "summary.csv",
                "barrier_maturation.pdf",
                "barrier_maturation.png",
                "landscape_timing_summary.pdf",
                "landscape_timing_summary.png",
                "run_metadata.json",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            with (output / "summary.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["case"], "balanced")
            arrays = np.load(output / "balanced_l0.npz")
            self.assertIn("barrier_height", arrays.files)
            self.assertIn("well_separation", arrays.files)


if __name__ == "__main__":
    unittest.main()

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.archive.macroscopic_timescale_ratio_b41.run import (
    TimescaleResult,
    analyze_predictors,
    fit_transition_offsets,
    main,
    optimal_threshold,
    record_schedule,
)
from theory.mesoscopic.recommender_scan import CONFIGURATIONS


class MacroscopicTimescaleWorkflowTests(unittest.TestCase):
    def test_record_schedule_is_dense_before_split(self):
        self.assertEqual(
            record_schedule(12, early_until=5, early_every=1, late_every=4),
            (0, 1, 2, 3, 4, 5, 9, 12),
        )

    def test_optimal_threshold_separates_orderings(self):
        score = optimal_threshold(
            np.asarray([-2.0, -1.0, 1.0, 2.0]),
            np.asarray([False, False, True, True]),
        )
        self.assertEqual(score.accuracy, 1.0)
        self.assertEqual(score.balanced_accuracy, 1.0)
        self.assertEqual(score.auc, 1.0)
        self.assertAlmostEqual(score.threshold, 1.0)

    def test_analysis_reports_global_and_configuration_transfer(self):
        results = []
        for configuration, recsys, steepness in CONFIGURATIONS:
            for gamma, path, pathway in ((0.5, "PbS", 0.2), (2.0, "SbP", 0.8)):
                results.append(
                    TimescaleResult(
                        configuration,
                        recsys,
                        steepness,
                        0.1,
                        0.1,
                        path,
                        pathway,
                        10.0 if path == "PbS" else 20.0,
                        20.0 if path == "PbS" else 10.0,
                        1.0 if path == "PbS" else -1.0,
                        0.1,
                        0.1 * gamma,
                        gamma,
                        gamma,
                        gamma,
                        gamma,
                        gamma,
                        5.0,
                    )
                )
        metrics, thresholds = analyze_predictors(results)
        self.assertEqual(metrics["gamma"]["balanced_accuracy"], 1.0)
        self.assertEqual(metrics["gamma"]["loco_balanced_accuracy"], 1.0)
        self.assertEqual(len(thresholds), 6 * len(CONFIGURATIONS))

    def test_common_slope_fit_reduces_each_configuration_to_one_offset(self):
        results = []
        for configuration_index, (configuration, recsys, steepness) in enumerate(
            CONFIGURATIONS[:2]
        ):
            threshold = -0.5 + configuration_index
            for value in (-2.0, -1.0, 0.0, 1.0, 2.0):
                path = "SbP" if value >= threshold else "PbS"
                gamma = 10.0**value
                results.append(
                    TimescaleResult(
                        configuration,
                        recsys,
                        steepness,
                        0.1,
                        0.1,
                        path,
                        float(path == "SbP"),
                        1.0,
                        1.0,
                        0.0,
                        1.0,
                        gamma,
                        gamma,
                        gamma,
                        gamma,
                        gamma,
                        gamma,
                        1.0,
                    )
                )
        _, rows = fit_transition_offsets(results, "gamma_initial", CONFIGURATIONS[:2])
        offsets = np.asarray([float(row["offset_log10"]) for row in rows])
        self.assertAlmostEqual(float(offsets.sum()), 0.0)
        self.assertLess(offsets[0], 0.0)
        self.assertGreater(offsets[1], 0.0)

    def test_smoke_run_writes_auditable_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            main(
                [
                    "--output-dir",
                    str(output),
                    "--grid-size",
                    "11",
                    "--steps",
                    "4",
                    "--early-until",
                    "2",
                    "--rates",
                    "0.01",
                    "0.1",
                    "--configurations",
                    "random",
                    "--jobs",
                    "1",
                    "--skip-plots",
                ]
            )
            expected = {
                "collapse_metrics.json",
                "collapse_metrics_rates_below_one.json",
                "configuration_thresholds.csv",
                "configuration_thresholds_rates_below_one.csv",
                "run_metadata.json",
                "summary.csv",
                "timescale_ratio.npz",
                "transition_offset_metrics.json",
                "transition_offsets.csv",
            }
            self.assertEqual({path.name for path in output.iterdir()}, expected)
            with (output / "summary.csv").open(encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            metrics = json.loads((output / "collapse_metrics.json").read_text())
            self.assertIn("gamma", metrics)
            with np.load(output / "timescale_ratio.npz") as arrays:
                self.assertEqual(arrays["gamma"].shape, (1, 2, 2))


if __name__ == "__main__":
    unittest.main()

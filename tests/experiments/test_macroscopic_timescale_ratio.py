import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.theory_guided.macroscopic_timescale_ratio.analyze import (
    fit_transition_offsets,
)
from experiments.theory_guided.macroscopic_timescale_ratio.run import (
    TimescaleCell,
    main,
)
from experiments.theory_guided.macroscopic_timescale_ratio.scenarios import (
    SCENARIOS,
    record_schedule,
)


def _synthetic_cell(
    configuration: str,
    recsys: str,
    steepness: float,
    value: float,
    path: str,
) -> TimescaleCell:
    gamma = 10.0**value
    return TimescaleCell(
        configuration=configuration,
        recsys=recsys,
        steepness=steepness,
        closure_note="",
        alpha_index=0,
        q_index=0,
        alpha=0.1,
        q=0.1,
        path=path,
        pathway=float(path == "SbP"),
        t_polarization=1.0,
        t_homophily=1.0,
        precedence=0.0,
        polarization_final=0.5,
        homophily_final=0.5,
        subjective_final=0.5,
        opinion_rate_gamma_0=1.0,
        rewiring_rate_gamma_0=gamma,
        gamma_0=gamma,
        gamma_0p1_time=1.0,
        gamma_0p1_reached=True,
        opinion_rate_gamma_0p1=1.0,
        rewiring_rate_gamma_0p1=gamma,
        gamma_0p1=gamma,
    )


class MacroscopicTimescaleRatioTests(unittest.TestCase):
    def test_protocol_has_seven_ordered_scenarios(self):
        self.assertEqual(len(SCENARIOS), 7)
        self.assertEqual(SCENARIOS[0].key, "random")
        self.assertEqual(SCENARIOS[-1].key, "structure_random_l1_zeta4")
        self.assertIn("mean-power", SCENARIOS[-1].closure_note)

    def test_record_schedule_preserves_dense_early_states(self):
        self.assertEqual(
            record_schedule(12, early_until=5, early_every=1, late_every=4),
            (0, 1, 2, 3, 4, 5, 9, 12),
        )

    def test_common_slope_fit_returns_one_centered_offset_per_scenario(self):
        scenarios = SCENARIOS[:2]
        cells = []
        for scenario_index, scenario in enumerate(scenarios):
            threshold = -0.5 + scenario_index
            for value in (-2.0, -1.0, 0.0, 1.0, 2.0):
                path = "SbP" if value >= threshold else "PbS"
                cells.append(
                    _synthetic_cell(
                        scenario.key,
                        scenario.recsys,
                        scenario.steepness,
                        value,
                        path,
                    )
                )
        _, rows = fit_transition_offsets(cells, "gamma_0", scenarios)
        offsets = np.asarray([float(row["offset_log10"]) for row in rows])
        self.assertAlmostEqual(float(offsets.sum()), 0.0)
        self.assertLess(offsets[0], 0.0)
        self.assertGreater(offsets[1], 0.0)

    def test_smoke_run_writes_resumable_cell_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            arguments = [
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
                "--skip-analysis",
            ]
            main(arguments)
            main(arguments)
            self.assertTrue((output / "protocol.json").is_file())
            self.assertTrue((output / "timescale_ratio.npz").is_file())
            self.assertEqual(len(list((output / "cells" / "random").glob("*.json"))), 4)
            with (output / "summary.csv").open(encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)


if __name__ == "__main__":
    unittest.main()

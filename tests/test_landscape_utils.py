import tempfile
import unittest
from pathlib import Path

import numpy as np

from works.landscape.config import SCENARIOS, SCENARIO_ORDER
from works.landscape.plot_utils import potential_from_force
from works.landscape.probe_results import load_probe_landscape


class LandscapeUtilityTests(unittest.TestCase):
    def test_potential_integration_supports_multiple_curves(self):
        x = np.array([-1.0, 0.0, 1.0])
        forces = np.array([
            [1.0, 1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ])
        potential = potential_from_force(x, forces)
        np.testing.assert_allclose(
            potential,
            np.array([
                [1.0, 0.0, -1.0],
                [-1.0, 0.0, 1.0],
            ]),
        )
        np.testing.assert_allclose(np.mean(potential, axis=1), 0.0)

    def test_scenario_order_covers_every_mechanism_map_record(self):
        self.assertEqual(len(SCENARIO_ORDER), 11)
        self.assertEqual(set(SCENARIO_ORDER), set(SCENARIOS))
        self.assertEqual(
            [SCENARIOS[key].unique_name for key in SCENARIO_ORDER],
            [
                "s_mech_baseline",
                "s_mech_baseline_pbs",
                "s_mech_baseline_sbp",
                "s_mech_influence",
                "s_mech_retweet",
                "s_mech_op_recsys",
                "s_mech_phase1",
                "s_mech_phase2",
                "s_mech_phase3",
                "s_mech_phase4",
                "s_mech_phase5",
            ],
        )

    def test_probe_loader_validates_scenario_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "counterfactual_probe_baseline.npz"
            np.savez_compressed(
                path,
                scenario_key=np.asarray("pbs"),
                x=np.array([-1.0, 1.0]),
                normalized_time=np.array([0.0]),
                social_force_mean=np.zeros((1, 2)),
                social_force_neighbor_mean=np.zeros((1, 2)),
            )
            with self.assertRaises(ValueError):
                load_probe_landscape("baseline", directory)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from ehk.metrics import (
    DensityIndexCalculator,
    calculate_channel_contributions,
    calculate_index_series,
)
from ehk.modeling.mesoscopic import KineticParameters, solve


class MacroscopicTimescaleTests(unittest.TestCase):
    def test_public_index_components_match_joint_calculation(self):
        trajectory = solve(
            KineticParameters(grid_size=21, steps=2, record_every=1)
        )
        calculator = DensityIndexCalculator(
            trajectory.x,
            trajectory.parameters.epsilon,
            trajectory.parameters.mean_degree,
        )
        row = calculator.calculate(trajectory.rho[-1], trajectory.edge[-1])
        self.assertAlmostEqual(calculator.polarization(trajectory.rho[-1]), row[0])
        self.assertAlmostEqual(calculator.subjective(trajectory.edge[-1]), row[1])
        self.assertAlmostEqual(calculator.homophily(trajectory.edge[-1])[0], row[2])

    def test_zero_rewiring_has_zero_structural_drive(self):
        trajectory = solve(
            KineticParameters(
                grid_size=21,
                steps=20,
                record_every=1,
                influence=0.05,
                rewiring=0.0,
                noise_diffusion=0.0,
            )
        )
        contributions = calculate_channel_contributions(trajectory)
        np.testing.assert_allclose(contributions.rewiring_homophily_rate, 0.0)
        self.assertEqual(contributions.rewiring_drive, 0.0)
        self.assertEqual(contributions.gamma_integrated, 0.0)

    def test_zero_influence_has_infinite_ratio_when_rewiring_is_active(self):
        trajectory = solve(
            KineticParameters(
                grid_size=21,
                steps=20,
                record_every=1,
                influence=0.0,
                rewiring=0.1,
                noise_diffusion=1e-4,
            )
        )
        indices = calculate_index_series(trajectory)
        contributions = calculate_channel_contributions(trajectory, indices)
        np.testing.assert_allclose(contributions.opinion_polarization_rate, 0.0)
        self.assertGreater(contributions.rewiring_drive, 0.0)
        self.assertTrue(np.isinf(contributions.gamma_integrated))

    def test_channel_series_uses_requested_preordering_window(self):
        trajectory = solve(
            KineticParameters(
                grid_size=21,
                steps=100,
                record_every=1,
                influence=0.05,
                rewiring=0.05,
                noise_diffusion=1e-5,
            )
        )
        indices = calculate_index_series(trajectory)
        contributions = calculate_channel_contributions(
            trajectory, indices, progress_threshold=0.1
        )
        self.assertLessEqual(contributions.window_end, trajectory.time[-1])
        self.assertGreaterEqual(contributions.opinion_drive, 0.0)
        self.assertGreaterEqual(contributions.rewiring_drive, 0.0)
        self.assertEqual(contributions.gamma.shape, trajectory.time.shape)


if __name__ == "__main__":
    unittest.main()

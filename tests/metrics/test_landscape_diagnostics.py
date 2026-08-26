import unittest

import numpy as np

from ehk.metrics.landscape_diagnostics import (
    first_persistent_time,
    potential_from_force,
    quantify_landscape,
    quantify_landscape_series,
)


class LandscapeDiagnosticsTests(unittest.TestCase):
    def test_quartic_double_well_recovers_geometry(self):
        x = np.linspace(-1.0, 1.0, 81)
        well = 0.5
        potential = (x * x - well * well) ** 2
        result = quantify_landscape(x, potential)
        self.assertTrue(result.double_well)
        self.assertAlmostEqual(result.left_position, -well, places=12)
        self.assertAlmostEqual(result.right_position, well, places=12)
        self.assertAlmostEqual(result.barrier_position, 0.0, places=12)
        self.assertAlmostEqual(result.barrier_height, well**4, places=12)
        self.assertAlmostEqual(result.well_separation, 2 * well, places=12)
        self.assertAlmostEqual(result.left_curvature, 8 * well * well, delta=0.03)
        self.assertAlmostEqual(result.right_curvature, 8 * well * well, delta=0.03)
        self.assertLess(result.barrier_curvature, 0.0)

    def test_single_well_is_not_classified_as_double(self):
        x = np.linspace(-1.0, 1.0, 81)
        result = quantify_landscape(x, x * x)
        self.assertFalse(result.double_well)
        self.assertEqual(result.barrier_height, 0.0)
        self.assertTrue(np.isnan(result.left_position))

    def test_persistent_times_reject_one_record_flicker(self):
        time = np.arange(6, dtype=float)
        condition = np.asarray([False, True, False, True, True, True])
        self.assertEqual(first_persistent_time(time, condition, persistence=2), 3.0)
        self.assertTrue(
            np.isnan(first_persistent_time(time, condition, persistence=4))
        )

    def test_series_reports_barrier_maturation(self):
        x = np.linspace(-1.0, 1.0, 81)
        time = np.arange(6, dtype=float)
        shape = (x * x - 0.25) ** 2 / 0.25**2
        amplitudes = np.asarray([0.0, 0.005, 0.01, 0.02, 0.04, 0.08])
        potentials = amplitudes[:, None] * shape[None, :]
        series = quantify_landscape_series(
            x,
            time,
            potentials,
            barrier_thresholds=(0.015, 0.035),
            persistence=2,
        )
        self.assertEqual(series.formation_time, 1.0)
        np.testing.assert_allclose(series.barrier_crossing_times, [3.0, 4.0])
        self.assertEqual(series.half_final_barrier_time, 4.0)
        np.testing.assert_allclose(series.barrier_height, amplitudes, atol=1e-14)

    def test_force_integration_accepts_time_series(self):
        x = np.linspace(-1.0, 1.0, 101)
        force = np.stack((-2.0 * x, -4.0 * x))
        potential = potential_from_force(x, force, center=False)
        expected = np.stack((x * x - 1.0, 2.0 * (x * x - 1.0)))
        np.testing.assert_allclose(potential, expected, atol=1e-14)


if __name__ == "__main__":
    unittest.main()

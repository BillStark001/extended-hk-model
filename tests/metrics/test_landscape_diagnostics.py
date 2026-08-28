import unittest

import numpy as np

from ehk.metrics.landscape_diagnostics import (
    first_persistent_time,
    potential_from_force,
    quantify_landscape,
    quantify_landscape_series,
    quantify_multiwell,
    quantify_multiwell_series,
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
        self.assertTrue(np.isnan(first_persistent_time(time, condition, persistence=4)))

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

    def test_multiwell_extracts_adjacent_barriers_and_basin_masses(self):
        x = np.linspace(-1.0, 1.0, 401)
        potential = (x + 0.55) ** 2 * x**2 * (x - 0.55) ** 2
        rho = (
            np.exp(-(((x + 0.55) / 0.12) ** 2))
            + 2.0 * np.exp(-((x / 0.12) ** 2))
            + np.exp(-(((x - 0.55) / 0.12) ** 2))
        )
        rho /= rho.sum()

        result = quantify_multiwell(x, potential, rho)

        self.assertEqual(result.well_position.size, 3)
        self.assertEqual(result.barrier_position.size, 2)
        self.assertEqual(result.robust_well_count, 3)
        self.assertEqual(result.robust_barrier_count, 2)
        self.assertAlmostEqual(float(result.basin_mass.sum()), 1.0)
        self.assertIn(result.dominant_barrier_index, {0, 1})

    def test_multiwell_series_distinguishes_same_pair_relaxation(self):
        x = np.linspace(-1.0, 1.0, 201)
        time = np.arange(4, dtype=float)
        shape = (x * x - 0.25) ** 2
        amplitudes = np.asarray([0.2, 1.0, 0.8, 0.5])
        potential = amplitudes[:, None] * shape[None, :]
        rho = np.broadcast_to(np.ones_like(x) / x.size, potential.shape)

        result = quantify_multiwell_series(
            x,
            time,
            potential,
            rho,
            overshoot_tolerance=1e-8,
        )

        self.assertEqual(result.overshoot_class, "same_pair_relaxation")
        self.assertAlmostEqual(result.peak_time, 1.0)
        self.assertGreater(result.overshoot_absolute, 0.0)
        self.assertFalse(np.any(result.dominant_switch))

    def test_multiwell_series_retains_incumbent_across_near_ties(self):
        x = np.linspace(-1.0, 1.0, 401)
        base = (x + 0.55) ** 2 * x**2 * (x - 0.55) ** 2
        tilt = x * (x * x - 0.55**2) ** 2
        coefficients = np.asarray([0.005, -0.005, 0.005, -0.005])
        potential = base[None, :] + coefficients[:, None] * tilt[None, :]
        rho = np.broadcast_to(np.ones_like(x) / x.size, potential.shape)
        instantaneous = [
            quantify_multiwell(x, row, rho[0]).dominant_barrier_index
            for row in potential
        ]

        result = quantify_multiwell_series(
            x,
            np.arange(coefficients.size, dtype=float),
            potential,
            rho,
            dominant_score_margin=0.05,
            dominant_switch_persistence=2,
        )

        self.assertEqual(instantaneous, [1, 0, 1, 0])
        self.assertFalse(np.any(result.dominant_switch))
        self.assertEqual(np.unique(result.dominant_left_id).size, 1)
        self.assertEqual(np.unique(result.dominant_right_id).size, 1)

    def test_multiwell_series_rejects_one_record_pair_challenger(self):
        x = np.linspace(-1.0, 1.0, 401)
        base = (x + 0.55) ** 2 * x**2 * (x - 0.55) ** 2
        tilt = x * (x * x - 0.55**2) ** 2
        coefficients = np.asarray([0.2, 0.2, -0.2, 0.2])
        potential = base[None, :] + coefficients[:, None] * tilt[None, :]
        rho = np.broadcast_to(np.ones_like(x) / x.size, potential.shape)

        result = quantify_multiwell_series(
            x,
            np.arange(coefficients.size, dtype=float),
            potential,
            rho,
            dominant_switch_persistence=2,
        )

        self.assertFalse(np.any(result.dominant_switch))

    def test_multiwell_series_confirms_persistent_pair_switch(self):
        x = np.linspace(-1.0, 1.0, 401)
        base = (x + 0.55) ** 2 * x**2 * (x - 0.55) ** 2
        tilt = x * (x * x - 0.55**2) ** 2
        coefficients = np.asarray([0.2, -0.2, -0.2, -0.2])
        potential = base[None, :] + coefficients[:, None] * tilt[None, :]
        rho = np.broadcast_to(np.ones_like(x) / x.size, potential.shape)

        result = quantify_multiwell_series(
            x,
            np.arange(coefficients.size, dtype=float),
            potential,
            rho,
            dominant_switch_persistence=3,
        )

        self.assertEqual(result.well_count.tolist(), [3, 3, 3, 3])
        self.assertEqual(result.dominant_switch.tolist(), [False, False, False, True])
        self.assertNotEqual(
            (result.dominant_left_id[0], result.dominant_right_id[0]),
            (result.dominant_left_id[-1], result.dominant_right_id[-1]),
        )


if __name__ == "__main__":
    unittest.main()

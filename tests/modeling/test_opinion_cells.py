import unittest

import numpy as np

from ehk.metrics.homophily import uniform_concordance_probability
from ehk.modeling.opinion_cells import confidence_geometry


class ConfidenceGeometryTests(unittest.TestCase):
    @staticmethod
    def _grid(size: int) -> np.ndarray:
        dx = 2.0 / size
        return -1.0 + (np.arange(size, dtype=float) + 0.5) * dx

    def test_cell_average_removes_uniform_grid_staircase(self):
        epsilon = 0.45
        exact = uniform_concordance_probability(epsilon)
        for size in (15, 21, 31, 41, 81):
            with self.subTest(size=size):
                weights = np.full(size, 1.0 / size)
                geometry = confidence_geometry(self._grid(size), epsilon)
                measured = float(weights @ geometry.concordance @ weights)
                self.assertAlmostEqual(measured, exact, places=13)

    def test_pair_geometry_has_exchange_symmetry(self):
        geometry = confidence_geometry(self._grid(21), epsilon=0.45)
        np.testing.assert_allclose(
            geometry.concordance,
            geometry.concordance.T,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            geometry.displacement,
            -geometry.displacement.T,
            atol=2e-14,
        )
        self.assertGreaterEqual(float(geometry.concordance.min()), 0.0)
        self.assertLessEqual(float(geometry.concordance.max()), 1.0)

    def test_full_confidence_recovers_exact_cell_moments(self):
        x = self._grid(15)
        geometry = confidence_geometry(x, epsilon=2.0)
        dx = x[1] - x[0]
        expected_second = x * x + dx * dx / 12.0
        np.testing.assert_allclose(geometry.concordance, 1.0, atol=2e-14)
        np.testing.assert_allclose(
            geometry.target_first,
            np.broadcast_to(x, geometry.target_first.shape),
            atol=2e-14,
        )
        np.testing.assert_allclose(
            geometry.target_second,
            np.broadcast_to(expected_second, geometry.target_second.shape),
            atol=2e-14,
        )
        np.testing.assert_allclose(
            geometry.displacement,
            x[None, :] - x[:, None],
            atol=2e-14,
        )

    def test_center_mode_reproduces_legacy_mask(self):
        x = self._grid(15)
        geometry = confidence_geometry(x, epsilon=0.45, mode="center")
        expected = (np.abs(x[:, None] - x[None, :]) <= 0.45).astype(float)
        np.testing.assert_array_equal(geometry.concordance, expected)
        np.testing.assert_allclose(
            geometry.displacement,
            expected * (x[None, :] - x[:, None]),
        )


if __name__ == "__main__":
    unittest.main()

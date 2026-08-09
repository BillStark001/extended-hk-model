import unittest

import numpy as np

from ehk.metrics import calculate_index_series
from ehk.modeling.mesoscopic import KineticParameters, solve
from ehk.modeling.mesoscopic.solver import (
    _recommendation_channels,
    _recommendation_kernel,
)
from ehk.metrics.homophily import uniform_concordance_probability


class KineticSolverTests(unittest.TestCase):
    def test_uniform_homophily_baseline(self):
        self.assertAlmostEqual(
            uniform_concordance_probability(0.45),
            0.399375,
            places=12,
        )

    def test_mass_and_fixed_out_degree_are_conserved(self):
        params = KineticParameters(
            grid_size=31,
            steps=20,
            record_every=2,
            influence=0.05,
            rewiring=0.025,
        )
        trajectory = solve(params)
        np.testing.assert_allclose(trajectory.rho.sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(
            trajectory.edge.sum(axis=(1, 2)),
            params.mean_degree,
            atol=1e-11,
        )
        np.testing.assert_allclose(
            trajectory.edge.sum(axis=2),
            params.mean_degree * trajectory.rho,
            atol=1e-11,
        )

    def test_reflecting_diffusion_preserves_mass_and_positivity(self):
        params = KineticParameters(
            grid_size=31,
            steps=10,
            record_every=1,
            noise_diffusion=1e-4,
        )
        trajectory = solve(params)
        self.assertGreaterEqual(float(trajectory.rho.min()), 0.0)
        self.assertGreaterEqual(float(trajectory.edge.min()), 0.0)
        np.testing.assert_allclose(trajectory.rho.sum(axis=1), 1.0, atol=1e-12)

    def test_uniform_initial_state_has_zero_normalized_indices(self):
        trajectory = solve(
            KineticParameters(grid_size=31, steps=1, record_every=1)
        )
        indices = calculate_index_series(trajectory)
        self.assertAlmostEqual(float(indices.polarization[0]), 0.0, places=12)
        self.assertAlmostEqual(float(indices.homophily[0]), 0.0, places=12)

    def test_m9_uses_one_random_and_nine_ranked_slots(self):
        x = np.linspace(-1.0, 1.0, 31)
        rho = np.full(x.size, 1 / x.size)
        neighbors = np.broadcast_to(rho, (x.size, x.size)).copy()
        random_kernel, core, random_slots, core_slots = (
            _recommendation_channels(
                KineticParameters(recsys="opinionm9"), x, rho, neighbors
            )
        )
        self.assertEqual((random_slots, core_slots), (1, 9))
        m9 = _recommendation_kernel(
            KineticParameters(recsys="opinionm9"), x, rho, neighbors
        )
        np.testing.assert_allclose(m9, 0.1 * random_kernel + 0.9 * core)
        pure = _recommendation_kernel(
            KineticParameters(recsys="opinion"), x, rho, neighbors
        )
        np.testing.assert_allclose(pure, core)
        self.assertGreater(float(np.max(np.abs(m9 - pure))), 1e-6)

    def test_structure_m9_is_distinct_from_pure_structure(self):
        x = np.linspace(-1.0, 1.0, 21)
        rho = np.full(x.size, 1 / x.size)
        neighbors = np.broadcast_to(rho, (x.size, x.size)).copy()
        neighbors[0] = 0.0
        neighbors[0, :4] = 0.25
        neighbors[-1] = 0.0
        neighbors[-1, -4:] = 0.25
        pure = _recommendation_kernel(
            KineticParameters(recsys="structure"), x, rho, neighbors
        )
        m9 = _recommendation_kernel(
            KineticParameters(recsys="structurem9"), x, rho, neighbors
        )
        self.assertGreater(float(np.max(np.abs(m9 - pure))), 1e-6)

    def test_baseline_parameters_produce_opposite_path_orderings(self):
        results = {}
        for label, influence in (("pbs", 0.05), ("sbp", 0.005)):
            trajectory = solve(
                KineticParameters(
                    grid_size=41,
                    steps=1000,
                    record_every=10,
                    influence=influence,
                    rewiring=0.025,
                    noise_diffusion=1e-5,
                )
            )
            indices = calculate_index_series(trajectory)

            def crossing(values):
                found = np.flatnonzero(values >= 0.5)
                return indices.time[found[0]] if found.size else np.inf

            results[label] = (
                crossing(indices.polarization),
                crossing(indices.homophily),
                indices.pathway,
            )

        self.assertLess(results["pbs"][0], results["pbs"][1])
        self.assertGreater(results["sbp"][0], results["sbp"][1])
        self.assertLess(results["pbs"][2], results["sbp"][2])


if __name__ == "__main__":
    unittest.main()

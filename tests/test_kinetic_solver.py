import unittest

import numpy as np

from works.kinetic import KineticParameters, calculate_index_series, solve
from stats.homophily import uniform_concordance_probability


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

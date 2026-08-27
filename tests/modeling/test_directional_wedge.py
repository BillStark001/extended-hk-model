import unittest

import numpy as np

from ehk.modeling.mesoscopic import KineticParameters, solve
from ehk.modeling.mesoscopic.directional_wedge import (
    independent_directional_wedge,
)
from ehk.modeling.mesoscopic.solver import _recommendation_kernel


class DirectionalWedgeTests(unittest.TestCase):
    def test_independent_uniform_wedge_recovers_random_candidate_mass(self):
        size = 11
        rho = np.full(size, 1.0 / size)
        edge = 15.0 * np.outer(rho, rho)
        state = independent_directional_wedge(rho, edge)
        state.validate(size)

        score = state.union_score_mass()
        normalized = score / score.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(
            normalized, np.broadcast_to(rho, score.shape), atol=1e-14
        )

    def test_l1_kernel_uses_directional_wedge_score_mass(self):
        size = 5
        x = np.linspace(-1.0, 1.0, size)
        rho = np.full(size, 1.0 / size)
        neighbors = np.broadcast_to(rho, (size, size)).copy()
        score = np.ones((size, size))
        score[0, -1] = 20.0

        kernel = _recommendation_kernel(
            KineticParameters(
                grid_size=11,
                recsys="structure_random_l1",
                recommendation_steepness=1.0,
            ),
            x,
            rho,
            neighbors,
            structural_score=score,
        )
        self.assertGreater(kernel[0, -1], 0.8)
        self.assertAlmostEqual(float(kernel[0].sum()), 1.0, places=14)

    def test_l1_solver_preserves_pair_invariants_and_records_scores(self):
        params = KineticParameters(
            grid_size=11,
            steps=5,
            record_every=1,
            recsys="structure_random_l1",
            rewiring=0.05,
            influence=0.05,
            recommendation_steepness=1.0,
        )
        trajectory = solve(params)
        self.assertIsNotNone(trajectory.structural_score)
        assert trajectory.structural_score is not None
        self.assertEqual(trajectory.structural_score.shape, (6, 11, 11))
        self.assertGreaterEqual(float(trajectory.structural_score.min()), 0.0)
        np.testing.assert_allclose(trajectory.rho.sum(axis=1), 1.0, atol=1e-12)
        np.testing.assert_allclose(
            trajectory.edge.sum(axis=2),
            params.mean_degree * trajectory.rho,
            atol=1e-10,
        )

    def test_l0_remains_available_without_l1_storage(self):
        trajectory = solve(
            KineticParameters(
                grid_size=11,
                steps=2,
                record_every=1,
                recsys="structure_random_l0",
            )
        )
        self.assertIsNone(trajectory.structural_score)

    def test_l1_rejects_unclosed_higher_score_power(self):
        with self.assertRaisesRegex(ValueError, "first score moment"):
            solve(
                KineticParameters(
                    grid_size=11,
                    steps=1,
                    recsys="structure_random_l1",
                    recommendation_steepness=4.0,
                )
            )

    def test_l1_mean_power_closure_supports_higher_steepness(self):
        size = 5
        x = np.linspace(-1.0, 1.0, size)
        rho = np.full(size, 1.0 / size)
        neighbors = np.broadcast_to(rho, (size, size)).copy()
        score_mass = np.ones((size, size))
        score_mass[0, -1] = 2.0

        rows = []
        for steepness in (1.0, 4.0):
            rows.append(
                _recommendation_kernel(
                    KineticParameters(
                        grid_size=11,
                        recsys="structure_random_l1_mean_power",
                        recommendation_steepness=steepness,
                    ),
                    x,
                    rho,
                    neighbors,
                    structural_score=score_mass,
                )[0]
            )
        self.assertGreater(rows[1][-1], rows[0][-1])

        trajectory = solve(
            KineticParameters(
                grid_size=11,
                steps=2,
                recsys="structure_random_l1_mean_power",
                recommendation_steepness=4.0,
            )
        )
        self.assertIsNotNone(trajectory.structural_score)


if __name__ == "__main__":
    unittest.main()

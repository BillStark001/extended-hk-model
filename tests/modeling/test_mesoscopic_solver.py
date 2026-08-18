import unittest

import numpy as np

from ehk.metrics import calculate_index_series
from ehk.modeling.mesoscopic import KineticParameters, solve
from ehk.modeling.mesoscopic.solver import (
    _advance_transport_diffusion,
    _finite_volume_system,
    _recommendation_channels,
    _recommendation_kernel,
    _validate_state,
)
from ehk.metrics.density_indices import _DensityIndexCalculator
from ehk.metrics.homophily import uniform_concordance_probability


class KineticSolverTests(unittest.TestCase):
    def test_solver_grid_uses_finite_volume_cell_centers(self):
        trajectory = solve(
            KineticParameters(grid_size=25, steps=1, record_every=1)
        )
        dx = 2.0 / 25
        self.assertAlmostEqual(float(trajectory.x[0]), -1.0 + dx / 2)
        self.assertAlmostEqual(float(trajectory.x[-1]), 1.0 - dx / 2)
        np.testing.assert_allclose(np.diff(trajectory.x), dx, atol=1e-15)

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

    def test_finite_volume_operator_has_zero_boundary_flux(self):
        velocity = np.asarray([-0.4, -0.1, 0.2, 0.5])
        lower, diagonal, upper = _finite_volume_system(
            velocity, diffusion=0.03, dx=0.25, dt=0.2
        )
        system = np.diag(diagonal)
        system += np.diag(lower, k=-1)
        system += np.diag(upper, k=1)
        # Column sums of I-dt*L equal one iff the finite-volume generator has
        # no boundary leakage and conserves total mass.
        np.testing.assert_allclose(system.sum(axis=0), 1.0, atol=1e-14)

        rho = np.asarray([0.7, 0.2, 0.1, 0.0])
        edge = 15.0 * np.outer(rho, rho)
        rho_next, edge_next = _advance_transport_diffusion(
            rho, edge, velocity, diffusion=0.03, dx=0.25, dt=0.2
        )
        self.assertGreaterEqual(float(rho_next.min()), 0.0)
        self.assertGreaterEqual(float(edge_next.min()), 0.0)
        self.assertAlmostEqual(float(rho_next.sum()), 1.0, places=13)
        np.testing.assert_allclose(
            edge_next.sum(axis=1), 15.0 * rho_next, atol=1e-12
        )

    def test_full_confidence_pde_converges_to_affine_contraction(self):
        errors = []
        for grid_size, dt in ((41, 0.02), (81, 0.01)):
            steps = round(1.0 / dt)
            trajectory = solve(
                KineticParameters(
                    grid_size=grid_size,
                    epsilon=2.0,
                    influence=0.2,
                    rewiring=0.0,
                    noise_diffusion=0.0,
                    dt=dt,
                    steps=steps,
                    record_every=steps,
                )
            )
            initial_variance = np.sum(
                trajectory.rho[0] * trajectory.x * trajectory.x
            )
            exact_variance = initial_variance * np.exp(-0.4)
            numerical_variance = np.sum(
                trajectory.rho[-1] * trajectory.x * trajectory.x
            )
            errors.append(abs(float(numerical_variance - exact_variance)))
        self.assertLess(errors[1], 0.55 * errors[0])

    def test_invalid_explicit_time_step_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dt \\* rewiring"):
            solve(KineticParameters(dt=2.0, rewiring=0.75))

    def test_implicit_transport_does_not_impose_a_courant_rejection(self):
        trajectory = solve(
            KineticParameters(
                grid_size=31,
                influence=0.75,
                rewiring=0.0,
                dt=2.0,
                steps=1,
                record_every=1,
            )
        )
        self.assertGreaterEqual(float(trajectory.rho.min()), 0.0)
        self.assertAlmostEqual(float(trajectory.rho[-1].sum()), 1.0, places=13)

    def test_nonfinite_and_noninteger_parameters_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "must be finite"):
            solve(KineticParameters(noise_diffusion=float("nan")))
        with self.assertRaisesRegex(ValueError, "must be integers"):
            solve(KineticParameters(grid_size=81.0))

    def test_invariant_check_does_not_project_material_errors(self):
        rho = np.asarray([0.5, 0.5])
        edge = np.asarray([[4.0, 3.0], [3.0, 4.5]])
        with self.assertRaisesRegex(FloatingPointError, "invariants"):
            _validate_state(rho, edge, mean_degree=15.0)

    def test_density_kde_axis_retains_boundary_tails(self):
        calculator = _DensityIndexCalculator(
            np.linspace(-1.0, 1.0, 31), epsilon=0.45, mean_degree=15.0
        )
        self.assertLess(float(calculator.distance_axis[0]), 0.0)
        self.assertGreater(float(calculator.distance_axis[-1]), 2.0)

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

    def test_opinion_random_uses_tent_score_and_steepness(self):
        x = np.asarray([-0.5, 0.0, 0.25, 0.5])
        rho = np.full(x.size, 1 / x.size)
        neighbors = np.broadcast_to(rho, (x.size, x.size)).copy()
        params = KineticParameters(
            recsys="opinion_random",
            opinion_tolerance=0.5,
            recommendation_steepness=2.0,
        )
        kernel = _recommendation_kernel(params, x, rho, neighbors)
        expected_score = np.maximum(
            1 - np.abs(x[None, :] - x[:, None]) / 0.5, 0
        ) ** 2
        expected = expected_score * rho[None, :]
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(kernel, expected, atol=1e-15)

    def test_structure_random_l0_uses_steepness_power(self):
        x = np.linspace(-1.0, 1.0, 5)
        rho = np.full(x.size, 1 / x.size)
        neighbors = np.asarray([
            [0.6, 0.4, 0.0, 0.0, 0.0],
            [0.3, 0.5, 0.2, 0.0, 0.0],
            [0.0, 0.2, 0.6, 0.2, 0.0],
            [0.0, 0.0, 0.2, 0.5, 0.3],
            [0.0, 0.0, 0.0, 0.4, 0.6],
        ])
        params = KineticParameters(
            recsys="structure_random_l0",
            recommendation_steepness=2.0,
        )
        kernel = _recommendation_kernel(params, x, rho, neighbors)
        expected = (neighbors @ neighbors.T) ** 2 * rho[None, :]
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(kernel, expected, atol=1e-15)

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

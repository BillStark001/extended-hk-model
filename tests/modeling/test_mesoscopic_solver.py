import unittest

import numpy as np

from ehk.metrics import calculate_index_series
from ehk.metrics.density_indices import _DensityIndexCalculator
from ehk.metrics.homophily import uniform_concordance_probability
from ehk.modeling.mesoscopic import KineticParameters, solve
from ehk.modeling.mesoscopic.solver import (
    _advance_transport_diffusion,
    _deffuant_transition_from_selection,
    _finite_volume_system,
    _hk_transition_from_selection,
    _moment_matched_transition,
    _recommendation_channels,
    _recommendation_kernel,
    _validate_state,
)


class KineticSolverTests(unittest.TestCase):
    def test_solver_grid_uses_finite_volume_cell_centers(self):
        trajectory = solve(KineticParameters(grid_size=25, steps=1, record_every=1))
        dx = 2.0 / 25
        self.assertAlmostEqual(float(trajectory.x[0]), -1.0 + dx / 2)
        self.assertAlmostEqual(float(trajectory.x[-1]), 1.0 - dx / 2)
        np.testing.assert_allclose(np.diff(trajectory.x), dx, atol=1e-15)

    def test_custom_record_steps_include_endpoints(self):
        trajectory = solve(
            KineticParameters(grid_size=25, steps=10, record_every=7),
            record_steps=(1, 2, 5),
        )
        np.testing.assert_array_equal(trajectory.time, [0, 1, 2, 5, 10])

    def test_invalid_custom_record_step_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "record_steps"):
            solve(KineticParameters(steps=10), record_steps=(1, 11))

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
        np.testing.assert_allclose(edge_next.sum(axis=1), 15.0 * rho_next, atol=1e-12)

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
            initial_variance = np.sum(trajectory.rho[0] * trajectory.x * trajectory.x)
            exact_variance = initial_variance * np.exp(-0.4)
            numerical_variance = np.sum(
                trajectory.rho[-1] * trajectory.x * trajectory.x
            )
            errors.append(abs(float(numerical_variance - exact_variance)))
        self.assertLess(errors[1], 0.55 * errors[0])

    def test_invalid_explicit_time_step_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dt \\* rewiring"):
            solve(KineticParameters(dt=2.0, rewiring=0.75))

    def test_invalid_opinion_dynamics_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "opinion dynamics"):
            solve(KineticParameters(dynamics="not-a-rule"))

    def test_invalid_opinion_method_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "opinion method"):
            solve(KineticParameters(opinion_method="not-a-method"))

    def test_deffuant_transition_preserves_expected_first_moment(self):
        x = -1.0 + (np.arange(11) + 0.5) * (2.0 / 11)
        selection = np.zeros((x.size, x.size))
        for source in range(x.size):
            selection[source, source] = 0.25
            selection[source, x.size - source - 1] += 0.75
        influence = 0.37
        transition = _deffuant_transition_from_selection(x, influence, selection)
        np.testing.assert_allclose(transition.sum(axis=1), 1.0, atol=1e-15)
        expected = x + influence * (selection @ x - x)
        np.testing.assert_allclose(transition @ x, expected, atol=1e-14)

        hk_transition = _hk_transition_from_selection(x, influence, selection)
        np.testing.assert_allclose(hk_transition @ x, expected, atol=1e-14)

    def test_moment_closure_matches_discrete_reference_moments(self):
        x = np.linspace(-1.0, 1.0, 31)
        rng = np.random.default_rng(20260828)
        reference = rng.random((x.size, x.size))
        reference /= reference.sum(axis=1, keepdims=True)
        closure = _moment_matched_transition(x, reference)

        self.assertLessEqual(int(np.diff(closure.indptr).max()), 4)
        np.testing.assert_allclose(
            np.asarray(closure.sum(axis=1)).ravel(), 1.0, atol=1e-14
        )
        np.testing.assert_allclose(closure @ x, reference @ x, atol=5e-12)
        np.testing.assert_allclose(closure @ (x * x), reference @ (x * x), atol=5e-12)

    def test_hk_full_and_moment_closed_updates_coincide(self):
        common = {
            "grid_size": 41,
            "steps": 30,
            "record_every": 5,
            "epsilon": 0.4,
            "influence": 0.1,
            "rewiring": 0.01,
            "noise_diffusion": 1e-5,
            "recsys": "opinion_random",
            "recommendation_steepness": 4.0,
            "dynamics": "hk",
        }
        jump = solve(KineticParameters(opinion_method="nonlocal_jump", **common))
        closure = solve(KineticParameters(opinion_method="fokker_planck", **common))
        np.testing.assert_allclose(closure.rho, jump.rho, atol=2e-13)
        np.testing.assert_allclose(closure.edge, jump.edge, atol=2e-12)

    def test_deffuant_solver_conserves_mass_and_fixed_out_degree(self):
        for dynamics in ("hk", "deffuant"):
            for method in ("fokker_planck", "nonlocal_jump"):
                with self.subTest(dynamics=dynamics, method=method):
                    parameters = KineticParameters(
                        dynamics=dynamics,
                        opinion_method=method,
                        grid_size=21,
                        steps=8,
                        record_every=1,
                        influence=0.2,
                        rewiring=0.05,
                        noise_diffusion=1e-5,
                    )
                    trajectory = solve(parameters)
                    np.testing.assert_allclose(
                        trajectory.rho.sum(axis=1), 1.0, atol=1e-12
                    )
                    np.testing.assert_allclose(
                        trajectory.edge.sum(axis=2),
                        parameters.mean_degree * trajectory.rho,
                        atol=1e-11,
                    )
                    self.assertGreaterEqual(float(trajectory.rho.min()), 0.0)

    def test_hk_and_deffuant_density_updates_are_distinct(self):
        common = {
            "grid_size": 41,
            "steps": 3,
            "record_every": 1,
            "epsilon": 2.0,
            "influence": 0.4,
            "rewiring": 0.0,
            "noise_diffusion": 0.0,
        }
        for method in ("fokker_planck", "nonlocal_jump"):
            hk = solve(
                KineticParameters(dynamics="hk", opinion_method=method, **common)
            )
            deffuant = solve(
                KineticParameters(dynamics="deffuant", opinion_method=method, **common)
            )
            self.assertGreater(
                float(np.max(np.abs(hk.rho[-1] - deffuant.rho[-1]))),
                1e-3,
            )

    def test_deffuant_second_moment_exceeds_hk_by_conditional_variance(self):
        common = {
            "grid_size": 21,
            "steps": 1,
            "record_every": 1,
            "epsilon": 0.8,
            "influence": 0.2,
            "rewiring": 0.0,
        }
        hk = solve(KineticParameters(dynamics="hk", **common))
        deffuant = solve(KineticParameters(dynamics="deffuant", **common))
        np.testing.assert_allclose(hk.endogenous_diffusion, 0.0, atol=0.0)
        self.assertGreater(
            float(np.max(deffuant.endogenous_diffusion - hk.endogenous_diffusion)),
            0.0,
        )
        np.testing.assert_allclose(
            hk.displacement_variance[0],
            deffuant.displacement_variance[0],
            atol=1e-15,
        )
        coefficient = 0.5 * common["influence"] ** 2
        np.testing.assert_allclose(
            deffuant.endogenous_diffusion[0],
            coefficient * deffuant.displacement_variance[0],
            atol=1e-15,
        )
        np.testing.assert_allclose(
            deffuant.displacement_second_moment[0],
            deffuant.displacement_variance[0]
            + deffuant.velocity[0] ** 2 / common["influence"] ** 2,
            atol=1e-15,
        )

    def test_deffuant_retains_l1_transport_capability(self):
        for dynamics in ("hk", "deffuant"):
            for method in ("fokker_planck", "nonlocal_jump"):
                trajectory = solve(
                    KineticParameters(
                        dynamics=dynamics,
                        opinion_method=method,
                        recsys="structure_random_l1_mean_power",
                        recommendation_steepness=4.0,
                        grid_size=11,
                        steps=2,
                        record_every=1,
                    )
                )
                self.assertIsNotNone(trajectory.structural_score)
                self.assertEqual(trajectory.structural_score.shape, (3, 11, 11))

    def test_discrete_opinion_step_rejects_oversized_compromise(self):
        with self.assertRaisesRegex(ValueError, "dt \\* influence"):
            solve(
                KineticParameters(
                    grid_size=31,
                    influence=0.75,
                    rewiring=0.0,
                    dt=2.0,
                    steps=1,
                    record_every=1,
                )
            )

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
        trajectory = solve(KineticParameters(grid_size=31, steps=1, record_every=1))
        indices = calculate_index_series(trajectory)
        self.assertAlmostEqual(float(indices.polarization[0]), 0.0, places=12)
        self.assertAlmostEqual(float(indices.homophily[0]), 0.0, places=12)

    def test_m9_uses_one_random_and_nine_ranked_slots(self):
        x = np.linspace(-1.0, 1.0, 31)
        rho = np.full(x.size, 1 / x.size)
        neighbors = np.broadcast_to(rho, (x.size, x.size)).copy()
        random_kernel, core, random_slots, core_slots = _recommendation_channels(
            KineticParameters(recsys="opinionm9"), x, rho, neighbors
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
        expected_score = np.maximum(1 - np.abs(x[None, :] - x[:, None]) / 0.5, 0) ** 2
        expected = expected_score * rho[None, :]
        expected /= expected.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(kernel, expected, atol=1e-15)

    def test_structure_random_l0_uses_steepness_power(self):
        x = np.linspace(-1.0, 1.0, 5)
        rho = np.full(x.size, 1 / x.size)
        neighbors = np.asarray(
            [
                [0.6, 0.4, 0.0, 0.0, 0.0],
                [0.3, 0.5, 0.2, 0.0, 0.0],
                [0.0, 0.2, 0.6, 0.2, 0.0],
                [0.0, 0.0, 0.2, 0.5, 0.3],
                [0.0, 0.0, 0.0, 0.4, 0.6],
            ]
        )
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

            def crossing(values, current_indices=indices):
                found = np.flatnonzero(values >= 0.5)
                return current_indices.time[found[0]] if found.size else np.inf

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

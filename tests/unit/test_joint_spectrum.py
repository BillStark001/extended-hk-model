import unittest

import numpy as np
from scipy.linalg import expm

from theory.mesoscopic.joint_spectrum_operator import (
    JointSpectrumParameters,
    affine_jacobian_bases,
    base_state,
    build_joint_grid,
    discordant_derivative,
    eigenspectrum,
    joint_jacobian,
    pair_rhs,
    physical_transform,
    recommendation_slots,
    time_ordered_spectrum,
)


class JointSpectrumTests(unittest.TestCase):
    def test_weighted_random_kernels_are_normalized_and_steepness_resolved(self):
        parameters = JointSpectrumParameters(grid_size=32)
        opinion_low = build_joint_grid(
            recommendation_slots("opinion_random", steepness=1), parameters
        )
        opinion_high = build_joint_grid(
            recommendation_slots("opinion_random", steepness=4), parameters
        )
        structure_low = build_joint_grid(
            recommendation_slots("structure_random_l0", steepness=1), parameters
        )
        structure_high = build_joint_grid(
            recommendation_slots("structure_random_l0", steepness=4), parameters
        )
        for grid in (opinion_low, opinion_high, structure_low, structure_high):
            np.testing.assert_allclose(grid.opinion.sum(axis=1) * grid.dx, 1.0)
        self.assertGreater(
            float(np.linalg.norm(opinion_low.opinion - opinion_high.opinion)), 1e-3
        )
        # A random product graph has constant expected overlap, so both L0
        # steepnesses intentionally share the same unperturbed kernel.
        np.testing.assert_allclose(structure_low.opinion, structure_high.opinion)

    def test_structure_random_steepness_changes_pair_tangent(self):
        parameters = JointSpectrumParameters(grid_size=16)
        matrices = []
        for steepness in (1.0, 4.0):
            slots = recommendation_slots(
                "structure_random_l0", steepness=steepness
            )
            grid = build_joint_grid(slots, parameters)
            matrices.append(
                joint_jacobian(
                    slots=slots,
                    alpha=0.05,
                    rewiring=0.05,
                    mode=2,
                    discordant_fraction=0.5 * grid.initial_discordant,
                    parameters=parameters,
                    grid=grid,
                )
            )
        self.assertGreater(float(np.linalg.norm(matrices[0] - matrices[1])), 1e-3)

    def test_random_q0_product_branch_recovers_scalar_spectrum(self):
        parameters = JointSpectrumParameters(grid_size=64)
        slots = recommendation_slots("random")
        grid = build_joint_grid(slots, parameters)
        mode = 2
        matrix = joint_jacobian(
            slots=slots,
            alpha=0.05,
            rewiring=0.0,
            mode=mode,
            discordant_fraction=grid.initial_discordant,
            parameters=parameters,
            grid=grid,
        )
        spectrum = eigenspectrum(matrix, parameters=parameters, grid=grid)
        measured = float(spectrum.eigenvalues[spectrum.order[0]].real)
        kappa = 2 * np.pi * mode / parameters.length
        z = kappa * parameters.epsilon
        expected = 0.05 * (np.sin(z) - z * np.cos(z)) / z
        expected -= parameters.diffusion * kappa**2
        self.assertAlmostEqual(measured, expected, delta=5e-4)

    def test_q_positive_product_state_moves_along_rewiring_orbit(self):
        parameters = JointSpectrumParameters(grid_size=32, diffusion=0.0)
        slots = recommendation_slots("opinionm9")
        grid = build_joint_grid(slots, parameters)
        rho, edge, _neighbor = base_state(
            grid,
            parameters,
            discordant_fraction=grid.initial_discordant,
        )
        rho_rhs, edge_rhs = pair_rhs(
            rho,
            edge,
            slots=slots,
            alpha=0.0,
            rewiring=0.3,
            parameters=parameters,
            grid=grid,
        )
        measured = float(
            ((~grid.concordant) * edge_rhs.real).sum()
            * grid.dx**2
            / parameters.mean_degree
        )
        expected = discordant_derivative(
            grid.initial_discordant,
            rewiring=0.3,
            parameters=parameters,
            grid=grid,
        )
        self.assertLess(float(np.linalg.norm(rho_rhs)), 1e-10)
        self.assertGreater(float(np.linalg.norm(edge_rhs)), 1e-5)
        self.assertAlmostEqual(measured, expected, delta=1e-11)

    def test_alpha_q_affine_reconstruction_matches_direct_jacobian(self):
        parameters = JointSpectrumParameters(grid_size=16)
        slots = recommendation_slots("random")
        grid = build_joint_grid(slots, parameters)
        discordant = 0.5 * grid.initial_discordant
        base, alpha_matrix, q_matrix = affine_jacobian_bases(
            slots=slots,
            mode=4,
            discordant_fraction=discordant,
            parameters=parameters,
            grid=grid,
        )
        reconstructed = base + 0.05 * alpha_matrix + 0.3 * q_matrix
        direct = joint_jacobian(
            slots=slots,
            alpha=0.05,
            rewiring=0.3,
            mode=4,
            discordant_fraction=discordant,
            parameters=parameters,
            grid=grid,
        )
        relative_error = np.linalg.norm(reconstructed - direct) / np.linalg.norm(
            direct
        )
        self.assertLess(float(relative_error), 1e-8)

    def test_q0_time_ordering_equals_constant_matrix_exponential(self):
        parameters = JointSpectrumParameters(grid_size=16)
        slots = recommendation_slots("opinion")
        grid = build_joint_grid(slots, parameters)
        base, alpha_matrix, q_matrix = affine_jacobian_bases(
            slots=slots,
            mode=2,
            discordant_fraction=grid.initial_discordant,
            parameters=parameters,
            grid=grid,
        )
        horizon = 7.0
        result = time_ordered_spectrum(
            fractions=np.asarray([1.0]),
            base_table=base[None, :, :],
            alpha_table=alpha_matrix[None, :, :],
            q_table=q_matrix[None, :, :],
            alpha=0.05,
            rewiring=0.0,
            horizon=horizon,
            time_steps=13,
            parameters=parameters,
            grid=grid,
        )
        transform, inverse = physical_transform(parameters, grid)
        physical_matrix = transform @ (base + 0.05 * alpha_matrix) @ inverse
        direct = expm(physical_matrix * horizon)
        expected = np.log(np.linalg.svd(direct, compute_uv=False)) / horizon
        np.testing.assert_allclose(
            result.finite_time_rates,
            expected,
            rtol=1e-11,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()

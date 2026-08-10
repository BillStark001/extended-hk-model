import unittest

import numpy as np

from theory.mesoscopic.spectrum_steady_states import (
    PeriodicParameters,
    _build_operators,
    _initial_density,
    _random_growth_rate,
    _rk4_step,
    _solver_growth,
)


class SpectrumSteadyStateTests(unittest.TestCase):
    def test_periodic_solver_reproduces_selected_linear_growth(self):
        params = PeriodicParameters(grid_size=256)
        operators = _build_operators(params)
        kappa = 2 * np.pi * params.mode / params.length
        expected = _random_growth_rate(
            kappa,
            epsilon=params.epsilon,
            influence=params.influence,
            diffusion=params.diffusion,
        )
        measured = _solver_growth("random", params, operators)
        self.assertAlmostEqual(measured, expected, delta=5e-4)

    def test_rk4_step_conserves_mass_and_mode_symmetry(self):
        params = PeriodicParameters(grid_size=128, dt=0.01)
        operators = _build_operators(params)
        density = _initial_density(params, operators)
        for _ in range(10):
            density = _rk4_step(density, "opinion", params, operators)

        self.assertAlmostEqual(
            float(density.sum() * operators.dx),
            1.0,
            places=13,
        )
        spectrum = np.fft.fft(density)
        self.assertLess(float(np.max(np.abs(spectrum[~operators.symmetry]))), 1e-12)


if __name__ == "__main__":
    unittest.main()

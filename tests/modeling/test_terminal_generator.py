import unittest

import numpy as np

from ehk.modeling.terminal_generator import (
    TerminalGeneratorParameters,
    initialize_terminal_state,
    opinion_grid,
    recommendation_kernel,
    run_terminal_generator,
    validate_terminal_state,
)


class TerminalGeneratorFactorialTests(unittest.TestCase):
    def _parameters(self, state_level: str, timescale: str):
        return TerminalGeneratorParameters(
            population=60,
            grid_size=7,
            mean_degree=5,
            recsys_count=4,
            influence=0.02,
            rewiring=0.2,
            recsys="structure_random",
            state_level=state_level,
            timescale=timescale,
            max_steps=2,
            seed=17,
            fast_slow_ratio_threshold=5.0,
            fast_max_steps=10,
            fast_zero_checks=2,
        )

    def test_state_level_controls_only_score_moment_storage(self):
        pair_parameters = self._parameters("pair", "unsplit")
        lifted_parameters = self._parameters("score_moments", "unsplit")
        pair = initialize_terminal_state(pair_parameters)
        lifted = initialize_terminal_state(lifted_parameters)
        np.testing.assert_allclose(pair.rho, lifted.rho)
        np.testing.assert_allclose(pair.edge, lifted.edge)
        self.assertFalse(pair.lifted)
        self.assertTrue(lifted.lifted)
        validate_terminal_state(pair_parameters, pair)
        validate_terminal_state(lifted_parameters, lifted)

    def test_all_four_factorial_cells_execute(self):
        for state_level in ("pair", "score_moments"):
            for timescale in ("unsplit", "fast_slow"):
                with self.subTest(state_level=state_level, timescale=timescale):
                    result = run_terminal_generator(
                        self._parameters(state_level, timescale)
                    )
                    validate_terminal_state(result.parameters, result.state)
                    self.assertIn(
                        result.category, {"k1", "k2", "k3", "k4plus", "censored"}
                    )
                    self.assertEqual(result.fast_slow_applied, timescale == "fast_slow")

    def test_pair_and_score_moment_structure_kernels_are_distinct(self):
        pair_parameters = self._parameters("pair", "unsplit")
        lifted_parameters = self._parameters("score_moments", "unsplit")
        pair = initialize_terminal_state(pair_parameters)
        lifted = initialize_terminal_state(lifted_parameters)
        x = opinion_grid(pair_parameters.grid_size)
        pair_kernel, _, _ = recommendation_kernel(pair_parameters, x, pair)
        lifted_kernel, _, _ = recommendation_kernel(lifted_parameters, x, lifted)
        self.assertGreater(float(np.max(np.abs(pair_kernel - lifted_kernel))), 1e-8)
        np.testing.assert_allclose(pair_kernel.sum(axis=1), 1.0, atol=1e-14)
        np.testing.assert_allclose(lifted_kernel.sum(axis=1), 1.0, atol=1e-14)


if __name__ == "__main__":
    unittest.main()

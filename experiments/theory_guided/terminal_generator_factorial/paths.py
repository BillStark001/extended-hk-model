"""Path registrations for the terminal-generator factorial workflow."""

from ehk.common.settings import register_path

OUTPUT_DIR = register_path(
    "EHK_TERMINAL_GENERATOR_FACTORIAL_OUTPUT_DIR",
    "outputs/experiments/theory_guided/terminal_generator_factorial",
)

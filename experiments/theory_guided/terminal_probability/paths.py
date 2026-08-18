"""Path registrations for the terminal-probability workflow."""

from ehk.common.settings import register_path


OUTPUT_DIR = register_path(
    "EHK_TERMINAL_PROBABILITY_OUTPUT_DIR",
    "outputs/experiments/theory_guided/terminal_probability",
)

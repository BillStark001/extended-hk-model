"""Path registrations for the terminal-generator factorial workflow."""

from ehk.common.settings import register_path

OUTPUT_DIR = register_path(
    "EHK_BASE_TERMINAL_OUTPUT_DIR",
    "outputs/experiments/theory_guided/base_terminal_probability",
)
